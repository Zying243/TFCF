import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    """
    优化版RevIN模块：保留核心功能，修复稳定性问题，增强MAE与MSE的平衡能力
    """

    def __init__(self, num_features: int, eps: float = 1e-5, hidden_dim: int = 64,
                 num_layers: int = 2, affine: bool = True, subtract_last: bool = False,
                 use_dyt: bool = False, dyt_mode: str = 'augment',
                 dyt_alpha_init: float = 0.5, dyt_gamma_init: float = 0.1,
                 use_adaptive_dyt: bool = False, activation_before_dyt: str = None,
                 use_dynamic_balance: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.use_dyt = use_dyt
        self.dyt_mode = dyt_mode
        self.use_adaptive_dyt = use_adaptive_dyt
        self.activation_before_dyt = activation_before_dyt
        self.use_dynamic_balance = use_dynamic_balance

        # 归一化流网络
        self.normalize_flow = self._build_flow(num_layers, num_features, hidden_dim)
        # 反归一化流网络
        self.denormalize_flow = self._build_flow(num_layers, num_features, hidden_dim)

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        # DyT 参数
        if self.use_dyt:
            self.dyt_alpha = nn.Parameter(torch.full((num_features,), dyt_alpha_init))
            self.dyt_gamma = nn.Parameter(torch.full((num_features,), dyt_gamma_init))
            self.dyt_beta = nn.Parameter(torch.zeros(num_features))
            if self.use_adaptive_dyt:
                self.dyt_scale = nn.Parameter(torch.ones(num_features) * 0.5)
            if activation_before_dyt:
                self.pre_dyt_act = self._get_activation(activation_before_dyt)
            else:
                self.pre_dyt_act = None

        # 动态平衡权重
        if self.use_dynamic_balance and self.use_dyt and self.dyt_mode in ['residual', 'augment']:
            self.balance_weight = nn.Parameter(torch.tensor(0.8))

    def _build_flow(self, num_layers, num_features, hidden_dim):
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(num_features, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_features),
                nn.LayerNorm(num_features)
            ])
        return nn.Sequential(*layers)

    def _get_activation(self, activation_name):
        if activation_name.lower() == 'gelu':
            return nn.GELU()
        elif activation_name.lower() == 'relu':
            return nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Expected 'norm' or 'denorm'.")

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = 1
        with torch.no_grad():
            if self.subtract_last:
                self.mean = x[:, -1:, :].detach()
            else:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True)
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_dyt:
            if self.dyt_mode == 'replace':
                y = self._apply_dyt(x)
                return y * 0.5
            elif self.dyt_mode in ['augment', 'residual']:
                z = (x - self.mean) / self.stdev
                z_dyt = self._apply_dyt(z)
                if self.use_dynamic_balance:
                    return self.balance_weight * z + (1 - self.balance_weight) * z_dyt
                else:
                    return z + z_dyt * 0.3

        z = (x - self.mean) / self.stdev
        residual = self.normalize_flow(z.reshape(-1, self.num_features)).reshape(z.shape)
        z_flow = z + residual * 0.5

        if self.affine:
            z_flow = z_flow * self.affine_weight + self.affine_bias

        return z_flow

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_dyt:
            if self.dyt_mode == 'replace':
                x_inv = self._apply_dyt_inverse(x)
                return x_inv
            elif self.dyt_mode == 'augment':
                x_inv = self._apply_dyt_inverse(x)
                return x_inv * self.stdev + self.mean
            elif self.dyt_mode == 'residual':
                x_dyt_inv = self._apply_dyt_inverse(x)
                x_normal = x * self.stdev + self.mean
                return x_normal + x_dyt_inv

        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        residual = self.denormalize_flow(x.reshape(-1, self.num_features))
        residual = residual.reshape(x.shape)
        x_flow = x - residual
        return x_flow * self.stdev + self.mean

    def _apply_dyt(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_dyt_act:
            x = self.pre_dyt_act(x)
        alpha = torch.clamp(self.dyt_alpha, 0.01, 1.0)
        y = torch.tanh(alpha * x)
        if self.use_adaptive_dyt:
            y = self.dyt_scale * y
        return self.dyt_gamma * y + self.dyt_beta * 0.1

    def _apply_dyt_inverse(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = torch.clamp((x - self.dyt_beta * 0.1) / self.dyt_gamma, -0.99, 0.99)
        if self.use_adaptive_dyt:
            x_clamped = x_clamped / (self.dyt_scale + self.eps)
        return torch.atanh(x_clamped) / torch.clamp(self.dyt_alpha, 0.01, 1.0)