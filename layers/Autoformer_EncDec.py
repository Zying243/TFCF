import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
import math

# 尝试导入最新的加速库
try:
    import xformers.ops as xops

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

try:
    from flash_attn.flash_attn2 import flash_attn_with_kvcache
    from flash_attn.ops.fused_dense import FusedDense

    FLASH_ATTENTION2_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION2_AVAILABLE = False

try:
    from flash_attn import FlashAttention

    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


class GatedResidual(nn.Module):
    """增强型门控残差连接，使用更平滑的激活函数"""

    def __init__(self, d_model, use_swish=False):
        super().__init__()
        activation = nn.Swish() if use_swish else nn.Sigmoid()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            activation
        )

    def forward(self, x, residual):
        combined = torch.cat([x, residual], dim=-1)
        gate_value = self.gate(combined)
        return x * gate_value + residual * (1 - gate_value)


class InvertedResidualConv(nn.Module):
    """MobileNetV3风格的倒置残差卷积块"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expand_ratio=4,
                 use_dwconv=True, dropout=0.1):
        super().__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_dwconv = use_dwconv
        self.stride = stride

        layers = []
        if expand_ratio != 1:
            # 扩展层
            layers.append(nn.Conv1d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.SiLU())  # 使用SiLU激活函数

        if use_dwconv:
            # 深度卷积
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride,
                                    padding=(kernel_size - 1) // 2, groups=hidden_dim, bias=False))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.SiLU())

        # 点卷积(投影层)
        layers.append(nn.Conv1d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm1d(out_channels))

        self.conv = nn.Sequential(*layers)
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.use_residual:
            return x + self.dropout(self.conv(x))
        else:
            return self.dropout(self.conv(x))


class GLUActivation(nn.Module):
    """改进的门控线性单元，支持更多激活函数"""

    def __init__(self, dim=-1, gate_act="sigmoid"):
        super().__init__()
        self.dim = dim
        if gate_act == "sigmoid":
            self.gate = nn.Sigmoid()
        elif gate_act == "swish":
            self.gate = nn.SiLU()
        elif gate_act == "gelu":
            self.gate = nn.GELU()
        else:
            raise ValueError(f"不支持的门控激活函数: {gate_act}")

    def forward(self, x):
        x, gate = x.chunk(2, dim=self.dim)
        return x * self.gate(gate)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization，更高效的归一化方法"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _rms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._rms(x)
        return output * self.weight


class SeasonalLayerNorm(nn.Module):
    """增强型季节性归一化层，结合RMSNorm"""

    def __init__(self, channels, use_rmsnorm=True):
        super().__init__()
        if use_rmsnorm:
            self.norm = RMSNorm(channels)
        else:
            self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        x_hat = self.norm(x)
        bias = torch.mean(x_hat, dim=1, keepdim=True)
        return x_hat - bias


class moving_avg(nn.Module):
    """改进的移动平均模块，支持空洞卷积"""

    def __init__(self, kernel_size, stride, use_dilation=False, dilation_rate=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_dilation = use_dilation
        self.dilation_rate = dilation_rate

        if use_dilation:
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride,
                                    padding=(kernel_size - 1) // 2 * dilation_rate, dilation=dilation_rate)
        else:
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride,
                                    padding=(kernel_size - 1) // 2)

    def forward(self, x):
        # 改进的填充策略，保持序列长度不变
        if self.use_dilation:
            padding = ((self.kernel_size - 1) // 2 * self.dilation_rate)
            x = F.pad(x, (padding, padding), mode='reflect')
        else:
            padding = (self.kernel_size - 1) // 2
            x = F.pad(x, (padding, padding), mode='reflect')

        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """增强型序列分解模块，支持多尺度分解"""

    def __init__(self, kernel_size, num_scales=1, use_dilation=False):
        super().__init__()
        scales = [kernel_size // (2 ** i) for i in range(num_scales)] if num_scales > 1 else [kernel_size]
        self.decomps = nn.ModuleList([
            moving_avg(scale, stride=1, use_dilation=use_dilation) for scale in scales
        ])
        self.num_scales = num_scales

    def forward(self, x):
        moving_means = []
        for decomp in self.decomps:
            moving_mean = decomp(x)
            moving_means.append(moving_mean)

        # 多尺度移动平均融合
        moving_mean = sum(moving_means) / len(moving_means)
        res = x - moving_mean
        return res, moving_mean


### 改进的掩码生成工具类
class MaskGenerator:
    @staticmethod
    def local_mask(seq_len, window, is_causal=False, device=None, dtype=torch.bool):
        """生成局部注意力掩码，支持更高效的向量化操作"""
        row = torch.arange(seq_len, device=device).view(-1, 1)
        col = torch.arange(seq_len, device=device).view(1, -1)
        mask = (col >= row - window // 2) & (col < row + window // 2 + 1)

        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype, device=device), diagonal=1)
            mask = mask & (~causal_mask)

        return mask

    @staticmethod
    def create_sliding_window_mask(seq_len, window_size, is_causal=False, device=None):
        """创建滑动窗口掩码，优化内存使用"""
        if window_size >= seq_len:
            return None  # 全连接，无需掩码

        # 使用更高效的方式生成掩码
        indices = torch.arange(seq_len, device=device)
        mask = (indices[None, :] >= indices[:, None] - window_size // 2) & \
               (indices[None, :] < indices[:, None] + window_size // 2 + 1)

        if is_causal:
            mask = mask & torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=0)

        return ~mask  # 转换为PyTorch注意力所需的掩码格式


### 高性能MQA局部注意力模块
class MQALocalAttention(nn.Module):
    """结合MultiQueryAttention的局部注意力，减少KV缓存占用"""

    def __init__(self, d_model, local_window=12, num_heads=4, num_kv_heads=1, dropout=0.1,
                 use_flash2=False, use_xformers=False):
        super().__init__()
        self.local_window = local_window
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.kv_head_dim = d_model // num_kv_heads

        self.use_flash2 = use_flash2 and FLASH_ATTENTION2_AVAILABLE
        self.use_xformers = use_xformers and XFORMERS_AVAILABLE
        self.use_flash = FLASH_ATTENTION_AVAILABLE and not use_flash2

        # 线性投影层，支持MultiQueryAttention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model // num_heads * num_kv_heads)
        self.v_proj = nn.Linear(d_model, d_model // num_heads * num_kv_heads)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm = SeasonalLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gated_residual = GatedResidual(d_model, use_swish=True)
        self.attn_entropy = None

        # 梯度检查点
        self.checkpoint = False

    @custom_fwd(cast_inputs=torch.float16)
    def _flash2_forward(self, q, k, v, mask, is_causal):
        # FlashAttention-2实现，支持KV缓存和MQA
        batch_size, seq_len, _ = q.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim)

        # 转换为FlashAttention-2所需的格式
        q = q.transpose(1, 2)  # [batch, num_heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 生成因果掩码
        cu_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=q.device)
        max_seq_len = seq_len
        attn_output, _ = flash_attn_with_kvcache(
            q, k, v, cu_seqlens, max_seq_len,
            dropout_p=0.0 if not self.training else dropout,
            因果=is_causal,
            softmax_scale=None,
            return_attn_probs=False
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return attn_output

    def _xformers_forward(self, q, k, v, mask, is_causal):
        # xFormers实现，支持高效的局部注意力
        batch_size, seq_len, _ = q.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)

        # 创建xFormers所需的掩码
        if mask is not None:
            # 转换为xFormers的掩码格式
            mask = mask.view(1, 1, seq_len, seq_len)
        else:
            mask = None

        # 使用xFormers的高效注意力
        attn_output = xops.memory_efficient_attention(
            q, k, v, attn_bias=mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scale=1.0 / math.sqrt(self.head_dim)
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return attn_output

    def _standard_forward(self, q, k, v, mask, is_causal):
        # 标准实现，支持MultiQueryAttention
        batch_size, seq_len, _ = q.shape

        # 线性投影
        q = self.q_proj(q).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(k).view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim)
        v = self.v_proj(v).view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim)

        # 调整维度以进行注意力计算
        q = q.transpose(1, 2)  # [batch, num_heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)

        # 应用因果掩码
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

        # 应用softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 计算注意力输出
        attn_output = torch.matmul(attn_probs, v)

        # 调整回原始维度
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)

        # 计算注意力熵
        if self.training:
            self.attn_entropy = -torch.mean(attn_probs * torch.log(attn_probs + 1e-8))

        return attn_output

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        # 输入维度适配：[batch, seq_len, d_model]
        q = query
        k = key
        v = value

        # 生成局部窗口掩码
        seq_len = q.size(1)
        device = q.device
        mask = MaskGenerator.create_sliding_window_mask(seq_len, self.local_window, is_causal, device)

        # 合并原始掩码
        if attn_mask is not None:
            if mask is None:
                mask = attn_mask
            else:
                mask = mask | attn_mask

        # 根据可用库选择实现方式
        if self.use_flash2:
            attn_output = self._flash2_forward(q, k, v, mask, is_causal)
        elif self.use_xformers:
            attn_output = self._xformers_forward(q, k, v, mask, is_causal)
        else:
            attn_output = self._standard_forward(q, k, v, mask, is_causal)

        # 门控残差连接和归一化
        output = self.gated_residual(query, self.dropout(attn_output))
        return self.norm(output), None


### 多尺度MQA-LNN模块
class MQALNNMultiScale(nn.Module):
    """结合MultiQueryAttention的多尺度局部注意力模块"""

    def __init__(self, d_model, windows=[8, 16, 32], num_heads=4, num_kv_heads=1,
                 dropout=0.1, use_flash2=False, use_xformers=False):
        super().__init__()
        self.lnns = nn.ModuleList([
            MQALocalAttention(d_model, window, num_heads, num_kv_heads, dropout, use_flash2, use_xformers)
            for window in windows
        ])
        self.proj = InvertedResidualConv(d_model * len(windows), d_model, kernel_size=1, expand_ratio=0.5)
        self.norm = SeasonalLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gated_residual = GatedResidual(d_model, use_swish=True)

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        outputs = []

        # 并行计算不同窗口大小的局部注意力
        for lnn in self.lnns:
            out, _ = lnn(query, key, value, attn_mask, is_causal)
            outputs.append(out)

        # 拼接多尺度特征
        concat = torch.cat(outputs, dim=-1)

        # 转换维度以适应卷积: [batch, seq_len, channels] → [batch, channels, seq_len]
        concat = concat.transpose(1, 2)

        # 应用改进的倒置残差卷积进行特征融合
        output = self.proj(concat)

        # 转回原始维度
        output = output.transpose(1, 2)

        # 门控残差连接和归一化
        output = self.gated_residual(query, self.dropout(output))
        return self.norm(output), None


### 动态窗口MQA-LNN模块
class DynamicMQALNNLocalAttention(nn.Module):
    """动态窗口大小的MultiQuery局部注意力"""

    def __init__(self, d_model, num_heads=4, num_kv_heads=1, max_window=32,
                 dropout=0.1, use_flash2=False, use_xformers=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_model = d_model
        self.max_window = max_window
        self.min_window = 3
        self.head_dim = d_model // num_heads
        self.kv_head_dim = d_model // num_kv_heads

        self.use_flash2 = use_flash2 and FLASH_ATTENTION2_AVAILABLE
        self.use_xformers = use_xformers and XFORMERS_AVAILABLE

        # 动态窗口预测器，使用更高效的卷积结构
        self.window_predictor = nn.Sequential(
            InvertedResidualConv(d_model, d_model // 4, kernel_size=3, stride=1),
            nn.GELU(),
            InvertedResidualConv(d_model // 4, 1, kernel_size=1, stride=1)
        )

        # MultiQueryAttention投影层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model // num_heads * num_kv_heads)
        self.v_proj = nn.Linear(d_model, d_model // num_heads * num_kv_heads)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm = SeasonalLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gated_residual = GatedResidual(d_model, use_swish=True)

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        # 输入维度适配
        q = query
        k = key
        v = value
        seq_len = q.size(1)
        device = q.device

        # 基于序列特征预测窗口大小
        query_conv = query.transpose(1, 2)  # [batch, channels, seq_len]
        window_ratio = self.window_predictor(query_conv)  # [batch, 1, seq_len]

        # 取平均作为全局窗口大小
        window_ratio = torch.mean(window_ratio, dim=(1, 2), keepdim=True)

        # 将比率转换为实际窗口大小
        window = int(self.min_window + window_ratio.item() * (self.max_window - self.min_window))
        window = min(window, seq_len)  # 确保窗口不超过序列长度

        # 生成对应窗口的掩码
        mask = MaskGenerator.create_sliding_window_mask(seq_len, window, is_causal, device)

        # 合并原始掩码（若有）
        if attn_mask is not None:
            if mask is None:
                mask = attn_mask
            else:
                mask = mask | attn_mask

        # 线性投影
        q = self.q_proj(q).view(q.size(0), q.size(1), self.num_heads, self.head_dim)
        k = self.k_proj(k).view(k.size(0), k.size(1), self.num_kv_heads, self.kv_head_dim)
        v = self.v_proj(v).view(v.size(0), v.size(1), self.num_kv_heads, self.kv_head_dim)

        # 调整维度以进行注意力计算
        q = q.transpose(1, 2)  # [batch, num_heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 根据可用库选择实现方式
        if self.use_flash2:
            # FlashAttention-2实现
            cu_seqlens = torch.full((q.size(0),), seq_len, dtype=torch.int32, device=device)
            attn_output, _ = flash_attn_with_kvcache(
                q, k, v, cu_seqlens, seq_len,
                dropout_p=0.0 if not self.training else self.dropout.p,
                因果=is_causal,
                softmax_scale=None,
                attn_mask=mask
            )
            attn_output = attn_output.transpose(1, 2).reshape(q.size(0), seq_len, -1)
        elif self.use_xformers:
            # xFormers实现
            if mask is not None:
                mask = mask.view(1, 1, seq_len, seq_len)
            attn_output = xops.memory_efficient_attention(
                q, k, v, attn_bias=mask,
                dropout=0.0 if not self.training else self.dropout.p,
                scale=1.0 / math.sqrt(self.head_dim)
            )
            attn_output = attn_output.transpose(1, 2).reshape(q.size(0), seq_len, -1)
        else:
            # 标准实现
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)
            if is_causal:
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
                attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, v)
            attn_output = attn_output.transpose(1, 2).reshape(q.size(0), seq_len, -1)
            attn_output = self.out_proj(attn_output)

        # 门控残差连接和归一化
        output = self.gated_residual(query, self.dropout(attn_output))
        return self.norm(output), None


### 增强型FFN模块
class EnhancedFFN(nn.Module):
    """改进的前馈网络，结合InvertedResidualConv和GLU激活"""

    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.activation = activation

        # 使用InvertedResidualConv作为基础构建块
        self.net = nn.Sequential(
            InvertedResidualConv(d_model, d_ff * 2, kernel_size=1, expand_ratio=0.5),
            GLUActivation(dim=1, gate_act=activation),
            nn.Dropout(dropout),
            InvertedResidualConv(d_ff, d_model, kernel_size=1, expand_ratio=0.5)
        )

        self.norm = SeasonalLayerNorm(d_model)
        self.gated_residual = GatedResidual(d_model, use_swish=True)

    def forward(self, x):
        # 前馈网络
        x_input = x
        x = self.net(x)

        # 门控残差连接和归一化
        output = self.gated_residual(x, x_input)
        return self.norm(output)


### 增强型编码器层
class EnhancedEncoderLayer(nn.Module):
    """集成MQA-LNN的增强型编码器层，结合最新序列分解技术"""

    def __init__(self, d_model, d_ff=None, local_window=12, num_heads=4, num_kv_heads=1,
                 moving_avg=25, dropout=0.1, activation="gelu",
                 use_multi_scale=False, use_dynamic_window=False, use_flash2=False, use_xformers=False):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        # 选择LNN类型
        if use_multi_scale:
            self.attention = MQALNNMultiScale(
                d_model, windows=[local_window // 2, local_window, local_window * 2],
                num_heads=num_heads, num_kv_heads=num_kv_heads,
                dropout=dropout, use_flash2=use_flash2, use_xformers=use_xformers
            )
        elif use_dynamic_window:
            self.attention = DynamicMQALNNLocalAttention(
                d_model, num_heads=num_heads, num_kv_heads=num_kv_heads,
                max_window=min(64, local_window * 3), dropout=dropout,
                use_flash2=use_flash2, use_xformers=use_xformers
            )
        else:
            self.attention = MQALocalAttention(
                d_model, local_window, num_heads, num_kv_heads, dropout,
                use_flash2=use_flash2, use_xformers=use_xformers
            )

        # 增强的FFN模块
        self.ffn = EnhancedFFN(d_model, d_ff, dropout, activation)

        # 增强的序列分解模块，支持多尺度分解
        self.decomp1 = series_decomp(moving_avg, num_scales=2, use_dilation=True)
        self.decomp2 = series_decomp(moving_avg, num_scales=2, use_dilation=True)

    def forward(self, x, attn_mask=None):
        # 注意力子层
        new_x, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x, _ = self.decomp1(x + new_x)

        # FFN子层
        y = self.ffn(x)
        res, _ = self.decomp2(x + y)

        return res, None


### 增强型解码器层
class EnhancedDecoderLayer(nn.Module):
    """集成MQA-LNN的增强型解码器层，支持高效的因果注意力"""

    def __init__(self, d_model, c_out, self_window=12, cross_window=12, num_heads=4, num_kv_heads=1,
                 moving_avg=25, dropout=0.1, activation="gelu",
                 use_multi_scale=False, use_dynamic_window=False, use_flash2=False, use_xformers=False):
        super().__init__()
        d_ff = 4 * d_model

        # 选择LNN类型
        if use_multi_scale:
            self.self_attention = MQALNNMultiScale(
                d_model, windows=[self_window // 2, self_window, self_window * 2],
                num_heads=num_heads, num_kv_heads=num_kv_heads,
                dropout=dropout, use_flash2=use_flash2, use_xformers=use_xformers
            )
            self.cross_attention = MQALNNMultiScale(
                d_model, windows=[cross_window // 2, cross_window, cross_window * 2],
                num_heads=num_heads, num_kv_heads=num_kv_heads,
                dropout=dropout, use_flash2=use_flash2, use_xformers=use_xformers
            )
        elif use_dynamic_window:
            self.self_attention = DynamicMQALNNLocalAttention(
                d_model, num_heads=num_heads, num_kv_heads=num_kv_heads,
                max_window=min(64, self_window * 3), dropout=dropout,
                use_flash2=use_flash2, use_xformers=use_xformers
            )
            self.cross_attention = DynamicMQALNNLocalAttention(
                d_model, num_heads=num_heads, num_kv_heads=num_kv_heads,
                max_window=min(64, cross_window * 3), dropout=dropout,
                use_flash2=use_flash2, use_xformers=use_xformers
            )
        else:
            self.self_attention = MQALocalAttention(
                d_model, self_window, num_heads, num_kv_heads, dropout,
                use_flash2=use_flash2, use_xformers=use_xformers
            )
            self.cross_attention = MQALocalAttention(
                d_model, cross_window, num_heads, num_kv_heads, dropout,
                use_flash2=use_flash2, use_xformers=use_xformers
            )

        # 增强的FFN模块
        self.ffn = EnhancedFFN(d_model, d_ff, dropout, activation)

        # 多尺度序列分解
        self.decomp1 = series_decomp(moving_avg, num_scales=2, use_dilation=True)
        self.decomp2 = series_decomp(moving_avg, num_scales=2, use_dilation=True)
        self.decomp3 = series_decomp(moving_avg, num_scales=2, use_dilation=True)

        # 改进的投影层
        self.projection = InvertedResidualConv(d_model, c_out, kernel_size=3, stride=1, expand_ratio=0.5)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # 自注意力：生成因果掩码
        if x_mask is None:
            x_mask = torch.triu(torch.ones(x.size(1), x.size(1),
                                           device=x.device, dtype=torch.bool), diagonal=1)
        x, _ = self.self_attention(x, x, x, attn_mask=x_mask, is_causal=True)
        x, trend1 = self.decomp1(x)

        # 交叉注意力
        x, _ = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x, trend2 = self.decomp2(x)

        # FFN子层
        y = self.ffn(x)
        x, trend3 = self.decomp3(x + y)

        # 趋势合并和投影
        residual_trend = trend1 + trend2 + trend3
        residual_trend = residual_trend.transpose(1, 2)
        residual_trend = self.projection(residual_trend)
        residual_trend = residual_trend.transpose(1, 2)

        return x, residual_trend


class EnhancedEncoder(nn.Module):
    """增强型Autoformer编码器，支持梯度检查点"""

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, use_checkpoint=False):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.use_checkpoint = use_checkpoint

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                if self.use_checkpoint and self.training:
                    x, attn = torch.utils.checkpoint.checkpoint(attn_layer, x, attn_mask)
                else:
                    x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            if self.use_checkpoint and self.training:
                x, attn = torch.utils.checkpoint.checkpoint(self.attn_layers[-1], x)
            else:
                x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                if self.use_checkpoint and self.training:
                    x, attn = torch.utils.checkpoint.checkpoint(attn_layer, x, attn_mask)
                else:
                    x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class EnhancedDecoder(nn.Module):
    """增强型Autoformer解码器，支持混合精度训练"""

    def __init__(self, layers, norm_layer=None, projection=None, use_amp=False):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        self.use_amp = use_amp

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        if trend is None:
            trend = torch.zeros_like(x, device=x.device)

        for layer in self.layers:
            if self.use_amp and self.training:
                with torch.cuda.amp.autocast(enabled=True):
                    x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            else:
                x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x, trend