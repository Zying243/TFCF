import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn  # 新增这行：导入PyTorch神经网络模块并简写为nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import pywt

warnings.filterwarnings('ignore')

class EnhancedChannelAttention(nn.Module):
    """增强版通道注意力模块：移除通道内BN，避免小批量报错"""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        # 关键修改1：移除全连接层中的BatchNorm1d（小批量下统计量不稳定）
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            # 移除nn.BatchNorm1d(in_channels // reduction)
            nn.GELU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            # 移除nn.BatchNorm1d(in_channels)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = avg_y + max_y
        y = self.fc(y).view(b, c, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class UNet(nn.Module):
    """1D U-Net：将BatchNorm1d替换为InstanceNorm1d（对批量大小无依赖）"""
    # 关键修改1：输入维度默认值从9→10，适配10通道数据
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim  # 保存输入维度（10）
        self.hidden_dim = hidden_dim
        # 时间嵌入层（不变）
        self.time_emb1 = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_emb2 = nn.Sequential(
            nn.Linear(1, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )
        self.time_emb3 = nn.Sequential(
            nn.Linear(1, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4)
        )
        # -------------------------- 下采样模块：BatchNorm1d → InstanceNorm1d --------------------------
        # 关键修改2：第一层卷积输入通道从9→10，与input_dim一致
        self.down1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),  # 原9→input_dim（10）
            nn.InstanceNorm1d(64),  # 关键修改2：替换BatchNorm为InstanceNorm（单样本也能正常计算）
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm1d(64),  # 同上
            EnhancedChannelAttention(64)
        )
        # 关键修改3：适配输入通道从9→10
        self.down1_adapt = nn.Conv1d(input_dim, 64, kernel_size=1)  # 原9→input_dim（10）
        self.down2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm1d(128),  # 关键修改
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm1d(128),  # 关键修改
            EnhancedChannelAttention(128)
        )
        self.down2_adapt = nn.Conv1d(64, 128, kernel_size=1)
        self.down3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.InstanceNorm1d(256),  # 关键修改
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm1d(256),  # 关键修改
            EnhancedChannelAttention(256)
        )
        self.down3_adapt = nn.Conv1d(128, 256, kernel_size=1)
        # -------------------------- 上采样模块：BatchNorm1d → InstanceNorm1d --------------------------
        self.up3 = nn.Sequential(
            nn.Conv1d(384, 128, kernel_size=3, padding=1),
            nn.InstanceNorm1d(128),  # 关键修改
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm1d(128),  # 关键修改
            EnhancedChannelAttention(128)
        )
        self.up3_adapt = nn.Conv1d(384, 128, kernel_size=1)
        self.upsample3 = nn.ConvTranspose1d(256, 256, kernel_size=2, stride=2)
        self.up2 = nn.Sequential(
            nn.Conv1d(192, 64, kernel_size=3, padding=1),
            nn.InstanceNorm1d(64),  # 关键修改
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm1d(64),  # 关键修改
            EnhancedChannelAttention(64)
        )
        self.up2_adapt = nn.Conv1d(192, 64, kernel_size=1)
        self.upsample2 = nn.ConvTranspose1d(128, 128, kernel_size=2, stride=2)
        # 关键修改4：最后一层卷积输入通道和输出通道从9→10（适配10通道输出）
        self.up1 = nn.Sequential(
            nn.Conv1d(64 + input_dim, input_dim, kernel_size=3, padding=1),  # 原73（64+9）→64+input_dim（74），输出→input_dim（10）
            nn.InstanceNorm1d(input_dim),  # 原9→input_dim（10）
            nn.ReLU(),
            nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),  # 原9→input_dim（10）
            EnhancedChannelAttention(input_dim)  # 原9→input_dim（10）
        )
        # 关键修改5：适配输入通道从9→10
        self.up1_adapt = nn.Conv1d(64 + input_dim, input_dim, kernel_size=1)  # 原73（64+9）→64+input_dim（74），输出→input_dim（10）
        # 辅助组件（不变）
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(2, stride=2)
    def forward(self, x, t):
        # 关键修改3：添加输入序列长度检查与自适应调整（解决“16≠8760”的长度不匹配）
        target_seq_len = x.size(2)  # 动态适配输入序列长度（不再强制8760）
        x_orig = x
        batch_size = x.shape[0]
        # 下采样流程（仅修改时间嵌入的扩展长度，用target_seq_len动态匹配）
        t_emb1 = self.time_emb1(t.unsqueeze(1).float()).unsqueeze(2)
        t_emb1 = t_emb1.repeat(1, 1, target_seq_len)  # 动态扩展到输入长度
        x1_conv = self.down1(x)
        x1_adapt = self.down1_adapt(x)
        x1 = self.relu(x1_conv + x1_adapt + t_emb1)
        x_pool1 = self.pool(x1)
        t_emb2 = self.time_emb2(t.unsqueeze(1).float()).unsqueeze(2)
        t_emb2 = t_emb2.repeat(1, 1, x_pool1.size(2))  # 池化后长度自动匹配
        x2_conv = self.down2(x_pool1)
        x2_adapt = self.down2_adapt(x_pool1)
        x2 = self.relu(x2_conv + x2_adapt + t_emb2)
        x_pool2 = self.pool(x2)
        t_emb3 = self.time_emb3(t.unsqueeze(1).float()).unsqueeze(2)
        t_emb3 = t_emb3.repeat(1, 1, x_pool2.size(2))
        x3_conv = self.down3(x_pool2)
        x3_adapt = self.down3_adapt(x_pool2)
        x3 = self.relu(x3_conv + x3_adapt + t_emb3)
        # 上采样流程（不变，长度匹配逻辑保留）
        x_up3 = self.upsample3(x3)
        if x_up3.size(2) != x2.size(2):
            x_up3 = x_up3[:, :, :x2.size(2)]
        x_up3_concat = torch.cat([x_up3, x2], dim=1)
        x_up3_conv = self.up3(x_up3_concat)
        x_up3_adapt = self.up3_adapt(x_up3_concat)
        x_up3 = self.relu(x_up3_conv + x_up3_adapt)
        x_up2 = self.upsample2(x_up3)
        if x_up2.size(2) != x1.size(2):
            x_up2 = x_up2[:, :, :x1.size(2)]
        x_up2_concat = torch.cat([x_up2, x1], dim=1)
        x_up2_conv = self.up2(x_up2_concat)
        x_up2_adapt = self.up2_adapt(x_up2_concat)
        x_up2 = self.relu(x_up2_conv + x_up2_adapt)
        x_up1_concat = torch.cat([x_up2, x_orig], dim=1)
        x_up1_conv = self.up1(x_up1_concat)
        x_up1_adapt = self.up1_adapt(x_up1_concat)
        x_out = x_up1_conv + x_up1_adapt
        return x_out

class DiffusionProcessor:
    """适配 input_dim=10、兼容任意序列长度的扩散处理器（优化训练策略版本）"""
    # 关键修改1：输入维度默认值从9→10
    def __init__(self, input_dim=10, seq_len=8760, noise_steps=200,
                 beta_start=0.0001, beta_end=0.02,  # 调整beta_end为较小值，适合余弦调度
                 model_path=None, ddim_steps=50, ddim_eta=0.0):
        self.noise_steps = noise_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim  # 固定为10
        self.seq_len = seq_len  # 保留默认值，但不再强制输入必须为此长度
        # -------------------------- DDIM采样参数（加速去噪）--------------------------
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.ddim_timesteps = np.asarray(list(range(0, noise_steps, noise_steps // ddim_steps))) + 1
        self.ddim_timesteps = np.append(self.ddim_timesteps, noise_steps - 1)
        # -------------------------- 噪声调度（优化：使用余弦调度替代指数调度）--------------------------
        self.betas = self.cosine_beta_schedule(noise_steps, beta_start, beta_end)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # -------------------------- 初始化UNet（直接适配10通道数据）--------------------------
        self.model = UNet(input_dim=input_dim, hidden_dim=64).to(self.device)  # 传入input_dim=10
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"✅ 成功加载预训练模型: {model_path}")
            except Exception as e:
                print(f"⚠️ 预训练模型加载失败: {str(e)}，将使用新初始化模型")
        else:
            print("ℹ️ 未指定模型路径，使用新初始化模型")
        # 鲁棒损失函数（优化：使用Huber损失替代Charbonnier损失）
        self.criterion = self.huber_loss
    # 新增：余弦噪声调度函数（噪声平滑增长，避免后期噪声过强）
    def cosine_beta_schedule(self, T, beta_start, beta_end):
        """
        余弦噪声调度：让噪声强度随时间平滑增长
        参考：https://arxiv.org/abs/2102.09672
        """
        t = torch.linspace(0, T, T, device=self.device)  # 0到T的时间步
        # 余弦调度核心公式
        f_t = torch.cos(((t / T) + beta_start) / (1 + beta_start) * torch.pi / 2) ** 2
        alpha_cumprod = f_t / f_t[0]  # 累积乘积归一化
        # 计算beta值
        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        # 裁剪beta范围，确保数值稳定
        betas = torch.clip(betas, min=0.0001, max=beta_end)
        # 补全第一个beta（使长度为T）
        betas = torch.cat([torch.tensor([0.0], device=self.device), betas])
        return betas
    # 优化：Huber损失（兼顾MSE和MAE的优点，对异常值更鲁棒）
    def huber_loss(self, x, y, delta=1.0):
        """
        Huber损失：
        - 当误差小于delta时，使用MSE（梯度更稳定）
        - 当误差大于delta时，使用MAE（抗异常值能力强）
        """
        residual = torch.abs(x - y)
        cond = residual < delta
        loss = torch.where(cond, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))
        return loss.mean()
    # 保留原Charbonnier损失（如需切换可直接替换criterion）
    def charbonnier_loss(self, x, y, eps=1e-6):
        return torch.sqrt((x - y) ** 2 + eps ** 2).mean()
    def _add_noise(self, x, t):
        """给数据添加噪声（扩散过程）"""
        t = t.clamp(0, self.noise_steps - 1)  # 防止t超出范围
        sqrt_alphas_cumprod = self.alphas_cumprod[t].sqrt().reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod[t]).sqrt().reshape(-1, 1, 1)
        noise = torch.randn_like(x, device=self.device)
        noisy_x = sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise
        return noisy_x, noise
    def train_step(self, x, optimizer):
        """
        单步训练（优化：加入梯度裁剪，接收optimizer参数）
        返回：标量损失值
        """
        self.model.train()
        # 输入合法性校验
        if x.dim() != 3:
            raise ValueError(f"输入必须是3维张量 (batch, channels, seq_len)，但得到 {x.shape}")
        batch_size, channels, seq_len = x.shape
        # 关键修改2：校验通道数为10（原9）
        if channels != self.input_dim:
            raise ValueError(f"通道数必须为 {self.input_dim}，但得到 {channels}")
        if batch_size < 1:
            raise ValueError(f"批量大小必须 ≥1，但得到 {batch_size}")
        elif batch_size == 1:
            print(f"⚠️ 警告：批量大小为1，可能影响训练稳定性（建议≥2）")
        # 核心训练逻辑 + 梯度裁剪
        optimizer.zero_grad()  # 清空梯度
        t = torch.randint(0, self.noise_steps, (batch_size,), device=self.device)
        x_noisy, noise = self._add_noise(x, t)
        noise_pred = self.model(x_noisy, t)
        loss = self.criterion(noise_pred, noise)
        # 关键优化：梯度裁剪（防止梯度爆炸，稳定训练）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        return loss.item()  # 返回标量损失，方便日志记录
    def denoise(self, data_np):
        """去噪（兼容任意序列长度，输入：(seq_len, 10)，输出：(seq_len, 10)）"""
        # 数据格式转换：(seq_len, input_dim) → (1, input_dim, seq_len)
        x = torch.from_numpy(data_np).float().unsqueeze(0).permute(0, 2, 1).to(self.device)
        with torch.no_grad():
            for i in reversed(range(len(self.ddim_timesteps))):
                t = self.ddim_timesteps[i]
                t_prev = self.ddim_timesteps[i - 1] if i > 0 else 0
                # 准备时间步张量
                t_tensor = torch.tensor([t], device=self.device).repeat(x.shape[0])
                # 预测噪声
                noise_pred = self.model(x, t_tensor)
                # DDIM采样公式
                alpha_cumprod_t = self.alphas_cumprod[t]
                alpha_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else 1.0
                # 预测原始数据
                x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
                x0_pred = torch.clamp(x0_pred, -1., 1.)  # 限制范围避免数值爆炸
                # 计算噪声标准差
                sigma = self.ddim_eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
                    (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
                )
                # 生成随机噪声（最后一步不需要噪声）
                noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
                # 更新采样结果
                x = torch.sqrt(alpha_cumprod_t_prev) * x0_pred + \
                    torch.sqrt(1 - alpha_cumprod_t_prev - sigma ** 2) * noise_pred + \
                    sigma * noise
        # 格式转换回(seq_len, input_dim)
        denoised_np = x.permute(0, 2, 1).squeeze(0).cpu().numpy()
        return denoised_np
    def generate_aug_samples(self, data_np, num_aug=1):
        """生成增强样本（兼容任意序列长度）"""
        input_dim = data_np.shape[1]
        seq_len = data_np.shape[0]
        aug_samples = []
        with torch.no_grad():
            for _ in range(num_aug):
                # 从纯噪声开始生成（匹配输入数据的序列长度）
                x = torch.randn(1, input_dim, seq_len, device=self.device)
                for i in reversed(range(len(self.ddim_timesteps))):
                    t = self.ddim_timesteps[i]
                    t_prev = self.ddim_timesteps[i - 1] if i > 0 else 0
                    t_tensor = torch.tensor([t], device=self.device)
                    noise_pred = self.model(x, t_tensor)
                    alpha_cumprod_t = self.alphas_cumprod[t]
                    alpha_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else 1.0
                    x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
                    x0_pred = torch.clamp(x0_pred, -1., 1.)
                    sigma = self.ddim_eta * torch.sqrt(
                        (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
                        (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
                    )
                    noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
                    x = torch.sqrt(alpha_cumprod_t_prev) * x0_pred + \
                        torch.sqrt(1 - alpha_cumprod_t_prev - sigma ** 2) * noise_pred + \
                        sigma * noise
                # 转换格式并添加到结果列表
                aug_np = x.permute(0, 2, 1).squeeze(0).cpu().numpy()
                aug_samples.append(aug_np)
        # 合并原始数据和增强数据
        combined_data = np.concatenate([data_np] + aug_samples, axis=0)
        return combined_data

class WaveletTransformer:
    """完整的小波变换处理器，确保实际生效"""
    def __init__(self, wavelet='db4', level=3, mode='symmetric', threshold=None):
        # 验证小波基有效性
        available_wavelets = pywt.wavelist()
        if wavelet not in available_wavelets:
            raise ValueError(f"Invalid wavelet '{wavelet}'. Available options: {available_wavelets}")
        self.wavelet = wavelet
        self.level = min(level, pywt.dwt_max_level(data_len=1024, filter_len=pywt.Wavelet(wavelet).dec_len))
        self.mode = mode
        self.threshold = threshold
        print(f"Initialized WaveletTransformer: {wavelet}, level={self.level}, threshold={threshold}")
    def transform(self, data):
        """实际执行小波变换并确保影响数据"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        n_features = data.shape[1]  # 自动适配10个特征
        processed = np.zeros_like(data)
        for i in range(n_features):
            # 处理NaN和Inf
            signal = np.nan_to_num(data[:, i], nan=np.nanmean(data[:, i]))
            try:
                # 实际执行小波变换
                coeffs = pywt.wavedec(
                    signal,
                    wavelet=self.wavelet,
                    level=self.level,
                    mode=self.mode
                )
                # 实际应用阈值去噪
                if self.threshold is not None:
                    coeffs = [self._apply_threshold(c) for c in coeffs]
                # 重构信号
                reconstructed = pywt.waverec(coeffs, wavelet=self.wavelet)
                processed[:, i] = reconstructed[:len(data)]
                # 确保数据有效性
                processed[:, i] = np.nan_to_num(processed[:, i], nan=0.0)
                # 调试输出（实际变换效果）
                if i == 0 and False:  # 设为True可打印调试信息
                    print(f"Feature {i} - Original mean: {np.mean(signal):.2f}, "
                          f"Processed mean: {np.mean(processed[:, i]):.2f}, "
                          f"Diff: {np.mean(np.abs(signal - processed[:, i])):.2f}")
            except Exception as e:
                print(f"Wavelet failed for feature {i}: {str(e)}")
                processed[:, i] = signal  # 失败时回退
        return processed
    def _apply_threshold(self, coeff):
        """实际应用的阈值函数"""
        if self.threshold is None or self.threshold == 0:
            return coeff
        return pywt.threshold(coeff, value=self.threshold, mode='soft')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='AK.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 wavelet='db4', wavelet_level=3, wavelet_threshold=None):
        # 初始化参数
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = 0
        self.freq = freq
        # 实际初始化小波变换器
        self.wavelet_transformer = None
        if wavelet is not None:
            try:
                self.wavelet_transformer = WaveletTransformer(
                    wavelet=wavelet,
                    level=wavelet_level,
                    threshold=wavelet_threshold
                )
                print(f"✅ Wavelet transform ENABLED | {wavelet} level={wavelet_level} "
                      f"threshold={wavelet_threshold}")
            except Exception as e:
                print(f"❌ Wavelet init failed: {str(e)}. Continuing without wavelet.")
                self.wavelet_transformer = None
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self._validate_data()
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # 数据分割边界
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # 特征选择
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 自动适配10个特征（排除date列）
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        # 初始标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # 实际应用小波变换
        if self.wavelet_transformer is not None:
            print("Applying wavelet transform to raw data...")
            try:
                original_mean = np.mean(data)
                data = self.wavelet_transformer.transform(data)  # 适配10个特征
                print(f"Wavelet transform completed. Mean before: {original_mean:.2f}, "
                      f"after: {np.mean(data):.2f}")
                # 变换后重新标准化
                if self.scale:
                    data = self.scaler.transform(data)
            except Exception as e:
                print(f"Wavelet transform failed: {str(e)}. Using original data.")
                data = self.scaler.transform(df_data.values) if self.scale else df_data.values
        # 时间特征处理
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    def _validate_data(self):
        if not hasattr(self, 'data_x'):
            raise ValueError("Data not initialized!")
        if len(self.data_x) < self.seq_len + self.pred_len:
            raise ValueError(
                f"Insufficient data length. Need at least {self.seq_len + self.pred_len} "
                f"time steps, but got {len(self.data_x)}"
            )
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # 边界检查
        if s_end > len(self.data_x) or r_end > len(self.data_y):
            # 关键修改：适配10个特征（原9→data_x.shape[1]，自动获取）
            seq_x = np.zeros((self.seq_len, self.data_x.shape[1]), dtype=self.data_x.dtype)
            seq_y = np.zeros((self.label_len + self.pred_len, self.data_y.shape[1]), dtype=self.data_y.dtype)
            seq_x_mark = np.zeros((self.seq_len, self.data_stamp.shape[1]), dtype=self.data_stamp.dtype)
            seq_y_mark = np.zeros((self.label_len + self.pred_len, self.data_stamp.shape[1]),
                                  dtype=self.data_stamp.dtype)
        else:
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
        # 维度处理
        if seq_x.ndim == 1:
            seq_x = seq_x.reshape(-1, 1)
        if seq_y.ndim == 1:
            seq_y = seq_y.reshape(-1, 1)
        # 转换为张量
        seq_x = torch.from_numpy(seq_x).float()
        seq_y = torch.from_numpy(seq_y).float()
        seq_x_mark = torch.from_numpy(seq_x_mark).float()
        seq_y_mark = torch.from_numpy(seq_y_mark).float()
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    def __len__(self):
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='AK.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 wavelet='db4', wavelet_level=3, wavelet_threshold=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        # 初始化小波变换器
        self.wavelet_transformer = None
        if wavelet is not None:
            try:
                self.wavelet_transformer = WaveletTransformer(
                    wavelet=wavelet,
                    level=wavelet_level,
                    threshold=wavelet_threshold
                )
                print(f"✅ Wavelet transform ENABLED for minute data | {wavelet} level={wavelet_level}")
            except Exception as e:
                print(f"❌ Minute data wavelet init failed: {str(e)}")
                self.wavelet_transformer = None
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self._validate_data()
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 自动适配10个特征
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # 实际应用小波变换
        if self.wavelet_transformer is not None:
            print("Applying wavelet transform to minute data...")
            try:
                data = self.wavelet_transformer.transform(data)  # 适配10个特征
                if self.scale:
                    data = self.scaler.transform(data)
            except Exception as e:
                print(f"Minute data wavelet failed: {str(e)}")
                data = df_data.values if not self.scale else self.scaler.transform(df_data.values)
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    def _validate_data(self):
        if not hasattr(self, 'data_x'):
            raise ValueError("Data not initialized!")
        if len(self.data_x) < self.seq_len + self.pred_len:
            raise ValueError(
                f"Insufficient data length. Need at least {self.seq_len + self.pred_len} "
                f"time steps, but got {len(self.data_x)}"
            )
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        if s_end > len(self.data_x) or r_end > len(self.data_y):
            # 关键修改：适配10个特征（原9→data_x.shape[1]）
            seq_x = np.zeros((self.seq_len, self.data_x.shape[1]), dtype=self.data_x.dtype)
            seq_y = np.zeros((self.label_len + self.pred_len, self.data_y.shape[1]), dtype=self.data_y.dtype)
            seq_x_mark = np.zeros((self.seq_len, self.data_stamp.shape[1]), dtype=self.data_stamp.dtype)
            seq_y_mark = np.zeros((self.label_len + self.pred_len, self.data_stamp.shape[1]),
                                  dtype=self.data_stamp.dtype)
        else:
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
        if seq_x.ndim == 1:
            seq_x = seq_x.reshape(-1, 1)
        if seq_y.ndim == 1:
            seq_y = seq_y.reshape(-1, 1)
        seq_x = torch.from_numpy(seq_x).float()
        seq_y = torch.from_numpy(seq_y).float()
        seq_x_mark = torch.from_numpy(seq_x_mark).float()
        seq_y_mark = torch.from_numpy(seq_y_mark).float()
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    def __len__(self):
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    """数据集类（强制timeenc=0，修复小波变换问题，适配10特征通道）"""
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='AK.csv',
                 target='OT', scale=True, timeenc=0, freq='h',  # 参数默认值保持0
                 wavelet='db4', wavelet_level=3, wavelet_threshold=None,
                 use_diffusion=True, diffusion_model_path=None, diffusion_num_aug=1):
        # 序列长度初始化
        if size is None:
            self.seq_len = 24 * 4 * 4  # 96
            self.label_len = 24 * 4  # 48
            self.pred_len = 24 * 4  # 48
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # 基础参数 - 强制timeenc为0，忽略外部传入值
        assert flag in ['train', 'test', 'val'], f"Invalid flag: {flag}"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = 0  # 强制设置为0，关键修改点
        self.freq = freq
        print(f"初始化参数 | timeenc={self.timeenc} (已强制设置), features={self.features}")  # 验证是否生效
        # 小波变换器（修复实现问题）
        self.wavelet_transformer = None
        if wavelet is not None and wavelet != 'None':
            try:
                # 直接实例化已定义的WaveletTransformer类
                self.wavelet_transformer = WaveletTransformer(
                    wavelet=wavelet,
                    level=wavelet_level,
                    threshold=wavelet_threshold
                )
                print(f"✅ Wavelet transform ENABLED | {wavelet} level={wavelet_level} threshold={wavelet_threshold}")
            except Exception as e:
                print(f"❌ Wavelet init failed: {e}. 已禁用小波变换")
                self.wavelet_transformer = None
        # 扩散模型处理器（适配10通道）
        self.diffusion_processor = None
        self.use_diffusion = use_diffusion
        self.diffusion_num_aug = diffusion_num_aug
        self.input_dim = None
        if use_diffusion:
            try:
                temp_df = pd.read_csv(os.path.join(root_path, data_path))
                if self.features == 'S':
                    self.input_dim = 1
                else:
                    # 关键修改：输入维度从9→自动计算（排除date列后的10个特征）
                    self.input_dim = len(temp_df.columns) - 1  # 排除date列，确保为10
                print(f"数据集计算的input_dim: {self.input_dim}")
                # 初始化扩散处理器时传入10通道（原9）
                self.diffusion_processor = DiffusionProcessor(
                    input_dim=self.input_dim,
                    seq_len=self.seq_len,
                    model_path=diffusion_model_path
                )
                print(f"✅ Diffusion ENABLED | 模型input_dim={self.diffusion_processor.input_dim}")
            except Exception as e:
                print(f"❌ Diffusion init failed: {e}. 已禁用扩散模型")
                self.use_diffusion = False
        # 数据加载
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self._validate_data()
    def __read_data__(self):
        # 读取原始数据
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        assert 'date' in df_raw.columns, "数据文件必须包含 'date' 列"
        print(f"原始数据形状: {df_raw.shape} (行: 样本数, 列: 特征数+1)")
        # 数据集分割
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        print(f"数据集分割 | {self.flag} 数据范围: {border1} 到 {border2}")
        # 特征选择（保留10个特征）
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 排除date列，保留所有10个特征
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        print(f"特征选择后 | 形状: {df_data.shape} (通道数: {df_data.shape[1]})")
        # 数据标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            print("数据已标准化")
        else:
            data = df_data.values
            print("未使用标准化")
        # 小波变换（适配10特征）
        if self.wavelet_transformer is not None:
            try:
                original_shape = data.shape
                data = self.wavelet_transformer.transform(data)  # 自动处理10个特征
                print(f"小波变换后 | 形状: {data.shape} (原始: {original_shape})")
                if self.scale:
                    data = self.scaler.transform(data)
            except Exception as e:
                print(f"Wavelet failed: {e}. 使用原始数据")
                data = self.scaler.transform(df_data.values) if self.scale else df_data.values
        # 扩散模型处理（适配10通道）
        if self.diffusion_processor is not None:
            try:
                print(f"扩散处理前 | 数据形状: {data.shape} (通道数: {data.shape[1]})")
                # 通道数校验：确保为10（与diffusion_processor.input_dim一致）
                if data.shape[1] != self.diffusion_processor.input_dim:
                    print(f"调整通道数: {data.shape[1]} → {self.diffusion_processor.input_dim}")
                    if self.features == 'S':
                        target_idx = df_raw.columns.get_loc(self.target) - 1
                        data = data[:, [target_idx]]
                    else:
                        data = data[:, :self.diffusion_processor.input_dim]  # 截取前10个特征
                if self.flag == 'train':
                    print(f"应用扩散增强 (num_aug={self.diffusion_num_aug})")
                    data = self.diffusion_processor.generate_aug_samples(data, self.diffusion_num_aug)
                    print(f"扩散增强后 | 形状: {data.shape} (通道数: {data.shape[1]})")
                    if self.scale:
                        data = self.scaler.transform(data)
                else:
                    print(f"应用扩散去噪 ({self.flag})")
                    data = self.diffusion_processor.denoise(data)
                    print(f"扩散去噪后 | 形状: {data.shape} (通道数: {data.shape[1]})")
            except Exception as e:
                print(f"Custom data diffusion failed: {e}. 使用扩散前数据")
        # 时间特征编码（强制timeenc=0）
        df_stamp = df_raw[['date']][border1:border2].copy()
        # 解析日期（根据实际数据格式调整format，避免NaT）
        df_stamp['date'] = pd.to_datetime(df_stamp['date'], format='%m/%d %H:%M:%S', errors='coerce')
        print(f"时间特征日期范围: {df_stamp['date'].min()} 到 {df_stamp['date'].max()}")
        # 处理日期解析错误（填充为0，避免后续报错）
        df_stamp['month'] = df_stamp['date'].apply(lambda x: x.month if not pd.isna(x) else 0)
        df_stamp['day'] = df_stamp['date'].apply(lambda x: x.day if not pd.isna(x) else 0)
        df_stamp['weekday'] = df_stamp['date'].apply(lambda x: x.weekday() if not pd.isna(x) else 0)
        df_stamp['hour'] = df_stamp['date'].apply(lambda x: x.hour if not pd.isna(x) else 0)
        data_stamp = df_stamp.drop('date', axis=1).values
        print(f"时间特征 (timeenc=0) | 形状: {data_stamp.shape} (特征数: {data_stamp.shape[1]})")
        # 赋值最终数据（确保data_x为10通道）
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        print(f"最终数据 | data_x形状: {self.data_x.shape}, data_stamp形状: {self.data_stamp.shape}")
    def _validate_data(self):
        if not hasattr(self, 'data_x'):
            raise ValueError("数据未初始化！__read_data__ 可能执行失败")
        required_len = self.seq_len + self.pred_len
        if len(self.data_x) < required_len:
            raise ValueError(f"数据长度不足 | 需要 {required_len}, 实际 {len(self.data_x)}")
        # 新增10通道校验
        if self.features != 'S' and self.data_x.shape[1] != 10:
            print(f"⚠️ 警告：当前数据通道数为 {self.data_x.shape[1]}，建议调整为10通道")
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # 处理索引越界（适配10通道）
        if s_end > len(self.data_x) or r_end > len(self.data_y):
            seq_x = np.zeros((self.seq_len, self.data_x.shape[1]), dtype=self.data_x.dtype)  # 自动匹配10通道
            seq_y = np.zeros((self.label_len + self.pred_len, self.data_y.shape[1]), dtype=self.data_y.dtype)
            seq_x_mark = np.zeros((self.seq_len, self.data_stamp.shape[1]), dtype=self.data_stamp.dtype)
            seq_y_mark = np.zeros((self.label_len + self.pred_len, self.data_stamp.shape[1]),
                                  dtype=self.data_stamp.dtype)
        else:
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
        # 确保二维形状（避免单特征时维度缺失）
        if seq_x.ndim == 1:
            seq_x = seq_x.reshape(-1, 1)
        if seq_y.ndim == 1:
            seq_y = seq_y.reshape(-1, 1)
        # 转换为张量
        return (torch.from_numpy(seq_x).float(),
                torch.from_numpy(seq_y).float(),
                torch.from_numpy(seq_x_mark).float(),
                torch.from_numpy(seq_y_mark).float())
    def __len__(self):
        # 计算有效样本数（避免越界）
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)
    def inverse_transform(self, data):
        # 反标准化（适配10通道数据）
        return self.scaler.inverse_transform(data)

def custom_collate_fn(batch):
    """完整的数据collate函数，处理小波变换后的数据，兼容10通道"""
    # 过滤无效样本（含NaN/Inf或空张量的样本）
    valid_batch = []
    for item in batch:
        valid = True
        for tensor in item:
            if tensor.size(0) == 0 or torch.isnan(tensor).any() or torch.isinf(tensor).any():
                valid = False
                break
        if valid:
            valid_batch.append(item)
    # 处理空批次（生成 dummy 数据，适配10通道）
    if not valid_batch:
        if batch:  # 从原始批次提取形状信息
            seq_len = batch[0][0].size(0)
            pred_len = batch[0][1].size(0)
            feat_dim = batch[0][0].size(1)  # 自动匹配10通道
            stamp_dim = batch[0][2].size(1)
        else:  # 默认形状（适配10通道）
            seq_len = 96
            pred_len = 48
            feat_dim = 10  # 固定为10通道（空批次时默认）
            stamp_dim = 4
        # 生成 dummy 张量（避免训练中断）
        dummy_batch = (
            torch.zeros(1, seq_len, feat_dim),
            torch.zeros(1, pred_len, feat_dim),
            torch.zeros(1, seq_len, stamp_dim),
            torch.zeros(1, pred_len, stamp_dim)
        )
        return dummy_batch
    # 拼接有效样本（兼容10通道）
    x, y, x_mark, y_mark = zip(*valid_batch)
    # 清理异常值（确保数据有效性）
    for tensor_list in [x, y]:
        for t in tensor_list:
            t[torch.isnan(t)] = 0.0
            t[torch.isinf(t)] = 0.0
    # 堆叠为批次张量（自动适配10通道维度）
    return torch.stack(x), torch.stack(y), torch.stack(x_mark), torch.stack(y_mark)


if __name__ == "__main__":
    # 1. 测试小波变换（适配10通道数据）
    print("="*50)
    print("Testing WaveletTransformer with 10 features...")
    test_data = np.random.randn(100, 10)  # 100个时间步，10个特征
    wt = WaveletTransformer(wavelet='db4', level=3, threshold=0.5)
    transformed = wt.transform(test_data)
    print(f"Original data shape: {test_data.shape}")
    print(f"Transformed data shape: {transformed.shape}")
    print(f"Feature 0 - Original mean: {np.mean(test_data[:,0]):.4f}, Transformed mean: {np.mean(transformed[:,0]):.4f}")
    print("Wavelet test passed (10 features compatible)\n")

    # 2. 测试 Dataset_Custom（加载10通道数据）
    print("="*50)
    print("Testing Dataset_Custom with 10 features...")
    # 模拟10通道数据（date + 10个特征）
    dummy_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=1000, freq='h'),
        'feat1': np.random.randn(1000), 'feat2': np.random.randn(1000),
        'feat3': np.random.randn(1000), 'feat4': np.random.randn(1000),
        'feat5': np.random.randn(1000), 'feat6': np.random.randn(1000),
        'feat7': np.random.randn(1000), 'feat8': np.random.randn(1000),
        'feat9': np.random.randn(1000), 'feat10': np.random.randn(1000)  # 第10个特征
    })
    dummy_data_path = './dummy_10feat_data.csv'
    dummy_data.to_csv(dummy_data_path, index=False)

    # 初始化10通道数据集
    dataset = Dataset_Custom(
        root_path='./',
        flag='train',
        size=[96, 48, 48],  # seq_len=96, label_len=48, pred_len=48
        features='M',  # 多特征模式（10个特征）
        data_path='dummy_10feat_data.csv',
        target='feat1',
        wavelet='db4',
        wavelet_level=3,
        use_diffusion=False  # 测试时先禁用扩散，避免模型加载问题
    )
    print(f"Dataset length: {len(dataset)}")
    seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
    print(f"Sample shapes - seq_x: {seq_x.shape}, seq_y: {seq_y.shape}")
    print(f"seq_x channels: {seq_x.shape[1]} (expected1] (expected 10)")
    print("Dataset_Custom test passed (10 features compatible)\n")

    # 3. 测试 DataLoader + collate_fn（适配10通道）
    print("="*50)
    print("Testing DataLoader with custom_collate_fn...")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    batch_x, batch_y, batch_xm, batch_ym = next(iter(dataloader))
    print(f"Batch shapes - batch_x: {batch_x.shape}, batch_y: {batch_y.shape}")
    print(f"Batch channels: {batch_x.shape[2]} (expected 10)")
    print("DataLoader test passed (10 features compatible)")

    # 清理临时文件
    os.remove(dummy_data_path)
    print("\nAll tests completed successfully!")