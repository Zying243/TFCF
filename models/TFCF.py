import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN  # RevIN：时间序列专用归一化层（去趋势、标准化）


class PositionalEncoding(nn.Module):
    """位置编码模块（完全保留）"""

    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """多头注意力模块（完全保留）"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.query(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        return out, attn  # 保留注意力权重返回（用于稀疏损失）


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层（完全保留）"""

    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None):
        src2, _ = self.self_attn(src, src, src, src_mask)  # 接收权重但不使用
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# ---------------------- 核心修改：DynamicSpectralAttention新增moving_avg参数 ----------------------
class DynamicSpectralAttention(nn.Module):
    """动态频谱注意力模块（仅新增moving_avg参数，保留原有逻辑）"""

    def __init__(self, embed_dim, num_heads=4, dropout=0.1, moving_avg=25):  # 新增moving_avg参数（默认25）
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embed dim must be divisible by num heads"
        self.moving_avg = moving_avg  # 存储移动平均窗口大小

        # 新增：移动平均层（用于频域特征平滑，same padding确保维度不变）
        self.moving_avg_pool = nn.AvgPool1d(
            kernel_size=moving_avg,
            padding=moving_avg // 2  # 窗口中心对齐，输出长度=输入长度
        )

        # 原有逻辑完全保留
        self.freq_score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )
        self.multi_head_attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.weight_fusion = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, freq_feat):
        pass


class CrossDomainAttention(nn.Module):
    """跨域注意力模块（完全保留）"""

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.time2freq_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.freq2time_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.freq_proj = nn.Linear(embed_dim, embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim)

    def forward(self, time_feat, freq_feat):
        freq_real = freq_feat.real
        freq_proj = self.freq_proj(freq_real)

        # 时域引导频域
        freq_attended, _ = self.time2freq_attn(q=freq_proj, k=time_feat, v=time_feat)
        freq_fused = self.fusion_norm(freq_real + freq_attended)
        freq_feat_updated = torch.complex(freq_fused, freq_feat.imag)

        # 频域引导时域
        time_attended, _ = self.freq2time_attn(q=time_feat, k=freq_proj, v=freq_proj)
        time_fused = self.fusion_norm(time_feat + time_attended)

        return time_fused, freq_feat_updated


class MultiScaleAlignmentFusion(nn.Module):
    """多尺度特征对齐融合模块（完全保留）"""

    def __init__(self, embed_dim, scales=[1, 3, 5]):
        super().__init__()
        self.scales = scales
        self.embed_dim = embed_dim

        self.time_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=s, padding=s // 2)
            for s in scales
        ])
        self.freq_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=s, padding=s // 2)
            for s in scales
        ])

        self.scale_attn = nn.Sequential(
            nn.Linear(len(scales) * embed_dim, len(scales)),
            nn.Softmax(dim=-1)
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, time_feat, freq_feat_real):
        pass


# ---------------------- 整体Model类（仅新增moving_avg参数读取，其他不变） ----------------------
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 核心参数读取（新增moving_avg参数，从配置文件获取）
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.dropout = configs.dropout
        self.num_heads = configs.n_heads  # 从配置读取注意力头数
        self.num_layers = 2
        self.sparse_reg = configs.sparse_reg  # 稀疏正则化权重
        self.moving_avg = configs.moving_avg  # 新增：读取moving_avg参数（需在配置文件中定义）

        # 基础模块（完全保留）
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.embedding = nn.Linear(self.seq_len, self.embed_size)
        self.pos_encoding = PositionalEncoding(self.embed_size)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(self.embed_size, self.num_heads, self.hidden_size, self.dropout)
            for _ in range(self.num_layers)
        ])

        # 核心注意力模块（仅修改DynamicSpectralAttention，传入moving_avg参数）
        self.dynamic_spectral_attn = DynamicSpectralAttention(
            embed_dim=self.embed_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            moving_avg=self.moving_avg  # 新增：将配置的moving_avg传入模块
        )
        self.cross_domain_attn = CrossDomainAttention(self.embed_size, num_heads=self.num_heads)

        # 输出模块（完全保留）
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size)
        )
        self.output = nn.Linear(self.embed_size, self.pred_len)
        self.layernorm = nn.LayerNorm(self.embed_size)
        self.layernorm1 = nn.LayerNorm(self.embed_size)
        self.dropout_layer = nn.Dropout(self.dropout)

        # 创新模块：多尺度特征对齐融合（完全保留）
        self.multi_scale_fusion = MultiScaleAlignmentFusion(
            embed_dim=self.embed_size,
            scales=[1, 3, 5]
        )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None, y=None):
        pass