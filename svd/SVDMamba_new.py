import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat

# 导入Mamba2的相关组件
from .Mamba2 import Mamba2, ssd, get_device, RMSNorm


class DifferentiableSVDPruning(nn.Module):
    """
    可微分奇异值剪枝机制
    创新点：通过可微分方式自动学习并剪枝不重要的奇异值
    """

    def __init__(self, initial_rank, min_rank_ratio=0.1, temperature=0.1):
        super().__init__()
        self.initial_rank = initial_rank
        self.min_rank = max(1, int(initial_rank * min_rank_ratio))
        self.temperature = temperature

        # 可学习的剪枝阈值
        self.threshold = nn.Parameter(torch.tensor(0.5))
        # 奇异值重要性权重
        self.importance_scores = nn.Parameter(torch.ones(initial_rank))

    def forward(self, sigma, U, V):
        # sigma: (rank,), U: (state_dim, rank), V: (rank, state_dim)

        # 计算每个奇异值的相对重要性
        normalized_sigma = F.softmax(sigma, dim=0)
        importance = F.sigmoid(self.importance_scores * normalized_sigma / self.temperature)

        # 可微分剪枝：使用soft mask而不是hard cut
        mask = torch.sigmoid((importance - self.threshold) / self.temperature)

        # 应用mask到奇异值
        pruned_sigma = sigma * mask

        # 重建状态矩阵
        A_pruned = U @ (torch.diag(pruned_sigma) @ V)

        # 计算有效秩（用于监控）
        effective_rank = (mask > 0.5).float().sum()

        return A_pruned, mask, effective_rank


class SVDParamSSM(nn.Module):
    """
    基于SSD机制的SVD参数化状态空间模型
    """

    def __init__(self, d_model, d_state, headdim=16, chunk_size=8,
                 expand=2, rank_ratio=0.5, use_pruning=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.nheads = (expand * d_model) // headdim
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.device = get_device()

        # SVD参数化
        self.rank = max(1, int(d_state * rank_ratio))
        self.use_pruning = use_pruning

        # SVD分解参数
        self.U = nn.Parameter(torch.randn(d_state, self.rank))
        self.sigma = nn.Parameter(torch.ones(self.rank))  # 奇异值
        self.V = nn.Parameter(torch.randn(self.rank, d_state))

        # 可微分剪枝机制
        if use_pruning:
            self.pruner = DifferentiableSVDPruning(self.rank)

        # 输入投影 (与Mamba2兼容)
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, device=self.device)

        # 卷积层
        self.d_conv = 4
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner + 2 * self.d_state,
            out_channels=self.d_inner + 2 * self.d_state,
            kernel_size=self.d_conv,
            groups=self.d_inner + 2 * self.d_state,
            padding=self.d_conv - 1,
            device=self.device,
        )

        self.dt_bias = nn.Parameter(torch.zeros(self.nheads, device=self.device))
        self.D = nn.Parameter(torch.ones(self.nheads, device=self.device))

        self.norm = RMSNorm(self.d_inner, device=self.device)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, device=self.device)

        # 初始化正交矩阵
        with torch.no_grad():
            nn.init.orthogonal_(self.U)
            nn.init.orthogonal_(self.V)

    def get_A_matrix(self):
        """获取SVD参数化的A矩阵"""
        if self.use_pruning:
            A, mask, effective_rank = self.pruner(self.sigma, self.U, self.V)
            self._current_effective_rank = effective_rank
            return A
        else:
            return self.U @ (torch.diag(self.sigma) @ self.V)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # 保存原始序列长度
        original_seq_len = u.shape[1]

        # 填充处理
        if original_seq_len % self.chunk_size != 0:
            pad_len = self.chunk_size - (original_seq_len % self.chunk_size)
            if original_seq_len >= 4:
                pad_values = torch.mean(u[:, :5], dim=1, keepdim=True)
                pad = pad_values.repeat(1, pad_len, 1)
            else:
                pad = u[:, :1].repeat(1, pad_len, 1)
            u = torch.cat([pad, u], dim=1)

        seq_len = u.shape[1]

        # 1. 输入投影
        zxbcdt = self.in_proj(u)

        # 分割
        z = zxbcdt[:, :, :self.d_inner]
        xBC = zxbcdt[:, :, self.d_inner:self.d_inner + self.d_inner + 2 * self.d_state]
        dt = zxbcdt[:, :, -self.nheads:]

        dt = F.softplus(dt + self.dt_bias)

        # 2. 卷积操作
        xBC_t = xBC.transpose(1, 2)
        xBC = self.conv1d(xBC_t).transpose(1, 2)[:, :seq_len, :]
        xBC = F.silu(xBC)

        # 3. 分割xBC
        x = xBC[:, :, :self.d_inner]
        B = xBC[:, :, self.d_inner:self.d_inner + self.d_state]
        C = xBC[:, :, -self.d_state:]

        x = x.view(x.shape[0], seq_len, self.nheads, self.headdim)

        # 4. 使用SVD参数化的A矩阵
        A_matrix = self.get_A_matrix()  # (d_state, d_state)

        # 修复：正确地将A矩阵转换为SSD需要的格式
        # 我们需要为每个head创建一个A值
        # 使用A矩阵的奇异值或者特征值来生成每个head的A值
        with torch.no_grad():
            # 计算A矩阵的奇异值
            _, S, _ = torch.svd(A_matrix)
            # 取前nheads个奇异值，如果不够则用0填充
            if len(S) >= self.nheads:
                A_values = S[:self.nheads]
            else:
                A_values = torch.cat([S, torch.zeros(self.nheads - len(S), device=self.device)])

        # 使用负指数确保稳定性
        A_diag = -torch.exp(A_values.unsqueeze(0).unsqueeze(0))  # (1, 1, nheads)

        # 计算A*dt和x*dt
        A_dt = A_diag * dt  # 现在维度匹配了
        x_dt = x * dt.unsqueeze(-1)

        # 重排B和C
        B_reshaped = B.unsqueeze(2)  # (b l 1 d_state)
        C_reshaped = C.unsqueeze(2)  # (b l 1 d_state)

        # 调用SSD
        y = ssd(
            x=x_dt,
            A=A_dt,
            B=B_reshaped,
            C=C_reshaped,
            chunk_size=self.chunk_size,
            device=self.device,
        )

        # 5. 残差融合
        D_expanded = self.D.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        y = y + x * D_expanded

        # 重排回原始形状
        y = y.reshape(y.shape[0], seq_len, self.d_inner)

        # 6. 归一化与输出投影
        y = self.norm(y, z)
        y = self.out_proj(y)

        # 裁剪填充
        if original_seq_len != seq_len:
            pad_len = seq_len - original_seq_len
            y = y[:, pad_len:, :]

        return y


class TuckerConv3d(nn.Module):
    """
    轻量级3D卷积 via Tucker分解
    """

    def __init__(self, in_chans, out_chans, kernel_size=(3, 3, 3), rank_ratio=0.25):
        super().__init__()
        self.kernel_size = kernel_size
        k1, k2, k3 = kernel_size

        # 计算秩
        rank1 = max(1, int(k1 * rank_ratio))
        rank2 = max(1, int(k2 * rank_ratio))
        rank3 = max(1, int(k3 * rank_ratio))

        # 核心张量和因子矩阵
        self.core = nn.Parameter(torch.randn(rank1, rank2, rank3, in_chans, out_chans))
        self.factor1 = nn.Parameter(torch.randn(k1, rank1))
        self.factor2 = nn.Parameter(torch.randn(k2, rank2))
        self.factor3 = nn.Parameter(torch.randn(k3, rank3))

    def forward(self, x):
        # 重建卷积核
        kernel = torch.einsum('pqrio,ap->aqrio', self.core, self.factor1)
        kernel = torch.einsum('aqrio,bq->abrio', kernel, self.factor2)
        kernel = torch.einsum('abrio,cr->abcio', kernel, self.factor3)

        # 重排为PyTorch conv3d格式
        kernel = kernel.permute(4, 3, 0, 1, 2)

        return F.conv3d(x, kernel, padding=[i // 2 for i in self.kernel_size])


class AdaptiveFeatureFusion(nn.Module):
    """
    自适应特征融合机制
    创新点：根据输入特性动态调整SVD-Mamba和Tucker-Head的贡献权重
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # 门控机制
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )

        # 特征变换
        self.transform_global = nn.Linear(d_model, d_model)
        self.transform_local = nn.Linear(d_model, d_model)

    def forward(self, global_feat, local_feat):
        # global_feat: (batch, h, w, d_model)
        # local_feat: (batch, h, w, d_model)

        batch, h, w, _ = global_feat.shape

        # 计算全局描述符
        global_desc = F.adaptive_avg_pool2d(
            global_feat.permute(0, 3, 1, 2), (1, 1)
        ).view(batch, self.d_model)

        local_desc = F.adaptive_avg_pool2d(
            local_feat.permute(0, 3, 1, 2), (1, 1)
        ).view(batch, self.d_model)

        # 计算融合权重
        desc_cat = torch.cat([global_desc, local_desc], dim=-1)
        gates = self.gate_net(desc_cat)  # (batch, 2)

        # 应用特征变换
        global_trans = self.transform_global(global_feat)
        local_trans = self.transform_local(local_feat)

        # 自适应融合
        gate_global = gates[:, 0].view(batch, 1, 1, 1)
        gate_local = gates[:, 1].view(batch, 1, 1, 1)

        fused_feat = gate_global * global_trans + gate_local * local_trans

        return fused_feat, gates


class SVDMamba(nn.Module):
    """
    改进的SVD-Mamba模型，集成三个创新点
    """

    def __init__(self, input_channels, num_classes, num_layers=2, d_model=64,
                 d_state=16, use_pruning=True, rank_ratio=0.5):
        super().__init__()
        self.input_channels = input_channels
        self.d_model = d_model
        self.use_pruning = use_pruning
        self.rank_ratio = rank_ratio  # 保存rank_ratio

        # Stem网络
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # 输入投影
        self.proj = nn.Linear(128, d_model)

        # SVD-Mamba层
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'svd_ssm': SVDParamSSM(
                    d_model=d_model,
                    d_state=d_state,
                    use_pruning=use_pruning,
                    rank_ratio=rank_ratio  # 传递rank_ratio
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'mlp': nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model)
                )
            }) for _ in range(num_layers)
        ])

        # Tucker-Head
        self.tucker_head = nn.Sequential(
            TuckerConv3d(in_chans=d_model, out_chans=d_model // 2, kernel_size=(3, 3, 3)),
            nn.GELU(),
            TuckerConv3d(in_chans=d_model // 2, out_chans=d_model // 2, kernel_size=(5, 5, 5)),
            nn.GELU(),
            TuckerConv3d(in_chans=d_model // 2, out_chans=d_model, kernel_size=(7, 7, 7)),
        )

        # 自适应特征融合
        self.fusion = AdaptiveFeatureFusion(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        # 监控变量
        self.register_buffer('effective_ranks', torch.zeros(num_layers))
        self.register_buffer('fusion_gates', torch.zeros(2))

    def forward(self, x, visualize=False):
        # 输入处理 - 修复维度问题
        if x.dim() == 5:
            # 如果是5维，可能是 (batch, 1, channels, height, width)
            if x.shape[1] == 1:
                x = x.squeeze(1)
            else:
                # 取第一个深度切片
                x = x[:, 0, :, :, :]

        # 确保输入是4维的 (batch, channels, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # 检查并调整通道维度
        if x.shape[1] != self.input_channels:
            # 如果通道维度不在第1维，尝试调整
            if x.shape[3] == self.input_channels:  # 通道在最后一维
                x = x.permute(0, 3, 1, 2)
            elif x.shape[2] == self.input_channels:  # 通道在第二维但不是第一维
                x = x.permute(0, 2, 1, 3)

        # 继续原来的处理
        x = self.stem(x)
        x = x.permute(0, 2, 3, 1)

        batch, h, w, c = x.shape
        x_flat = x.reshape(batch, h * w, c)

        # 输入投影
        z = self.proj(x_flat)

        # SVD-Mamba分支
        z_residual = z
        for i, layer in enumerate(self.layers):
            # SVD-SSM块
            z_norm = layer['norm1'](z_residual)
            z_ssm = layer['svd_ssm'](z_norm)
            z_ssm = z_ssm + z_residual

            # MLP块
            z_norm2 = layer['norm2'](z_ssm)
            z_mlp = layer['mlp'](z_norm2)
            z_residual = z_mlp + z_ssm

            # 记录有效秩
            if self.use_pruning and hasattr(layer['svd_ssm'], '_current_effective_rank'):
                self.effective_ranks[i] = layer['svd_ssm']._current_effective_rank

        z_global = z_residual.reshape(batch, h, w, self.d_model)

        # Tucker-Head分支
        z_local = z_global.permute(0, 3, 1, 2).unsqueeze(2)
        z_local = self.tucker_head(z_local).squeeze(2).permute(0, 2, 3, 1)

        # 自适应特征融合
        z_out, gates = self.fusion(z_global, z_local)
        self.fusion_gates = gates.mean(dim=0)  # 记录平均门控值

        # 分类
        logits = self.classifier(z_out.mean(dim=[1, 2]))

        if visualize:
            features = {
                'global': z_global.detach(),
                'local': z_local.detach(),
                'combined': z_out.detach(),
                'gates': gates.detach()
            }
            return logits, features

        return logits

    def get_diagnostics(self):
        """获取模型诊断信息"""
        diagnostics = {
            'effective_ranks': self.effective_ranks,
            'fusion_gates': self.fusion_gates,
        }

        # 安全地获取奇异值
        singular_values = []
        for layer in self.layers:
            if hasattr(layer['svd_ssm'], 'sigma'):
                singular_values.append(layer['svd_ssm'].sigma.detach())
        diagnostics['singular_values'] = singular_values

        return diagnostics


# 测试函数
def test_enhanced_svdmamba():
    """测试增强版SVDMamba"""
    print("测试增强版SVDMamba...")

    # 创建模型
    model = SVDMamba(
        input_channels=100,
        num_classes=10,
        num_layers=2,
        d_model=64,
        d_state=16,
        use_pruning=True
    )

    device = get_device()
    model = model.to(device)

    # 测试输入 - 确保序列长度是chunk_size的倍数
    x = torch.randn(2, 100, 15, 15).to(device)  # 8x8 = 64，是8的倍数

    # 前向传播
    logits, features = model(x, visualize=True)

    print(f"输入形状: {x.shape}")
    print(f"输出logits形状: {logits.shape}")
    print(f"全局特征形状: {features['global'].shape}")
    print(f"局部特征形状: {features['local'].shape}")
    print(f"融合门控值: {features['gates'].mean(dim=0)}")

    # 获取诊断信息
    diagnostics = model.get_diagnostics()
    print(f"有效秩: {diagnostics['effective_ranks']}")
    print(f"平均融合门控: {diagnostics['fusion_gates']}")

    # 测试训练步骤
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    y = torch.randint(0, 10, (2,)).to(device)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    print(f"训练损失: {loss.item():.4f}")
    print("增强版SVDMamba测试通过!")

    return model, features


if __name__ == "__main__":
    model, features = test_enhanced_svdmamba()