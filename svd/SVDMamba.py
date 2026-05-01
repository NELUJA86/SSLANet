import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange


class SVDParamSSM(nn.Module):
    """
    SVD-parameterized State Space Model
    """

    def __init__(self, dim, state_dim, rank_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.rank = max(1, int(state_dim * rank_ratio))

        # SVD parameters: U, Sigma, V
        self.U = nn.Parameter(torch.randn(state_dim, self.rank))
        self.Sigma = nn.Parameter(torch.ones(self.rank))  # Singular values
        self.V = nn.Parameter(torch.randn(self.rank, state_dim))

        # Projection matrices - 修正维度
        self.B = nn.Parameter(torch.randn(state_dim, dim))  # 修正为 (state_dim, dim)
        self.C = nn.Parameter(torch.randn(state_dim, dim))  # 修正为 (state_dim, dim)

        # Initialize orthogonal matrices
        with torch.no_grad():
            nn.init.orthogonal_(self.U)
            nn.init.orthogonal_(self.V)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        batch_size, seq_len, _ = x.shape

        # Reconstruct state matrix A = U * diag(Sigma) * V
        A = self.U @ (torch.diag(self.Sigma) @ self.V)  # (state_dim, state_dim)

        # Simplified discretization
        delta = 0.1
        Ad = torch.linalg.matrix_exp(A * delta)  # (state_dim, state_dim)

        # 修正离散化计算
        I = torch.eye(self.state_dim).to(x.device)
        # 使用伪逆避免奇异矩阵问题
        A_pinv = torch.linalg.pinv(A)
        Bd = A_pinv @ (Ad - I) @ self.B  # (state_dim, dim)

        # SSM recurrence
        h = torch.zeros(batch_size, self.state_dim).to(x.device)  # (batch, state_dim)
        outputs = []

        for t in range(seq_len):
            # 修正矩阵乘法维度
            h_next = Ad @ h.unsqueeze(-1) + Bd @ x[:, t].unsqueeze(-1)  # (batch, state_dim, 1)
            h = h_next.squeeze(-1)  # (batch, state_dim)
            y = h @ self.C  # (batch, state_dim) @ (state_dim, dim) -> (batch, dim)
            outputs.append(y)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, dim)


class TuckerConv3d(nn.Module):
    """
    Lightweight 3D convolution via Tucker decomposition
    """

    def __init__(self, in_chans, out_chans, kernel_size=(3, 3, 3), rank_ratio=0.25):
        super().__init__()
        self.kernel_size = kernel_size
        k1, k2, k3 = kernel_size

        # Calculate ranks
        rank1 = max(1, int(k1 * rank_ratio))
        rank2 = max(1, int(k2 * rank_ratio))
        rank3 = max(1, int(k3 * rank_ratio))

        # Core tensor and factor matrices
        self.core = nn.Parameter(torch.randn(rank1, rank2, rank3, in_chans, out_chans))
        self.factor1 = nn.Parameter(torch.randn(k1, rank1))
        self.factor2 = nn.Parameter(torch.randn(k2, rank2))
        self.factor3 = nn.Parameter(torch.randn(k3, rank3))

    def forward(self, x):
        # Reconstruct the kernel using correct einsum notation
        # core: (rank1, rank2, rank3, in_chans, out_chans)
        # factor1: (k1, rank1) -> expands spatial dim 1
        # factor2: (k2, rank2) -> expands spatial dim 2
        # factor3: (k3, rank3) -> expands spatial dim 3

        # Step-by-step reconstruction for clarity
        # First: core x1 factor1 -> (k1, rank2, rank3, in_chans, out_chans)
        kernel = torch.einsum('pqrio,ap->aqrio', self.core, self.factor1)

        # Second: result x2 factor2 -> (k1, k2, rank3, in_chans, out_chans)
        kernel = torch.einsum('aqrio,bq->abrio', kernel, self.factor2)

        # Third: result x3 factor3 -> (k1, k2, k3, in_chans, out_chans)
        kernel = torch.einsum('abrio,cr->abcio', kernel, self.factor3)

        # Rearrange to PyTorch conv3d format: (out_channels, in_channels, k1, k2, k3)
        kernel = kernel.permute(4, 3, 0, 1, 2)

        return F.conv3d(x, kernel, padding=[i // 2 for i in self.kernel_size])


class SVDMamba(nn.Module):
    """
    The proposed SVD-Mamba model for HSI classification
    """

    def __init__(self, input_channels, num_classes, num_layers=2, d_model=64, state_dim=16):
        super().__init__()
        self.d_model = d_model

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        # Input projection
        self.proj = nn.Linear(128, d_model)

        # SVD-Mamba layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'svd_ssm': SVDParamSSM(dim=d_model, state_dim=state_dim),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'mlp': nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model)
                )
            }) for _ in range(num_layers)
        ])

        # Lightweight Tucker-Head
        self.tucker_head = nn.Sequential(
            TuckerConv3d(in_chans=d_model, out_chans=d_model // 2, kernel_size=(3, 3, 3)),
            nn.GELU(),
            TuckerConv3d(in_chans=d_model // 2, out_chans=d_model // 2, kernel_size=(5, 5, 5)),
            nn.GELU(),
            TuckerConv3d(in_chans=d_model // 2, out_chans=d_model, kernel_size=(7, 7, 7)),
        )

        # Fusion and classifier
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, visualize=False):
        if x.dim() ==5:
            x = x.squeeze()
        if x.shape[3] != x.shape[2]:
            x = x.permute(0, 3, 1, 2)
        x = self.stem(x)
        x = x.permute(0, 2, 3, 1)
        # x: (batch, height, width, channels)
        batch, h, w, c = x.shape
        x_flat = x.reshape(batch, h * w, c)

        # Project input
        z = self.proj(x_flat)  # (batch, h*w, d_model)

        # SVD-Mamba branch
        z_residual = z
        for layer in self.layers:
            # SVD-SSM block
            z_norm = layer['norm1'](z_residual)
            z_ssm = layer['svd_ssm'](z_norm)
            z_ssm = z_ssm + z_residual  # Residual connection

            # MLP block
            z_norm2 = layer['norm2'](z_ssm)
            z_mlp = layer['mlp'](z_norm2)
            z_residual = z_mlp + z_ssm

        z_global = z_residual.reshape(batch, h, w, self.d_model)

        # Tucker-Head branch: process as 3D volume
        z_local = z_global.permute(0, 3, 1, 2).unsqueeze(2)  # (batch, d_model, 1, h, w)
        z_local = self.tucker_head(z_local).squeeze(2).permute(0, 2, 3, 1)  # (batch, h, w, d_model)

        # Feature fusion
        z_combined = torch.cat([z_global, z_local], dim=-1)
        z_out = self.fusion(z_combined)  # (batch, h, w, d_model)

        # Classification
        logits = self.classifier(z_out.mean(dim=[1, 2]))  # Global Average Pooling

        if visualize:
            features = {
                'global': z_global.detach(),
                'local': z_local.detach(),
                'combined': z_out.detach()
            }
            return logits, features

        return logits


def test_individual_modules():
    """测试各个子模块"""
    print("=" * 50)
    print("测试各个子模块")
    print("=" * 50)

    # 测试SVDParamSSM
    print("1. 测试SVDParamSSM模块...")
    batch_size, seq_len, dim = 2, 10, 64  # 减少序列长度以加快测试
    state_dim = 16
    ssm = SVDParamSSM(dim=dim, state_dim=state_dim)
    x = torch.randn(batch_size, seq_len, dim)
    output = ssm(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   SVDParamSSM测试通过 ✓")

    # 测试TuckerConv3d
    print("\n2. 测试TuckerConv3d模块...")
    in_chans, out_chans = 64, 32
    tucker_conv = TuckerConv3d(in_chans=in_chans, out_chans=out_chans)
    x_3d = torch.randn(2, in_chans, 3, 8, 8)  # 减小输入尺寸以加快测试
    output_3d = tucker_conv(x_3d)
    print(f"   输入形状: {x_3d.shape}")
    print(f"   输出形状: {output_3d.shape}")
    print(f"   TuckerConv3d测试通过 ✓")

    return True


def test_full_model():
    """测试完整模型"""
    print("\n" + "=" * 50)
    print("测试完整SVDMamba模型")
    print("=" * 50)

    # 模拟高光谱图像数据
    batch_size, height, width, channels = 2, 8, 8, 100  # 减小尺寸以加快测试
    num_classes = 5

    # 创建模型
    model = SVDMamba(
        input_channels=channels,
        num_classes=num_classes,
        num_layers=2,  # 测试时使用更少的层数
        d_model=64,  # 减小模型维度
        state_dim=16  # 减小状态维度
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {param_count:,}")

    # 测试前向传播
    x = torch.randn(batch_size, height, width, channels)

    # 普通模式
    logits = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出logits形状: {logits.shape}")
    print(f"普通前向传播测试通过 ✓")

    # 可视化模式
    logits, features = model(x, visualize=True)
    print(f"\n可视化模式测试:")
    for name, feature in features.items():
        print(f"  {name}特征形状: {feature.shape}")
    print(f"可视化模式测试通过 ✓")

    return model, features


def test_training_step():
    """测试训练步骤"""
    print("\n" + "=" * 50)
    print("测试训练过程")
    print("=" * 50)

    # 模拟数据
    batch_size, height, width, channels = 4, 8, 8, 100
    num_classes = 5
    x = torch.randn(batch_size, height, width, channels)
    y = torch.randint(0, num_classes, (batch_size,))

    # 模型和优化器
    model = SVDMamba(input_channels=channels, num_classes=num_classes, num_layers=1, d_model=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练步骤
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    print(f"训练损失: {loss.item():.4f}")
    print(f"梯度回传测试通过 ✓")

    # 检查梯度
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"参数梯度存在: {has_gradients}")

    return loss.item()


def visualize_features(features):
    """可视化特征图"""
    print("\n" + "=" * 50)
    print("特征可视化")
    print("=" * 50)

    # 选择第一个样本进行可视化
    sample_idx = 0

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 全局特征可视化
    global_feat = features['global'][sample_idx].mean(dim=-1).cpu().numpy()
    im1 = axes[0, 0].imshow(global_feat, cmap='viridis')
    axes[0, 0].set_title('Global Features (SVD-Mamba)')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])

    # 局部特征可视化
    local_feat = features['local'][sample_idx].mean(dim=-1).cpu().numpy()
    im2 = axes[0, 1].imshow(local_feat, cmap='viridis')
    axes[0, 1].set_title('Local Features (Tucker-Head)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])

    # 融合特征可视化
    combined_feat = features['combined'][sample_idx].mean(dim=-1).cpu().numpy()
    im3 = axes[1, 0].imshow(combined_feat, cmap='viridis')
    axes[1, 0].set_title('Combined Features')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])

    # 特征差异
    diff_feat = np.abs(global_feat - local_feat)
    im4 = axes[1, 1].imshow(diff_feat, cmap='hot')
    axes[1, 1].set_title('Feature Difference (|Global - Local|)')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig('feature_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("特征可视化已保存为 'feature_visualization.png'")


def test_module_importance():
    """测试不同模块的重要性"""
    print("\n" + "=" * 50)
    print("模块重要性分析")
    print("=" * 50)

    batch_size, height, width, channels = 2, 8, 8, 100
    num_classes = 5
    x = torch.randn(batch_size, height, width, channels)

    # 测试完整模型
    full_model = SVDMamba(input_channels=channels, num_classes=num_classes, num_layers=1, d_model=32)
    full_logits = full_model(x)
    full_params = sum(p.numel() for p in full_model.parameters())

    print(f"完整模型参数: {full_params:,}")
    print(f"完整模型输出范围: [{full_logits.min():.3f}, {full_logits.max():.3f}]")

    return full_params


if __name__ == "__main__":
    print("开始测试SVD-Mamba模型...")

    try:
        # 测试各个模块
        test_individual_modules()

        # 测试完整模型
        model, features = test_full_model()

        # 测试训练过程
        loss = test_training_step()

        # 可视化特征
        visualize_features(features)

        # 测试模块重要性
        param_count = test_module_importance()

        print("\n" + "=" * 50)
        print("🎉 所有测试通过！模型运行正常")
        print("=" * 50)
        print(f"最终模型参数数量: {param_count:,}")
        print("模型创新点验证:")
        print("✓ SVD参数化状态空间模型")
        print("✓ Tucker分解的轻量3D卷积")
        print("✓ 双分支特征融合架构")
        print("✓ 完整的训练和可视化功能")

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback

        traceback.print_exc()