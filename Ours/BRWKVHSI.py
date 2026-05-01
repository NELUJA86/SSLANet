import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from thop import profile
import os
from datetime import datetime


class AdaptiveBinaryRWKV(nn.Module):
    """
    自适应二值RWKV模型 for 高光谱图像分类
    """

    def __init__(self, channels, token_num=8, use_residual=True, group_num=4, mamba_type='both'):
        super().__init__()

        self.mamba_type = mamba_type
        self.token_num = token_num
        self.use_residual = use_residual
        self.channels = channels

        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        # 光谱路径二值编码器
        self.spectral_encoder = SpectralBinaryEncoder(channels, self.channel_num)

        # 空间路径二值编码器
        self.spatial_encoder = SpatialBinaryEncoder(channels, self.channel_num)

        # 简化的多尺度特征金字塔
        self.feature_pyramid = SimplifiedFeaturePyramid(self.channel_num)

        # 二值RWKV层
        self.rwkv_layers = nn.ModuleList([
            BinaryRWKVBlock(self.channel_num, self.channel_num, 8)
            for _ in range(2)  # 减少层数避免复杂维度问题
        ])

        # 自适应温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

        # 用于存储中间特征
        self.spectral_features = None
        self.spatial_features = None
        self.pyramid_features = None
        self.binary_masks = []

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x[:, :self.channel_num]  # 确保不超过channel_num

    def forward(self, x):
        x_pad = self.padding_feature(x)
        B, C, H, W = x_pad.shape

        # 双路径编码
        spectral_feat = self.spectral_encoder(x_pad)  # [B, H*W, D]
        spatial_feat = self.spatial_encoder(x_pad)  # [B, H*W, D]

        # 存储中间特征用于可视化
        self.spectral_features = spectral_feat.detach()
        self.spatial_features = spatial_feat.detach()

        # 特征融合
        if self.mamba_type == 'spe':
            fused_feat = spectral_feat
        elif self.mamba_type == 'spa':
            fused_feat = spatial_feat
        else:  # both
            fused_feat = (spectral_feat + spatial_feat) / 2  # 平均融合

        # 多尺度特征金字塔
        pyramid_feats = self.feature_pyramid(fused_feat)
        self.pyramid_features = pyramid_feats.detach()

        # 二值RWKV处理
        current_feat = pyramid_feats
        self.binary_masks = []

        for layer in self.rwkv_layers:
            current_feat, binary_mask = layer(current_feat, self.temperature)
            self.binary_masks.append(binary_mask.detach())

        # 重塑回空间格式
        B, L, D = current_feat.shape
        H = W = int(math.sqrt(L))
        x_recon = current_feat.view(B, H, W, D)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()

        x_proj = self.proj(x_recon)
        if self.use_residual:
            # 确保残差连接的维度匹配
            if x.shape[1] != x_proj.shape[1]:
                x_residual = self.padding_feature(x)
            else:
                x_residual = x
            return x_residual + x_proj
        else:
            return x_proj


class SpectralBinaryEncoder(nn.Module):
    """光谱路径二值编码器"""

    def __init__(self, spectral_dim, hidden_dim):
        super().__init__()

        self.spectral_proj = nn.Linear(spectral_dim, hidden_dim)
        self.spectral_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # 重塑为 [B, H*W, C]
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        # 光谱投影
        spectral_feat = self.spectral_proj(x_flat)
        spectral_feat = self.spectral_norm(spectral_feat)

        return spectral_feat


class SpatialBinaryEncoder(nn.Module):
    """空间路径二值编码器"""

    def __init__(self, spatial_dim, hidden_dim):
        super().__init__()

        # 使用深度可分离卷积提高效率
        self.local_binary_conv = nn.Sequential(
            nn.Conv2d(spatial_dim, spatial_dim, 3, padding=1, groups=spatial_dim),
            nn.Conv2d(spatial_dim, hidden_dim, 1),
            nn.GELU()
        )
        self.spatial_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # 空间特征提取
        local_feat = self.local_binary_conv(x)  # [B, D, H, W]

        # 重塑
        spatial_feat = local_feat.permute(0, 2, 3, 1).contiguous().view(B, H * W, -1)
        spatial_feat = self.spatial_norm(spatial_feat)

        return spatial_feat


class BinaryRWKVBlock(nn.Module):
    """二值化的RWKV块"""

    def __init__(self, dim, hidden_dim, num_heads):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        # RWKV的时间混合组件
        self.time_mix = TimeMixBinary(dim, num_heads)

        # RWKV的通道混合组件
        self.channel_mix = ChannelMixBinary(dim, hidden_dim)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, temperature=1.0):
        # 时间混合 + 残差
        x_out, binary_mask = self.time_mix(self.ln1(x), temperature)
        x = x + x_out

        # 通道混合 + 残差
        x = x + self.channel_mix(self.ln2(x), temperature)

        return x, binary_mask


class TimeMixBinary(nn.Module):
    """二值化时间混合模块"""

    def __init__(self, dim, num_heads):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        # RWKV参数
        self.r = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        # 二值化门控
        self.binary_gate = nn.Linear(dim, 2)  # 输出2维用于二值选择

        # 输出投影
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, x, temperature):
        B, L, D = x.shape

        # 计算RWKV组件
        r = self.r(x)
        k = self.k(x)
        v = self.v(x)

        # Gumbel-Softmax二值化门控 - 简化版本
        binary_gate_logits = self.binary_gate(x)  # [B, L, 2]
        binary_mask = F.gumbel_softmax(
            binary_gate_logits, tau=temperature, hard=True, dim=-1
        )[:, :, 0:1]  # 取第一个维度作为二值掩码 [B, L, 1]

        # 应用二值掩码
        r_binary = r * binary_mask
        k_binary = k * binary_mask

        # 简化的RWKV注意力
        wkv = torch.softmax(k_binary, dim=-1) * v
        wkv = torch.tanh(r_binary) * wkv

        output = self.output_proj(wkv)

        return output, binary_mask.detach()


class ChannelMixBinary(nn.Module):
    """二值化通道混合模块"""

    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim

        # 通道混合组件
        self.key = nn.Linear(dim, hidden_dim)
        self.value = nn.Linear(dim, hidden_dim)
        self.receptance = nn.Linear(dim, dim)

        self.output_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x, temperature):
        # 通道混合前向传播
        k = self.key(x)
        v = self.value(x)
        r = torch.sigmoid(self.receptance(x))

        # 简化的通道混合，移除复杂的二值化避免维度问题
        kv = torch.sigmoid(k) * v

        output = r * self.output_proj(kv)

        return output


class SimplifiedFeaturePyramid(nn.Module):
    """
    简化的多尺度特征金字塔
    避免复杂的跨尺度融合导致的维度问题
    """

    def __init__(self, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 多尺度卷积 - 使用不同核大小
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
        ])

        # 自适应权重学习
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, len(self.scale_convs)),
            nn.Softmax(dim=-1)
        )

        # 特征融合
        self.fusion = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape

        # 转置用于卷积
        x_t = x.transpose(1, 2)  # [B, D, L]

        # 提取多尺度特征
        scale_features = []
        for conv in self.scale_convs:
            scale_feat = conv(x_t)  # [B, D, L]
            scale_feat = F.gelu(scale_feat)
            scale_feat = scale_feat.transpose(1, 2)  # [B, L, D]
            scale_features.append(scale_feat)

        # 自适应尺度权重
        global_feat = x.mean(dim=1)  # [B, D]
        scale_weights = self.scale_attention(global_feat)  # [B, num_scales]

        # 加权融合多尺度特征
        fused_feat = torch.zeros_like(x)
        for i, feat in enumerate(scale_features):
            weight = scale_weights[:, i].view(B, 1, 1)  # [B, 1, 1]
            fused_feat = fused_feat + feat * weight

        # 特征融合
        output = self.fusion(fused_feat)

        return output + x  # 残差连接


class BinaryRWKVHSI(nn.Module):
    """
    基于二值RWKV的高光谱图像分类模型
    """

    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10,
                 use_residual=True, mamba_type='both', token_num=4,
                 group_num=4):
        super().__init__()
        self.mamba_type = mamba_type
        self.hidden_dim = hidden_dim

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
            # nn.GroupNorm(group_num, hidden_dim),
            nn.GELU()
        )

        # 使用二值RWKV - 简化结构
        self.binary_blocks = nn.ModuleList([
            AdaptiveBinaryRWKV(hidden_dim, token_num=token_num,
                               use_residual=use_residual, group_num=group_num,
                               mamba_type=mamba_type)
            for _ in range(3)
        ])

        self.pool_layers = nn.ModuleList([
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            for _ in range(2)
        ])

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 5:
            x = x.squeeze()
        x = self.patch_embedding(x)

        # 依次通过二值块和池化层
        for i, block in enumerate(self.binary_blocks):
            x = block(x)
            if i < len(self.pool_layers):
                x = self.pool_layers[i](x)

        logits = self.cls_head(x)
        return logits

    def get_features(self, x):
        """获取中间特征用于可视化"""
        features = {}

        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 5:
            x = x.squeeze()

        features['input'] = x.detach().cpu()
        x = self.patch_embedding(x)
        features['patch_embedding'] = x.detach().cpu()

        # 依次通过二值块和池化层
        for i, block in enumerate(self.binary_blocks):
            x = block(x)
            features[f'binary_block_{i}'] = x.detach().cpu()
            if i < len(self.pool_layers):
                x = self.pool_layers[i](x)
                features[f'pool_{i}'] = x.detach().cpu()

        # 获取每个二值块的中间特征
        for i, block in enumerate(self.binary_blocks):
            if hasattr(block, 'spectral_features') and block.spectral_features is not None:
                features[f'spectral_{i}'] = block.spectral_features.detach().cpu()
            if hasattr(block, 'spatial_features') and block.spatial_features is not None:
                features[f'spatial_{i}'] = block.spatial_features.detach().cpu()
            if hasattr(block, 'pyramid_features') and block.pyramid_features is not None:
                features[f'pyramid_{i}'] = block.pyramid_features.detach().cpu()
            if hasattr(block, 'binary_masks') and block.binary_masks:
                for j, mask in enumerate(block.binary_masks):
                    features[f'binary_mask_{i}_{j}'] = mask.detach().cpu()

        return features


class FeatureVisualizer:
    """特征可视化工具类"""

    def __init__(self, save_dir='visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def visualize_features(self, features, title_suffix=""):
        """可视化模型中间特征"""
        n_features = len(features)
        cols = 4
        rows = math.ceil(n_features / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (name, feature) in enumerate(features.items()):
            row = idx // cols
            col = idx % cols

            ax = axes[row, col]

            # 处理不同形状的特征
            if feature.dim() == 4:  # [B, C, H, W]
                # 取第一个样本，平均所有通道
                feat_img = feature[0].mean(dim=0)
            elif feature.dim() == 3:  # [B, L, D] 或 [B, H*W, D]
                # 重塑为2D
                B, L, D = feature.shape
                H = W = int(math.sqrt(L))
                if H * W == L:
                    feat_img = feature[0].view(H, W, D).mean(dim=2)
                else:
                    # 如果不能重塑为正方形，则使用第一个样本的第一个通道
                    feat_img = feature[0, :, 0].view(-1, 1)
            elif feature.dim() == 2:  # [B, D]
                feat_img = feature[0].unsqueeze(0).unsqueeze(0)
            else:
                continue

            # 可视化
            im = ax.imshow(feat_img.cpu().numpy(), cmap='viridis', aspect='auto')
            ax.set_title(f'{name}\n{feature.shape}')
            plt.colorbar(im, ax=ax)
            ax.axis('off')

        # 隐藏多余的子图
        for idx in range(n_features, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/features_{title_suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_binary_masks(self, binary_masks, title_suffix=""):
        """可视化二值掩码"""
        n_masks = len(binary_masks)
        cols = min(4, n_masks)
        rows = math.ceil(n_masks / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (name, mask) in enumerate(binary_masks.items()):
            row = idx // cols
            col = idx % cols

            ax = axes[row, col]

            # 处理二值掩码
            if mask.dim() == 3:  # [B, L, 1]
                B, L, _ = mask.shape
                H = W = int(math.sqrt(L))
                if H * W == L:
                    mask_img = mask[0].view(H, W)
                else:
                    mask_img = mask[0, :, 0].view(-1, 1)
            else:
                continue

            # 可视化二值掩码
            im = ax.imshow(mask_img.cpu().numpy(), cmap='binary', aspect='auto')
            ax.set_title(f'{name}\nBinary Ratio: {mask.float().mean():.3f}')
            ax.axis('off')

        # 隐藏多余的子图
        for idx in range(n_masks, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/binary_masks_{title_suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_ablation_results(self, ablation_results, title_suffix=""):
        """绘制消融实验结果"""
        config_names = list(ablation_results.keys())
        params = [results['params'] for results in ablation_results.values()]
        flops = [results['flops'] for results in ablation_results.values()]

        # 模拟准确率（实际应用中应该使用真实准确率）
        accuracies = [results.get('accuracy', np.random.uniform(0.7, 0.95)) for results in ablation_results.values()]

        # 创建子图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # 参数量对比
        bars1 = ax1.bar(config_names, [p / 1e6 for p in params], color='skyblue', alpha=0.7)
        ax1.set_title('Parameter Comparison')
        ax1.set_ylabel('Parameters (M)')
        ax1.set_xlabel('Configuration')
        for bar, param in zip(bars1, params):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{param / 1e6:.2f}M', ha='center', va='bottom')

        # FLOPs对比
        bars2 = ax2.bar(config_names, [f / 1e9 for f in flops], color='lightgreen', alpha=0.7)
        ax2.set_title('FLOPs Comparison')
        ax2.set_ylabel('FLOPs (G)')
        ax2.set_xlabel('Configuration')
        for bar, flop in zip(bars2, flops):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{flop / 1e9:.2f}G', ha='center', va='bottom')

        # 准确率对比
        bars3 = ax3.bar(config_names, accuracies, color='lightcoral', alpha=0.7)
        ax3.set_title('Accuracy Comparison')
        ax3.set_ylabel('Accuracy')
        ax3.set_xlabel('Configuration')
        ax3.set_ylim(0, 1)
        for bar, acc in zip(bars3, accuracies):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{acc:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/ablation_results_{title_suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        # 创建综合对比图
        fig, ax = plt.subplots(figsize=(12, 8))

        # 归一化数据
        params_norm = [p / max(params) for p in params]
        flops_norm = [f / max(flops) for f in flops]

        x = np.arange(len(config_names))
        width = 0.25

        ax.bar(x - width, params_norm, width, label='Parameters (norm)', color='skyblue')
        ax.bar(x, flops_norm, width, label='FLOPs (norm)', color='lightgreen')
        ax.bar(x + width, accuracies, width, label='Accuracy', color='lightcoral')

        ax.set_xlabel('Configuration')
        ax.set_title('Normalized Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(config_names)
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            f'{self.save_dir}/normalized_comparison_{title_suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
            dpi=300, bbox_inches='tight')
        plt.show()


def ablation_study():
    """二值RWKV消融实验"""
    print("=== 二值RWKV消融实验 ===")

    # 测试不同配置
    configs = {
        'binary_both': {'mamba_type': 'both', 'use_residual': True},
        'binary_spe': {'mamba_type': 'spe', 'use_residual': True},
        'binary_spa': {'mamba_type': 'spa', 'use_residual': True},
    }

    # 创建更小的测试输入
    input_tensor = torch.randn(2, 100, 7, 7).cuda()

    # 初始化可视化工具
    visualizer = FeatureVisualizer()

    ablation_results = {}

    for config_name, config in configs.items():
        print(f"\n测试配置: {config_name}")

        try:
            model = BinaryRWKVHSI(
                in_channels=100,  # 匹配输入通道
                hidden_dim=64,
                num_classes=16,
                mamba_type=config['mamba_type'],
                use_residual=config['use_residual']
            ).cuda()

            # 计算参数量和FLOPs
            flops, params = profile(model, (input_tensor,))

            # 获取特征用于可视化
            features = model.get_features(input_tensor)

            # 提取二值掩码
            binary_masks = {k: v for k, v in features.items() if 'binary_mask' in k}

            # 可视化特征和掩码
            visualizer.visualize_features(features, title_suffix=config_name)
            if binary_masks:
                visualizer.visualize_binary_masks(binary_masks, title_suffix=config_name)

            print(f'参数量: {params / 1000 ** 2:.2f}M')
            print(f'FLOPs: {flops / 1000 ** 3:.2f}G')

            # 测试前向传播
            with torch.no_grad():
                output = model(input_tensor)
                print(f'输出形状: {output.shape}')

            # 存储结果
            ablation_results[config_name] = {
                'params': params,
                'flops': flops,
                'accuracy': np.random.uniform(0.7, 0.95)  # 模拟准确率
            }

        except Exception as e:
            print(f"配置 {config_name} 出错: {e}")

    # 绘制消融实验结果
    visualizer.plot_ablation_results(ablation_results)


def main():
    """主函数"""

    # 创建更小的测试输入避免内存问题
    input_value = np.random.randn(2, 1, 100, 7, 7)  # 减小输入尺寸
    input_value = torch.from_numpy(input_value).float().cuda()

    print("=== 测试二值RWKV HSI模型 ===")

    # 初始化可视化工具
    visualizer = FeatureVisualizer()

    # 测试不同模式
    for mamba_type in ['both', 'spe', 'spa']:
        print(f"\n--- {mamba_type.upper()} 模式 ---")

        try:
            model = BinaryRWKVHSI(
                in_channels=100,  # 匹配输入通道
                num_classes=16,
                mamba_type=mamba_type,
                hidden_dim=64,  # 固定隐藏维度
                token_num=4  # 固定token数
            ).cuda()

            model.eval()

            with torch.no_grad():
                out = model(input_value)

            flops, params = profile(model, (input_value,))

            print(f'FLOPs = {flops / 1000 ** 3:.2f}G')
            print(f'Params = {params / 1000 ** 2:.2f}M')
            print(f'Output shape: {out.shape}')

            # 获取并可视化特征
            features = model.get_features(input_value)
            visualizer.visualize_features(features, title_suffix=f"full_{mamba_type}")

            # 可视化二值掩码
            binary_masks = {k: v for k, v in features.items() if 'binary_mask' in k}
            if binary_masks:
                visualizer.visualize_binary_masks(binary_masks, title_suffix=f"full_{mamba_type}")

        except Exception as e:
            print(f"{mamba_type} 模式出错: {e}")
            continue

    # 运行消融实验
    ablation_study()


# 调试函数 - 修复参数问题
def debug_model():
    """调试模型各层输出"""
    print("=== 模型调试 ===")

    # 创建小模型测试 - 使用正确的参数
    model = AdaptiveBinaryRWKV(
        channels=100,  # 使用正确的参数名
        token_num=4,
        use_residual=True,
        group_num=4,
        mamba_type='both'
    ).cuda()

    test_input = torch.randn(2, 100, 7, 7).cuda()

    print("输入形状:", test_input.shape)

    # 测试各组件
    with torch.no_grad():
        # 测试padding
        padded = model.padding_feature(test_input)
        print("Padding后形状:", padded.shape)

        # 测试光谱编码
        spectral = model.spectral_encoder(padded)
        print("光谱编码形状:", spectral.shape)

        # 测试空间编码
        spatial = model.spatial_encoder(padded)
        print("空间编码形状:", spatial.shape)

        # 测试特征金字塔
        fused = (spectral + spatial) / 2
        pyramid = model.feature_pyramid(fused)
        print("金字塔输出形状:", pyramid.shape)

        # 测试完整前向
        output = model(test_input)
        print("最终输出形状:", output.shape)


# 测试单个模块的函数
def test_individual_modules():
    """测试各个子模块"""
    print("=== 测试各个子模块 ===")

    # 测试光谱编码器
    spectral_encoder = SpectralBinaryEncoder(100, 64).cuda()
    test_input = torch.randn(2, 100, 7, 7).cuda()
    spectral_output = spectral_encoder(test_input)
    print(f"光谱编码器: 输入 {test_input.shape} -> 输出 {spectral_output.shape}")

    # 测试空间编码器
    spatial_encoder = SpatialBinaryEncoder(100, 64).cuda()
    spatial_output = spatial_encoder(test_input)
    print(f"空间编码器: 输入 {test_input.shape} -> 输出 {spatial_output.shape}")

    # 测试特征金字塔
    feature_pyramid = SimplifiedFeaturePyramid(64).cuda()
    test_feat = torch.randn(2, 49, 64).cuda()  # [B, L, D] = [2, 7*7, 64]
    pyramid_output = feature_pyramid(test_feat)
    print(f"特征金字塔: 输入 {test_feat.shape} -> 输出 {pyramid_output.shape}")

    # 测试二值RWKV块
    rwkv_block = BinaryRWKVBlock(64, 64, 8).cuda()
    rwkv_output, binary_mask = rwkv_block(test_feat, temperature=1.0)
    print(f"二值RWKV块: 输入 {test_feat.shape} -> 输出 {rwkv_output.shape}, 二值掩码 {binary_mask.shape}")


if __name__ == '__main__':
    # 先测试各个子模块
    test_individual_modules()

    # 再运行调试
    debug_model()

    # 最后运行主函数
    main()