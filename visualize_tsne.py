import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import seaborn as sns
import torch
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import random
import os


def visualize_tsne(
        model,
        data_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_classes=15,
        layer_name="to_latent",  # 可配置的目标层名称
        use_pca=False,
        pca_dims=64,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        figsize=(12, 10),
        save_path=None,
        dpi=300,
        n_components=2,  # 新增：支持2D或3D可视化
        view_3d=(25, 45),  # 3D视图角度(elev, azim)
        random_test=False  # 新增：随机输入测试模式
):
    """
    改进的t-SNE可视化函数，支持2D和3D可视化

    参数：
    model: 待可视化的模型
    data_loader: 数据加载器
    device: 计算设备
    num_classes: 类别数量
    layer_name: 特征提取的目标层名称
    use_pca: 是否使用PCA预降维
    pca_dims: PCA降维维度
    perplexity: t-SNE困惑度参数
    learning_rate: t-SNE学习率
    n_iter: t-SNE迭代次数
    figsize: 图像尺寸
    save_path: 结果保存路径
    dpi: 图像分辨率
    n_components: 降维维度(2或3)
    view_3d: 3D视图角度(elev, azim)
    random_test: 随机输入测试模式
    """

    if random_test:
        print("Running in random test mode with generated data...")
        return _test_with_random_data(num_classes, n_components, save_path, dpi)

    # 初始化存储
    features = []
    labels = []

    # 注册前向钩子
    activation = {}

    def hook_fn(module, input, output):
        activation["features"] = output.detach()

    try:
        # 支持点号分隔的层级访问 (如 "module.submodule.layer")
        layers = layer_name.split('.')
        target_module = model
        for layer in layers:
            target_module = getattr(target_module, layer)

        hook = target_module.register_forward_hook(hook_fn)
    except AttributeError:
        raise RuntimeError(f"Model does not have layer: {layer_name}")

    model.to(device)
    model.eval()

    # 特征提取
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Extracting features"):
            data, target = data.to(device), target.to(device)
            _ = model(data)
            feat = activation["features"]

            # 处理不同层输出格式
            if feat.dim() > 2:  # 处理空间特征
                # 使用全局平均池化处理空间维度
                feat = feat.mean(dim=[2, 3]) if feat.dim() == 4 else feat.mean(dim=2)

            features.append(feat.cpu().numpy())
            labels.append(target.cpu().numpy())

    # 移除钩子
    hook.remove()

    # 合并数据
    features = np.vstack(features)
    labels = np.concatenate(labels)

    print(f"Feature shape: {features.shape}, Label shape: {labels.shape}")

    # 预处理降维
    if use_pca and features.shape[1] > pca_dims:
        pca = PCA(n_components=pca_dims)
        features = pca.fit_transform(features)
        print(f"After PCA ({pca_dims}D): {features.shape}")

    # t-SNE降维
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        n_jobs=-1,
        random_state=42
    )
    embeddings = tsne.fit_transform(features)

    # 可视化
    return _plot_tsne(embeddings, labels, num_classes, n_components,
                      figsize, save_path, dpi, view_3d)


def _plot_tsne(embeddings, labels, num_classes, n_components,
               figsize, save_path, dpi, view_3d):
    """
    绘制t-SNE可视化图（2D或3D）
    """
    # 生成优化后的颜色映射
    if num_classes <= 10:
        colors = sns.color_palette("tab10", n_colors=num_classes)
    elif num_classes <= 20:
        colors = sns.color_palette("tab20", n_colors=num_classes)
    else:
        colors = sns.color_palette("hls", n_colors=num_classes)

    custom_cmap = ListedColormap(colors)

    # 根据维度选择绘图方式
    if n_components == 2:
        return _plot_2d(embeddings, labels, num_classes, custom_cmap,
                        figsize, save_path, dpi)
    elif n_components == 3:
        return _plot_3d(embeddings, labels, num_classes, custom_cmap,
                        figsize, save_path, dpi, view_3d)
    else:
        raise ValueError("n_components must be 2 or 3")


def _plot_2d(embeddings, labels, num_classes, cmap, figsize, save_path, dpi):
    """
    绘制2D t-SNE图
    """
    plt.figure(figsize=figsize)

    # 绘制散点图
    scatter = plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap=cmap,
        s=15,
        alpha=0.7,
        edgecolors='w',
        linewidths=0.3
    )

    # 添加颜色条
    cbar = plt.colorbar(scatter, ticks=range(num_classes))
    cbar.set_label('Class ID', rotation=270, labelpad=15)

    # 坐标轴优化
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.axis('equal')  # 保持纵横比一致

    # 自动调整坐标范围
    x_min, x_max = np.percentile(embeddings[:, 0], [0.5, 99.5])
    y_min, y_max = np.percentile(embeddings[:, 1], [0.5, 99.5])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # 添加标题
    plt.title("2D t-SNE Visualization", fontsize=14)

    # 保存结果
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 根据维度调整文件名
        save_path_2d = save_path.replace(".png", "_2d.png") if ".png" in save_path else save_path + "_2d.png"
        plt.savefig(save_path_2d, bbox_inches='tight', dpi=dpi)
        print(f"Saved 2D visualization to {save_path_2d}")

    plt.show()
    return embeddings


def _plot_3d(embeddings, labels, num_classes, cmap, figsize, save_path, dpi, view_3d):
    """
    绘制3D t-SNE图
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # 设置3D视图角度
    elev, azim = view_3d
    ax.view_init(elev=elev, azim=azim)

    # 绘制3D散点图
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        embeddings[:, 2],
        c=labels,
        cmap=cmap,
        s=15,
        alpha=0.7,
        edgecolors='w',
        linewidths=0.3,
        depthshade=True
    )

    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Class ID', rotation=270, labelpad=15)

    # 坐标轴标签
    ax.set_xlabel("t-SNE Dimension 1", labelpad=10)
    ax.set_ylabel("t-SNE Dimension 2", labelpad=10)
    ax.set_zlabel("t-SNE Dimension 3", labelpad=10)

    # 优化3D图显示
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(True, alpha=0.2, linestyle='--')

    # 添加标题
    plt.title("3D t-SNE Visualization", fontsize=14)

    # 保存结果
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 根据维度调整文件名
        save_path_3d = save_path.replace(".png", "_3d.png") if ".png" in save_path else save_path + "_3d.png"
        plt.savefig(save_path_3d, bbox_inches='tight', dpi=dpi)
        print(f"Saved 3D visualization to {save_path_3d}")

    plt.show()
    return embeddings


def _test_with_random_data(num_classes, n_components, save_path, dpi):
    """
    使用随机生成的数据进行测试
    """
    print("Generating random test data...")
    np.random.seed(42)

    # 生成模拟数据
    n_samples = 1000
    n_features = 128

    # 创建有聚类结构的数据
    features = []
    labels = []

    for i in range(num_classes):
        # 每个类别的中心点
        center = np.random.randn(n_features) * 5

        # 围绕中心点生成数据
        class_data = center + np.random.randn(n_samples // num_classes, n_features)
        features.append(class_data)
        labels.extend([i] * (n_samples // num_classes))

    features = np.vstack(features)
    labels = np.array(labels)

    print(f"Generated random features: {features.shape}, labels: {labels.shape}")

    # 可视化
    return _plot_tsne(features, labels, num_classes, n_components,
                      (10, 8), save_path, dpi, (25, 45))


# 示例用法
if __name__ == "__main__":
    # 测试模式 - 使用随机数据
    print("Running test with random data...")
    visualize_tsne(
        model=None,
        data_loader=None,
        num_classes=8,
        n_components=2,  # 测试3D可视化
        save_path="./test_tsne.png",
        random_test=True  # 启用随机测试模式
    )

    # 实际使用示例（需要真实模型和数据加载器）
    """
    # 假设已有model和data_loader
    visualize_tsne(
        model=net,
        data_loader=test_loader,
        num_classes=15,
        layer_name="features.8",  # 使用点号访问层级结构
        n_components=2,  # 2D可视化
        use_pca=True,
        save_path="./real_tsne.png",
        perplexity=40,
        learning_rate=100
    )
    """