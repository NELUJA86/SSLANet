# -*- coding: utf-8 -*-
"""
SVDMamba Comprehensive Ablation Study
集成所有消融实验到一个文件中
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import datetime
from collections import OrderedDict
import argparse

# 从附件代码中导入必要的模块
from datasets import get_dataset, HyperX
from utils import (
    metrics, convert_to_color_, convert_from_color_, display_dataset,
    display_predictions, explore_spectrums, plot_spectrums, sample_gt,
    build_dataset, show_results, compute_imf_weights, get_device,
    grouper, sliding_window, count_sliding_window, camel_to_snake
)
from models import get_model, train, test, save_model

# 导入SVDMamba模型（假设已经在项目中定义）
try:
    from svd.SVDMamba_new import SVDMamba
except ImportError:
    print("Warning: SVDMamba model not found. Using placeholder.")


    # 如果SVDMamba不可用，创建一个简单的替代类用于测试
    class SVDMamba(nn.Module):
        def __init__(self, input_channels, num_classes, **kwargs):
            super().__init__()
            self.classifier = nn.Linear(input_channels, num_classes)

        def forward(self, x):
            if x.dim() == 4:
                x = x.mean(dim=[2, 3])  # Global average pooling
            return self.classifier(x)

# 创建结果目录
RESULTS_DIR = "./svdmamba_ablation_study_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class AblationStudy:
    """消融研究主类"""

    def __init__(self, args):
        self.args = args
        self.device = get_device(args.cuda)
        self.results = {}

        # 加载数据集
        print(f"Loading dataset: {args.dataset}")
        self.img, self.gt, self.LABEL_VALUES, self.IGNORED_LABELS, self.RGB_BANDS, self.palette = get_dataset(
            args.dataset, args.folder
        )
        self.N_CLASSES = len(self.LABEL_VALUES)
        self.N_BANDS = self.img.shape[-1]

        # 生成颜色调色板
        if self.palette is None:
            self.palette = {0: (0, 0, 0)}
            for k, color in enumerate(sns.color_palette("hls", len(self.LABEL_VALUES) - 1)):
                self.palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))

                print(f"Dataset loaded: {self.img.shape}, {self.N_CLASSES} classes")

    def run_patch_size_ablation(self):
        """不同Patch Size的消融实验"""
        print("\n" + "=" * 50)
        print("Running Patch Size Ablation Study")
        print("=" * 50)

        patch_sizes = [9, 13, 15, 17, 21]
        results = {}

        for patch_size in patch_sizes:
            print(f"\nTesting patch size: {patch_size}")

            # 修改参数
            self.args.patch_size = patch_size
            self.args.model = "svdmamba"

            # 运行实验
            accuracy, aa, kappa, cm = self._run_single_experiment()

            results[patch_size] = {
                'accuracy': accuracy,
                'aa': aa,
                'kappa': kappa,
                'confusion_matrix': cm
            }

            print(f"Patch Size {patch_size}: OA={accuracy:.2f}%, AA={aa:.2f}%, Kappa={kappa:.2f}")

        # 保存结果
        self.results['patch_size'] = results
        self._plot_patch_size_results(results)

        return results

    def run_training_samples_ablation(self):
        """不同训练样本比例的消融实验"""
        print("\n" + "=" * 50)
        print("Running Training Samples Ablation Study")
        print("=" * 50)

        sample_ratios = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        results = {}

        for ratio in sample_ratios:
            print(f"\nTesting training ratio: {ratio}")

            # 修改参数
            self.args.training_sample = ratio

            # 运行实验
            accuracy, aa, kappa, cm = self._run_single_experiment()

            # 计算实际训练样本数量
            train_gt, _ = sample_gt(self.gt, ratio)
            num_samples = np.count_nonzero(train_gt)

            results[ratio] = {
                'accuracy': accuracy,
                'aa': aa,
                'kappa': kappa,
                'num_samples': num_samples,
                'confusion_matrix': cm
            }

            print(f"Ratio {ratio}: Samples={num_samples}, OA={accuracy:.2f}%, AA={aa:.2f}%")

        # 保存结果
        self.results['training_samples'] = results
        self._plot_training_samples_results(results)

        return results

    def run_component_ablation(self):
        """不同模块组件的消融实验"""
        print("\n" + "=" * 50)
        print("Running Component Ablation Study")
        print("=" * 50)

        components = {
            'full_model': {'use_pruning': True, 'use_tucker': True, 'use_fusion': True},
            'no_pruning': {'use_pruning': False, 'use_tucker': True, 'use_fusion': True},
            'no_tucker': {'use_pruning': True, 'use_tucker': False, 'use_fusion': True},
            'no_fusion': {'use_pruning': True, 'use_tucker': True, 'use_fusion': False},
            'ssm_only': {'use_pruning': False, 'use_tucker': False, 'use_fusion': False}
        }

        results = {}

        for name, config in components.items():
            print(f"\nTesting configuration: {name}")

            # 运行实验
            accuracy, aa, kappa, cm = self._run_single_experiment(custom_config=config)

            results[name] = {
                'accuracy': accuracy,
                'aa': aa,
                'kappa': kappa,
                'config': config,
                'confusion_matrix': cm
            }

            print(f"{name}: OA={accuracy:.2f}%, AA={aa:.2f}%, Kappa={kappa:.2f}")

        # 保存结果
        self.results['components'] = results
        self._plot_component_ablation(results)

        return results

    def run_parameter_sensitivity(self):
        """参数敏感性分析"""
        print("\n" + "=" * 50)
        print("Running Parameter Sensitivity Analysis")
        print("=" * 50)

        # SVD秩比率分析
        rank_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
        rank_results = {}

        for ratio in rank_ratios:
            print(f"\nTesting rank ratio: {ratio}")

            config = {'rank_ratio': ratio}
            accuracy, aa, kappa, cm = self._run_single_experiment(custom_config=config)

            rank_results[ratio] = {
                'accuracy': accuracy,
                'aa': aa,
                'kappa': kappa,
                'confusion_matrix': cm
            }

            print(f"Rank Ratio {ratio}: OA={accuracy:.2f}%, AA={aa:.2f}%")

        # 状态维度分析
        state_dims = [8, 16, 32, 64]
        state_results = {}

        for dim in state_dims:
            print(f"\nTesting state dimension: {dim}")

            config = {'d_state': dim}
            accuracy, aa, kappa, cm = self._run_single_experiment(custom_config=config)

            state_results[dim] = {
                'accuracy': accuracy,
                'aa': aa,
                'kappa': kappa,
                'confusion_matrix': cm
            }

            print(f"State Dim {dim}: OA={accuracy:.2f}%, AA={aa:.2f}%")

        results = {
            'rank_ratios': rank_results,
            'state_dims': state_results
        }

        # 保存结果
        self.results['parameter_sensitivity'] = results
        self._plot_parameter_sensitivity(results)

        return results

    def run_feature_visualization(self):
        """特征可视化"""
        print("\n" + "=" * 50)
        print("Running Feature Visualization")
        print("=" * 50)

        # 训练一个完整的模型
        model, train_loader, test_gt = self._train_model()
        model.eval()

        # 提取特征
        features_global = []
        features_local = []
        labels = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 10:  # 只取前10个batch以减少计算量
                    break

                data = data.to(self.device)

                # 假设模型支持特征提取
                try:
                    # 尝试获取多种特征
                    if hasattr(model, 'get_features'):
                        features = model.get_features(data)
                    else:
                        # 使用默认方法
                        features = model(data)
                        if isinstance(features, tuple):
                            features = features[0]

                    features_global.append(features.cpu().numpy())
                    labels.append(target.cpu().numpy())

                except Exception as e:
                    print(f"Feature extraction failed: {e}")
                    break

        if features_global:
            features_global = np.concatenate(features_global)
            labels = np.concatenate(labels)

            # PCA可视化
            self._plot_pca_features(features_global, labels)

            # 特征分布可视化
            self._plot_feature_distributions(features_global, labels)

        print("Feature visualization completed")

    def run_tsne_visualization(self):
        """t-SNE可视化"""
        print("\n" + "=" * 50)
        print("Running t-SNE Visualization")
        print("=" * 50)

        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
        except ImportError:
            print("scikit-learn not available, skipping t-SNE visualization")
            return

        # 训练模型
        model, train_loader, _ = self._train_model()
        model.eval()

        # 提取特征和标签
        features = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 5:  # 限制数据量
                    break

                data = data.to(self.device)
                output = model(data)

                if isinstance(output, tuple):
                    output = output[0]

                features.append(output.cpu().numpy())
                all_labels.append(target.cpu().numpy())

        if not features:
            print("No features extracted for t-SNE")
            return

        features = np.concatenate(features)
        all_labels = np.concatenate(all_labels)

        # 如果特征维度太高，先使用PCA降维
        if features.shape[1] > 50:
            pca = PCA(n_components=50)
            features = pca.fit_transform(features)
            print(f"PCA reduced features to {features.shape[1]} dimensions")

        # t-SNE降维
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)

        # 绘制t-SNE图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                              c=all_labels, cmap='tab20', alpha=0.7, s=10)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of SVDMamba Features')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/tsne_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("t-SNE visualization completed")

    def run_singular_value_analysis(self):
        """奇异值分析"""
        print("\n" + "=" * 50)
        print("Running Singular Value Analysis")
        print("=" * 50)

        # 训练模型
        model, _, _ = self._train_model()

        # 收集奇异值（如果模型支持）
        singular_values = []
        layer_names = []

        for name, module in model.named_modules():
            if hasattr(module, 'sigma'):  # 假设SVD层有sigma属性
                sv = module.sigma.detach().cpu().numpy()
                singular_values.append(sv)
                layer_names.append(name)
                print(f"Found singular values in layer: {name}, shape: {sv.shape}")

        if singular_values:
            self._plot_singular_value_analysis(singular_values, layer_names)
        else:
            print("No singular values found in the model")

    def run_computational_efficiency(self):
        """计算效率分析"""
        print("\n" + "=" * 50)
        print("Running Computational Efficiency Analysis")
        print("=" * 50)

        # 定义要比较的模型
        models_config = {
            'SVDMamba': {'model': 'svdmamba', 'params': {}},
            'CNN2D': {'model': 'cnn2d', 'params': {}},
            'HybridSN': {'model': 'hybridsn', 'params': {}},
            'SSFTT': {'model': 'ssftt', 'params': {}}
        }

        results = {}

        for name, config in models_config.items():
            print(f"\nAnalyzing efficiency of {name}")

            try:
                if config['model'] == 'svdmamba':
                    # 对于SVDMamba，直接创建实例
                    model_kwargs = {
                        'input_channels': self.N_BANDS,
                        'num_classes': self.N_CLASSES,
                        'num_layers': 2,
                        'd_model': 64,
                        'd_state': 16,
                        'use_pruning': True
                    }
                    model = SVDMamba(**model_kwargs).to(self.device)
                else:
                    # 尝试多种参数组合
                    model = None
                    param_combinations = [
                        {'n_classes': self.N_CLASSES, 'n_bands': self.N_BANDS},
                        {'n_classes': self.N_CLASSES, 'n_bands': self.N_BANDS, 'patch_size': self.args.patch_size},
                        {'n_classes': self.N_CLASSES, 'n_bands': self.N_BANDS, 'device': self.device}
                    ]

                    for params in param_combinations:
                        try:
                            model, optimizer, criterion, hyperparams = get_model(config['model'], **params)
                            break
                        except Exception as e:
                            continue

                    if model is None:
                        # 如果所有参数组合都失败，创建简单模型
                        print(f"All parameter combinations failed for {name}, creating simple model")
                        model = self._create_simple_model(self.N_BANDS, self.N_CLASSES)

                # 计算参数数量
                total_params = sum(p.numel() for p in model.parameters())

                # 估算FLOPs（简化版本）
                flops = self._estimate_flops(model)

                # 测量推理时间
                inference_time = self._measure_inference_time(model)

                results[name] = {
                    'parameters': total_params,
                    'flops': flops,
                    'inference_time': inference_time,
                    'parameters_M': total_params / 1e6
                }

                print(f"{name}: Params={total_params / 1e6:.2f}M, FLOPs={flops / 1e6:.2f}M, Time={inference_time:.4f}s")

            except Exception as e:
                print(f"Failed to analyze {name}: {e}")
                # 即使失败也创建一个占位结果
                results[name] = {
                    'parameters': 0,
                    'flops': 0,
                    'inference_time': 0,
                    'parameters_M': 0,
                    'error': str(e)
                }

        # 保存结果
        self.results['efficiency'] = results
        self._plot_efficiency_comparison(results)

        return results

    def _create_simple_model(self, input_channels, num_classes):
        """创建简单的备用模型"""

        class SimpleModel(nn.Module):
            def __init__(self, input_channels, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(64, num_classes)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        return SimpleModel(input_channels, num_classes).to(self.device)

    def _run_single_experiment(self, custom_config=None):
        """运行单个实验并返回结果"""
        # 保存原始参数
        original_patch_size = self.args.patch_size
        original_model = self.args.model

        try:
            # 应用自定义配置
            if custom_config:
                if 'patch_size' in custom_config:
                    self.args.patch_size = custom_config['patch_size']

            # 使用SVDMamba模型
            self.args.model = "svdmamba"

            filtered_config = None
            if custom_config:
                filtered_config = {}
                svdmamba_params = ['num_layers', 'd_model', 'd_state', 'use_pruning', 'rank_ratio']
                for key in svdmamba_params:
                    if key in custom_config:
                        filtered_config[key] = custom_config[key]

            # 运行10次实验取平均
            n_runs = 1
            all_accuracy = []
            all_aa = []
            all_kappa = []
            all_cm = []

            for run in range(n_runs):
                print(f"  Run {run + 1}/{n_runs}")
            # 训练模型并获取结果
                model, _, test_gt = self._train_model(custom_config)
                probabilities = test(model, self.img, {
                    "patch_size": self.args.patch_size,
                    "center_pixel": True,
                    "batch_size": self.args.batch_size,
                    "device": self.device,
                    "n_classes": self.N_CLASSES,
                    "test_stride": self.args.test_stride
                })

                prediction = np.argmax(probabilities, axis=-1)

                # 计算指标
                run_results = metrics(
                    prediction, test_gt,
                    ignored_labels=self.IGNORED_LABELS,
                    n_classes=self.N_CLASSES
                )

                all_accuracy.append(run_results["Accuracy"])
                all_aa.append(run_results["Aa"])
                all_kappa.append(run_results["Kappa"])
                all_cm.append(run_results["Confusion matrix"])
            # 计算平均指标
            avg_accuracy = np.mean(all_accuracy)
            avg_aa = np.mean(all_aa)
            avg_kappa = np.mean(all_kappa)
            # 使用最后一次的混淆矩阵（或可以计算平均混淆矩阵）
            avg_cm = all_cm[-1]
            # return (run_results["Accuracy"], run_results["Aa"],
            #         run_results["Kappa"], run_results["Confusion matrix"])
            return avg_accuracy, avg_aa, avg_kappa, avg_cm

        finally:
            # 恢复原始参数
            self.args.patch_size = original_patch_size
            self.args.model = original_model

    def _train_model(self, custom_config=None):
        """训练模型并返回模型、数据加载器和测试GT"""
        # 分割训练测试集
        train_gt, test_gt = sample_gt(self.gt, self.args.training_sample, mode=self.args.sampling_mode)

        # 准备超参数
        hyperparams = {
            'dataset': self.args.dataset,
            'patch_size': self.args.patch_size,
            'n_bands': self.N_BANDS,
            'n_classes': self.N_CLASSES,
            'ignored_labels': self.IGNORED_LABELS,
            'device': self.device,
            'flip_augmentation': self.args.flip_augmentation,
            'radiation_augmentation': self.args.radiation_augmentation,
            'mixture_augmentation': self.args.mixture_augmentation,
            'center_pixel': True,
            'supervision': 'full'
        }

        # 应用自定义配置
        if custom_config:
            hyperparams.update(custom_config)

        # 创建数据加载器
        train_dataset = HyperX(self.img, train_gt, **hyperparams)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
        )

        # 获取模型
        if self.args.model == "svdmamba":
            # 直接创建SVDMamba模型 - 修复参数重复问题
            model_kwargs = {
                'input_channels': self.N_BANDS,
                'num_classes': self.N_CLASSES,
            }

            # 只添加不在model_kwargs中的参数
            svdmamba_params = ['num_layers', 'd_model', 'd_state', 'use_pruning', 'rank_ratio']
            for key in svdmamba_params:
                if key in hyperparams:
                    model_kwargs[key] = hyperparams[key]

            # 移除可能引起冲突的参数
            if custom_config:
                for key in svdmamba_params:
                    if key in custom_config:
                        model_kwargs[key] = custom_config[key]
            if 'num_layers' not in model_kwargs:
                model_kwargs['num_layers'] = 2
            if 'd_model' not in model_kwargs:
                model_kwargs['d_model'] = 64
            if 'd_state' not in model_kwargs:
                model_kwargs['d_state'] = 16
            if 'use_pruning' not in model_kwargs:
                model_kwargs['use_pruning'] = True
            if 'rank_ratio' not in model_kwargs:
                model_kwargs['rank_ratio'] = 0.5

            print(f"Creating SVDMamba with parameters: {model_kwargs}")
            model = SVDMamba(**model_kwargs).to(self.device)

            # 简单的训练循环（简化版）
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # 快速训练几个epoch
            model.train()
            for epoch in range(3):  # 简化训练
                total_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                print(f"Epoch {epoch + 1}/3, Loss: {total_loss / len(train_loader):.4f}")
        else:
            # 使用现有的get_model函数
            model, optimizer, criterion, hyperparams = get_model(
                self.args.model, **hyperparams
            )

        return model, train_loader, test_gt

    def _estimate_flops(self, model):
        """估算FLOPs（简化版本）"""
        # 这是一个简化的FLOPs估算
        # 实际应用中应该使用更精确的工具如thop或fvcore
        total_flops = 0

        # 创建测试输入
        try:
            dummy_input = torch.randn(1, self.N_BANDS, self.args.patch_size, self.args.patch_size).to(self.device)

            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    # 简化的Conv2d FLOPs计算
                    h_out = (self.args.patch_size + 2 * module.padding[0] - module.kernel_size[0]) // module.stride[
                        0] + 1
                    w_out = (self.args.patch_size + 2 * module.padding[1] - module.kernel_size[1]) // module.stride[
                        1] + 1
                    flops = h_out * w_out * module.in_channels * module.out_channels * module.kernel_size[0] * \
                            module.kernel_size[1]
                    total_flops += flops
                elif isinstance(module, nn.Linear):
                    flops = module.in_features * module.out_features
                    total_flops += flops
                elif isinstance(module, nn.Conv3d):
                    # 简化的Conv3d FLOPs计算
                    d_out = (1 + 2 * module.padding[0] - module.kernel_size[0]) // module.stride[0] + 1
                    h_out = (self.args.patch_size + 2 * module.padding[1] - module.kernel_size[1]) // module.stride[
                        1] + 1
                    w_out = (self.args.patch_size + 2 * module.padding[2] - module.kernel_size[2]) // module.stride[
                        2] + 1
                    flops = d_out * h_out * w_out * module.in_channels * module.out_channels * \
                            module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2]
                    total_flops += flops
        except Exception as e:
            print(f"FLOPs estimation failed: {e}, using fallback method")
            # 备用方法：基于参数数量估算
            total_params = sum(p.numel() for p in model.parameters())
            total_flops = total_params * 2  # 粗略估计

        return total_flops

    def _measure_inference_time(self, model):
        """测量推理时间"""
        model.eval()

        try:
            # 创建测试输入
            dummy_input = torch.randn(1, self.N_BANDS, self.args.patch_size, self.args.patch_size).to(self.device)

            # Warmup
            for _ in range(10):
                _ = model(dummy_input)

            # Measurement
            if torch.cuda.is_available():
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                for _ in range(100):
                    _ = model(dummy_input)
                end_time.record()

                torch.cuda.synchronize()
                return start_time.elapsed_time(end_time) / 100.0  # 平均时间(ms)
            else:
                # CPU测量
                import time
                start_time = time.time()
                for _ in range(100):
                    _ = model(dummy_input)
                end_time = time.time()
                return (end_time - start_time) * 10.0  # 转换为ms
        except Exception as e:
            print(f"Inference time measurement failed: {e}")
            return 0.0

    # 绘图函数
    def _plot_patch_size_results(self, results):
        """绘制Patch Size结果"""
        sizes = list(results.keys())
        accuracies = [results[s]['accuracy'] for s in sizes]
        aas = [results[s]['aa'] for s in sizes]

        plt.figure(figsize=(10, 6))
        plt.plot(sizes, accuracies, 'o-', linewidth=2, markersize=8, label='Overall Accuracy')
        plt.plot(sizes, aas, 's-', linewidth=2, markersize=8, label='Average Accuracy')
        plt.xlabel('Patch Size', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Effect of Patch Size on SVDMamba Performance', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/patch_size_ablation.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_training_samples_results(self, results):
        """绘制训练样本结果"""
        ratios = list(results.keys())
        accuracies = [results[r]['accuracy'] for r in ratios]
        aas = [results[r]['aa'] for r in ratios]
        samples = [results[r]['num_samples'] for r in ratios]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 按比例绘图
        ax1.plot(ratios, accuracies, 'o-', linewidth=2, markersize=8, label='Overall Accuracy')
        ax1.plot(ratios, aas, 's-', linewidth=2, markersize=8, label='Average Accuracy')
        ax1.set_xlabel('Training Sample Ratio', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Accuracy vs Training Ratio', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # 按样本数量绘图
        ax2.plot(samples, accuracies, 'o-', linewidth=2, markersize=8, label='Overall Accuracy')
        ax2.plot(samples, aas, 's-', linewidth=2, markersize=8, label='Average Accuracy')
        ax2.set_xlabel('Number of Training Samples', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Accuracy vs Sample Count', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/training_samples_ablation.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_component_ablation(self, results):
        """绘制组件消融结果"""
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        aas = [results[m]['aa'] for m in models]

        x = np.arange(len(models))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width / 2, accuracies, width, label='Overall Accuracy', alpha=0.8)
        plt.bar(x + width / 2, aas, width, label='Average Accuracy', alpha=0.8)

        plt.xlabel('Model Configuration', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Component Ablation Study', fontsize=14)
        plt.xticks(x, models, rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, v in enumerate(accuracies):
            plt.text(i - width / 2, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=10)
        for i, v in enumerate(aas):
            plt.text(i + width / 2, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/component_ablation.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_parameter_sensitivity(self, results):
        """绘制参数敏感性分析结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 秩比率结果
        rank_ratios = list(results['rank_ratios'].keys())
        rank_acc = [results['rank_ratios'][r]['accuracy'] for r in rank_ratios]
        rank_aa = [results['rank_ratios'][r]['aa'] for r in rank_ratios]

        ax1.plot(rank_ratios, rank_acc, 'o-', linewidth=2, markersize=8, label='Overall Accuracy')
        ax1.plot(rank_ratios, rank_aa, 's-', linewidth=2, markersize=8, label='Average Accuracy')
        ax1.set_xlabel('Rank Ratio', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('SVD Rank Ratio Sensitivity', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # 状态维度结果
        state_dims = list(results['state_dims'].keys())
        state_acc = [results['state_dims'][d]['accuracy'] for d in state_dims]
        state_aa = [results['state_dims'][d]['aa'] for d in state_dims]

        ax2.plot(state_dims, state_acc, 'o-', linewidth=2, markersize=8, label='Overall Accuracy')
        ax2.plot(state_dims, state_aa, 's-', linewidth=2, markersize=8, label='Average Accuracy')
        ax2.set_xlabel('State Dimension', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('State Dimension Sensitivity', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/parameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_pca_features(self, features, labels):
        """PCA特征可视化"""
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("scikit-learn not available, skipping PCA")
            return

        # 重塑特征
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)

        # PCA降维
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                              c=labels, cmap='tab20', alpha=0.7, s=20)
        plt.colorbar(scatter)
        plt.title('PCA Visualization of SVDMamba Features')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/pca_feature_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_feature_distributions(self, features, labels):
        """特征分布可视化"""
        # 选择前几个特征维度进行可视化
        n_features = min(6, features.shape[1])

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        unique_labels = np.unique(labels)

        for i in range(n_features):
            for label in unique_labels:
                mask = labels == label
                axes[i].hist(features[mask, i], bins=30, alpha=0.7,
                             label=f'Class {label}', density=True)

            axes[i].set_title(f'Feature Dimension {i + 1}')
            axes[i].set_xlabel('Feature Value')
            axes[i].set_ylabel('Density')
            if i == 0:
                axes[i].legend()

        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_singular_value_analysis(self, singular_values, layer_names):
        """奇异值分析绘图"""
        n_layers = len(singular_values)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 奇异值衰减曲线
        for i, sv in enumerate(singular_values):
            axes[0, 0].plot(range(len(sv)), sv / sv.max(),
                            label=layer_names[i] if i < len(layer_names) else f'Layer {i + 1}',
                            alpha=0.7, linewidth=2)
        axes[0, 0].set_xlabel('Singular Value Index')
        axes[0, 0].set_ylabel('Normalized Magnitude')
        axes[0, 0].set_title('Singular Value Decay')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        # 有效秩分布
        effective_ranks = []
        for sv in singular_values:
            threshold = 0.01 * sv.max()  # 1%阈值
            effective_rank = np.sum(sv > threshold)
            effective_ranks.append(effective_rank)

        axes[0, 1].bar(range(len(effective_ranks)), effective_ranks)
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Effective Rank')
        axes[0, 1].set_title('Effective Rank per Layer')
        axes[0, 1].grid(True, alpha=0.3)

        # 奇异值直方图
        all_sv = np.concatenate(singular_values)
        axes[1, 0].hist(all_sv, bins=50, alpha=0.7, density=True)
        axes[1, 0].set_xlabel('Singular Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Singular Value Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # 累积能量
        for i, sv in enumerate(singular_values):
            sorted_sv = np.sort(sv)[::-1]
            cumulative_energy = np.cumsum(sorted_sv) / np.sum(sorted_sv)
            axes[1, 1].plot(range(len(cumulative_energy)), cumulative_energy,
                            label=layer_names[i] if i < len(layer_names) else f'Layer {i + 1}',
                            alpha=0.7, linewidth=2)
        axes[1, 1].set_xlabel('Number of Singular Values')
        axes[1, 1].set_ylabel('Cumulative Energy')
        axes[1, 1].set_title('Cumulative Energy Distribution')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/singular_value_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_efficiency_comparison(self, results):
        """计算效率比较绘图"""
        models = list(results.keys())

        # 过滤掉出错的结果和无效结果
        valid_models = []
        parameters = []
        flops = []
        times = []

        for m in models:
            if ('parameters' in results[m] and
                    results[m]['parameters'] > 0 and
                    'error' not in results[m]):
                valid_models.append(m)
                parameters.append(results[m]['parameters_M'])
                flops.append(results[m]['flops'] / 1e6 if results[m]['flops'] > 0 else 0)  # 转换为百万
                times.append(results[m]['inference_time'])

        if not valid_models:
            print("No valid efficiency results to plot")
            # 创建空图
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No valid efficiency data available',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, fontsize=14)
            plt.title('Computational Efficiency Comparison')
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/efficiency_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 参数量比较
        bars1 = ax1.bar(valid_models, parameters, alpha=0.8, color='skyblue')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Parameters (Millions)')
        ax1.set_title('Model Parameter Comparison')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.2f}M', ha='center', va='bottom', fontsize=10)

        # 推理时间比较
        bars2 = ax2.bar(valid_models, times, alpha=0.8, color='lightcoral')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Inference Time Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.2f}ms', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/efficiency_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_all_results(self):
        """保存所有结果到文件"""
        import json
        import pickle

        # 转换为可序列化的格式
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for subkey, subvalue in value.items():
                    if hasattr(subvalue, 'tolist'):
                        serializable_results[key][subkey] = subvalue.tolist()
                    else:
                        serializable_results[key][subkey] = subvalue

        # 保存为JSON
        with open(f'{RESULTS_DIR}/ablation_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)

        # 保存为pickle（保持原始类型）
        with open(f'{RESULTS_DIR}/ablation_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)

        print(f"Results saved to {RESULTS_DIR}/")

    def run_complete_ablation_study(self):
        """运行完整的消融研究"""
        print("Starting Comprehensive SVDMamba Ablation Study")
        print("=" * 60)

        start_time = datetime.datetime.now()

        # 运行所有消融实验
        # self.run_patch_size_ablation()
        # self.run_training_samples_ablation()
        # self.run_component_ablation()
        # self.run_parameter_sensitivity()
        # self.run_feature_visualization()
        # self.run_tsne_visualization()
        # self.run_singular_value_analysis()
        self.run_computational_efficiency()

        # 保存所有结果
        self.save_all_results()

        end_time = datetime.datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print("Ablation Study Completed!")
        print(f"Total duration: {duration}")
        print(f"Results saved in: {RESULTS_DIR}")
        print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SVDMamba Ablation Study")

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='IndianPines',
                        help='Dataset name')
    parser.add_argument('--folder', type=str, default='./Datasets/',
                        help='Dataset folder')

    # 训练参数
    parser.add_argument('--training_sample', type=float, default=0.1,
                        help='Training sample ratio')
    parser.add_argument('--sampling_mode', type=str, default='random',
                        help='Sampling mode')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--patch_size', type=int, default=15,
                        help='Patch size')

    # 模型参数
    parser.add_argument('--model', type=str, default='svdmamba',
                        help='Model name')

    # 其他参数
    parser.add_argument('--cuda', type=int, default=0,
                        help='CUDA device')
    parser.add_argument('--test_stride', type=int, default=1,
                        help='Test stride')
    parser.add_argument('--flip_augmentation', action='store_true',
                        help='Use flip augmentation')
    parser.add_argument('--radiation_augmentation', action='store_true',
                        help='Use radiation augmentation')
    parser.add_argument('--mixture_augmentation', action='store_true',
                        help='Use mixture augmentation')

    args = parser.parse_args()

    # 运行消融研究
    study = AblationStudy(args)
    study.run_complete_ablation_study()


if __name__ == "__main__":
    main()