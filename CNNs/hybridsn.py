# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
import cv2
import matplotlib.pyplot as plt
# utils
import math
import os
import datetime
import numpy as np
import joblib
import collections
import torchvision
from pylab import *

from PIL import Image
import torch.autograd as autograd

class ConvEtAl(torch.nn.Module):
    def __init__(self, input_channels, flatten=True):
        super(ConvEtAl, self).__init__()
        self.feature_size = 64
        self.name = "conv4"

        self.layer1 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(64)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer2 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(128)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer3 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(256)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer4 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(512)),
          ('relu',    nn.ReLU()),
          #('avgpool', nn.AvgPool2d(kernel_size=4))
          ('glbpool', nn.AdaptiveAvgPool2d(1))
        ]))

        self.is_flatten = flatten
        self.flatten = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
      #  h = self.layer4(h)
        #print(h.size())
        if(self.is_flatten): h = self.flatten(h)
        return h

class TESTEtAl(nn.Module):
    """
    A semi-supervised convolutional neural network for hyperspectral image classification
    Bing Liu, Xuchu Yu, Pengqiang Zhang, Xiong Tan, Anzhu Yu, Zhixiang Xue
    Remote Sensing Letters, 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=15, pool_size=None):
        super(TESTEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1

        # "W1 is a 3x3xB1 kernel [...] B1 is the number of the output bands for the convolutional
        # "and pooling layer" -> actually 3x3 2D convolutions with B1 outputs
        # "the value of B1 is set to be 80"
        # resnet = resnet50(pretrained=True)
        # resnet_feature = list(resnet.children())[:-2]
        self.feature = ConvEtAl(input_channels, flatten=True)
        self.features_sizes = self._get_sizes()
        self.classifier = nn.Linear(self.features_sizes, n_classes)

    def _get_sizes(self):
        x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size))
        x = self.feature(x)
        w, h = x.size()
        size0 = w * h
        return size0

    def forward(self, x):
        x = x.squeeze()
        x = self.feature(x)
        x = self.classifier(x)
        return x

layers_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

features_grad = 0

def extract(g):
    global features_grad
    features_grad = g

#
# def draw_CAM(model, img, save_path, transform=None, visual_heatmap=False, out_layer=None):
#     """
#     绘制 Class Activation Map
#     :param model: 加载好权重的Pytorch model
#     :param img_path: 测试图片路径
#     :param save_path: CAM结果保存路径
#     :param transform: 输入图像预处理方法
#     :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
#     :return:
#     """
#     # 读取图像并预处理
#     global layer2
#
#     # model转为eval模式
#     model.eval()
#
#     # 获取模型层的字典
#     layers_dict = {layers_names[i]: None for i in range(len(layers_names))}
#     for i, (name, module) in enumerate(model.features._modules.items()):
#         layers_dict[layers_names[i]] = module
#
#     # 遍历模型的每一层, 获得指定层的输出特征图
#     # features: 指定层输出的特征图, features_flatten: 为继续完成前端传播而设置的变量
#     features = img
#     start_flatten = False
#     features_flatten = None
#     for name, layer in layers_dict.items():
#         if name != out_layer and start_flatten is False:  # 指定层之前
#             features = layer(features)
#         elif name == out_layer and start_flatten is False:  # 指定层
#             features = layer(features)
#             start_flatten = True
#         else:  # 指定层之后
#             if features_flatten is None:
#                 features_flatten = layer(features)
#             else:
#                 features_flatten = layer(features_flatten)
#
#     features_flatten = torch.flatten(features_flatten, 1)
#     output = model.classifier(features_flatten)
#
#     # 预测得分最高的那一类对应的输出score
#     pred = torch.argmax(output, 1).item()
#     pred_class = output[:, pred]
#
#     # 求中间变量features的梯度
#     # 方法1
#     # features.register_hook(extract)
#     # pred_class.backward()
#     # 方法2
#     features_grad = autograd.grad(pred_class, features, allow_unused=True)[0]
#
#     grads = features_grad  # 获取梯度
#     pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
#     # 此处batch size默认为1，所以去掉了第0维（batch size维）
#     pooled_grads = pooled_grads[0]
#     features = features[0]
#     print("pooled_grads:", pooled_grads.shape)
#     print("features:", features.shape)
#     # features.shape[0]是指定层feature的通道数
#     for i in range(features.shape[0]):
#         features[i, ...] *= pooled_grads[i, ...]
#
#     # 计算heatmap
#     heatmap = features.detach().cpu().numpy()
#     heatmap = np.mean(heatmap, axis=0)
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)
#
#     # 可视化原始热力图
#     if visual_heatmap:
#         plt.matshow(heatmap)
#         plt.show()
#
#     img = cv2.imread(img_path)  # 用cv2加载原始图像
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
#     heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
#     superimposed_img = heatmap * 0.7 + img  # 这里的0.4是热力图强度因子
#     cv2.imwrite(save_path, superimposed_img)

class HybridEtAl(nn.Module):
    """
    HybridSN: Exploring 3D-2D CNN Feature Hierarchy for Hyperspectral Image Classification
    Roy, Swalpa Kumar and Krishna, Gopal and Dubey, Shiv Ram and Chaudhuri, Bidyut B
    IEEE Geoscience and Remote Sensing Letters 2020
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=15):
        super(HybridEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # the 8-neighbor pixels [...] are fed into the Conv1 convolved by n1 kernels
        # and s1 stride. Conv1 results are feature vectors each with height of and
        # the width is 1. After reshape layer, the feature vectors becomes an image-like
        # 2-dimension data.
        # Conv2 has 64 kernels size of 3x3, with stride s2.
        # After that, the 64 results are drawn into a vector as the input of the fully
        # connected layer FC1 which has n4 nodes.
        # In the four datasets, the kernel height nk1 is 24 and stride s1, s2 is 9 and 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, (7, 3, 3), padding=0, stride=(1, 1, 1)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, (5, 3, 3), padding=0, stride=(1, 1, 1)),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, (3, 3, 3), padding=0, stride=(1, 1, 1)),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(6016, 64, (3, 3), padding=0, stride=(1, 1)),
            nn.ReLU()
        ) ## 6016 for IN, 576 for Grss ##2912 for PU

        # self.features_size = self._get_final_flattened_size()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(3136, 256)
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.4)
        self.classfier = nn.Linear(128, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        b, c, s, h, w = x.size()
        x = x.view(b, c*s, h, w)
        # print(x.shape)
        x = self.conv4(x)
        x = self.flatten(x)
        # print(x.shape)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.classfier(x)

        return x


if __name__ == '__main__':
    img_size = 15
    x = torch.rand(10, 1, 200, img_size, img_size)
    model = HybridEtAl(200, 16, patch_size=15)

    optimal_batch_size = 2
    from thop import profile

    flops, params = profile(model, (x,))
    with torch.autograd.profiler.profile(enabled=True) as prof:
        out = model(x)

    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    repetitions = 100
    total_time = 0
    optimal_batch_size = 2
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * optimal_batch_size) / total_time
    print("FinalThroughput:", Throughput)
    import numpy as np

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(x)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(x)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
                                                                                         std_syn=std_syn,
                                                                                         mean_fps=mean_fps))
    print(mean_syn)