import argparse
import datetime
import os

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
# from torchmetrics.classification import MulticlassF1Score
def random_masking_3D(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    bs, L, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]  # ids_keep: [bs x len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # x_kept: [bs x len_keep x dim]

    # removed x
    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)  # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1,
                            index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # x_masked: [bs x num_patch x dim]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)  # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch]
    return x_masked, x_kept, mask, ids_restore


class ICB(L.LightningModule):
    def __init__(self, in_features, mlp_hidden=3, drop=0.):
        super().__init__()
        hidden_features = in_features * mlp_hidden
        # self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, padding=0, groups=in_features, stride=1)
        self.conv2 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=1, groups=in_features, stride=1)
        # self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, padding=0, stride=1)
        # self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        # x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        # print(x.shape)
        # x = x.transpose(1, 2)
        return x


class PatchEmbed(L.LightningModule):
    def __init__(self, in_chans, embed_dim):
        super().__init__()
        self.input_channels = in_chans
        self.proj = nn.Sequential(
            nn.Conv3d(1, 8, (7, 3, 3), padding=0, stride=(3, 1, 1)),
            nn.ReLU(),
        )
        self.feature_size = self._get_sizes()
        self.proj2d = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

    def _get_sizes(self):
        x = torch.zeros((1, 1, self.input_channels, 15, 15))
        x = self.proj(x)
        _, c, s, w, h = x.size()
        size0 = c * s
        # print(size0)
        return size0

    def forward(self, x):
        x = x.squeeze()
        x = self.proj2d(x)
        # b, c, s, h, w = x.size()
        # x = x.view(b, c * s, h, w)
        # x_out = self.proj2d(x)
        return x


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, adaptive_filter=True, input_dim: int=15):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.dim = dim
        self.input_feature = input_dim * input_dim
        # self.input_dim = self._get_sizes()
        self.complex_weight_high = nn.Parameter(torch.randn(self.input_feature, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(self.input_feature, 2, dtype=torch.float32) * 0.02)

        self.gammb = nn.Parameter(torch.ones(2))

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1)) # * 0.5)


    # def _get_sizes(self):
    #     x = torch.zeros((1, self.dim, 13, 13))
    #     x = self.proj(x)
    #     _, c, s, w, h = x.size()
    #     size0 = c * s
    #     # print(size0)
    #     return size0

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _,= x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, H, W = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        # x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        x_fft = torch.fft.rfft2(x, dim=1, norm='ortho')
        x_fft = x_fft.reshape(x_fft.shape[0], x_fft.shape[1], x_fft.shape[2]*x_fft.shape[3] )
        # print(x_fft.shape)
        weight = torch.view_as_complex(self.complex_weight)
        # print(weight.shape)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted = self.gammb[0] * x_weighted + self.gammb[1] * x_weighted2

            # x_weighted += x_weighted2

        # Apply Inverse FFT
        # x = torch.fft.irfft2(x_weighted, n=N, dim=1, norm='ortho')
        # print(x_weighted.shape)
        x_weighted = x_weighted.reshape(x_weighted.shape[0], x_weighted.shape[1], H,  W)
        x = torch.fft.irfft2(x_weighted, s=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, H, W)  # Reshape back to original shape
        # print(x.shape)

        return x


class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class TSLANet_layer(L.LightningModule):
    def __init__(self, dim, mlp_ratio=3, drop=0., drop_path=0., norm_layer=LayerNorm, input_s: int = 15):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.asb = Adaptive_Spectral_Block(dim, input_dim=input_s)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, mlp_hidden=mlp_ratio, drop=drop)

    def forward(self, x):
        # Check if both ASB and ICB are true
        # if self.ICB and self.ASB:
        #     x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # # If only ICB is true
        # elif self.ICB:
        #     x = x + self.drop_path(self.icb(self.norm2(x)))
        # # If only ASB is true
        # elif self.ASB:
        #     x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        x = x + self.asb(self.norm1(x))
        # print('the shape of asb out is:', x.shape)
        x = x + self.drop_path(self.icb(self.norm2(x)))
        return x


class SSLANet(L.LightningModule):
    def __init__(self, in_channels: int = 1, num_classes: int = 1, input_size: int = 15, dropout_rate=0., depth=4,):
        super().__init__()
        self.emb_dim = 256
        self.dropout_rate = dropout_rate
        self.patch_embed = PatchEmbed(in_chans=in_channels, embed_dim=self.emb_dim
        )
        num_patches = 2

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, 256), requires_grad=True)
        self.pos_drop = nn.Dropout(p=dropout_rate)

        self.input_layer = nn.Linear(input_size, self.emb_dim)
        self.gamma = nn.Parameter(torch.ones(2))

        dpr = [x.item() for x in torch.linspace(0, dropout_rate, depth)]  # stochastic depth decay rule

        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=self.emb_dim, drop=dropout_rate, drop_path=dpr[i])
            for i in range(depth)]
        )

        self.tsla_blocks_1 = nn.Sequential(
            TSLANet_layer(dim=self.emb_dim, drop=dropout_rate, input_s=input_size)
        )

        self.tsla_blocks_2 = nn.Sequential(
            TSLANet_layer(dim=self.emb_dim, drop=dropout_rate, input_s=input_size),
            nn.BatchNorm2d(self.emb_dim),
            nn.GELU(),
            nn.AvgPool2d(2)
        )

        self.tsla_blocks_3 = nn.Sequential(
            TSLANet_layer(dim=self.emb_dim, drop=dropout_rate, input_s=input_size//2),
            # nn.BatchNorm2d(self.emb_dim),
            # nn.MaxPool2d(2)
        )

        self.tsla_blocks_4 = nn.Sequential(
            TSLANet_layer(dim=self.emb_dim, drop=dropout_rate, input_s=input_size//2),
            nn.BatchNorm2d(self.emb_dim),
            nn.GELU(),
            nn.AvgPool2d(2)
        )



        # Classifier head
        self.pool = nn.Sequential(
            nn.Conv2d(self.emb_dim, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # nn.Linear(256, 256)
        )

        self.residual_pool = nn.Sequential(
            nn.Conv2d(self.emb_dim, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # nn.Linear(256, 256)
        )

        self.head = nn.Linear(256, num_classes)

        # init weights
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pretrain(self, x_in):
        x = self.patch_embed(x_in)
        x = x + self.pos_embed
        x_patched = self.pos_drop(x)

        x_masked, _, self.mask, _ = random_masking_3D(x, mask_ratio=self.dropout_rate)
        self.mask = self.mask.bool()  # mask: [bs x num_patch x n_vars]

        for tsla_blk in self.tsla_blocks:
            x_masked = tsla_blk(x_masked)

        return x_masked, x_patched

    def forward(self, x):
        x = self.patch_embed(x)
        residual = x
        # print('The shape of the first input X is', x.shape)
        # x = x + self.pos_embed
        x = self.pos_drop(x)

        # for tsla_blk in self.tsla_blocks:
        #     x = tsla_blk(x)

        x = self.tsla_blocks_1(x)
        x = self.tsla_blocks_2(x)
        x = self.tsla_blocks_3(x)
        x = self.tsla_blocks_4(x)


        # x = x.mean(1)
        residual = self.residual_pool(residual)
        x = self.pool(x)
        x = self.gamma[0] * residual + self.gamma[1] * x
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


def main():
    input_value = np.random.randn(2, 1, 200, 15, 15)
    input_value = torch.from_numpy(input_value).float().cuda()
    print(input_value.dtype)
    model = SSLANet(200, 2).cuda()
    model.train()
    out = model(input_value)
    from thop import profile
    flops, params = profile(model, (input_value,))

    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    print(out.shape)


if __name__ == '__main__':
    main()