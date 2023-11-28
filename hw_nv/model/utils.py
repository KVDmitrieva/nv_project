import torch
from torch import nn
import torch.nn.functional as F


__all__ = ["UpsamplerBlock", "ScaleDiscriminator", "PeriodDiscriminator"]

class UpsamplerBlock(nn.Module):
    def __init__(self, upsampler_params, res_block_kernels=(3, 7, 11), res_block_dilation=((1, 1), (3, 1), (5, 1))):
        super().__init__()
        self.upsampler = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose1d(**upsampler_params)
        )
        n = len(res_block_kernels)
        res_blocks = []
        for i in range(n):
            res_blocks.append(ResStack(upsampler_params["out_channels"], res_block_kernels[i], res_block_dilation))

        self.mfr = nn.ModuleList(res_blocks)

    def forward(self, x):
        x = self.upsampler(x)
        mfr_out = torch.zeros_like(x, device=x.device)
        for res_block in self.mfr:
            mfr_out = mfr_out + res_block(x)
        return mfr_out

class ResStack(nn.Module):
    def __init__(self, channels_num, kernel_size, block_dilation: list):
        super().__init__()
        n = len(block_dilation)
        net = []
        for i in range(n):
            net.append(ResBlock(channels_num=channels_num, kernel_size=kernel_size, dilation=block_dilation[i]))

        self.net = nn.Sequential(*net)
    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, channels_num, kernel_size, dilation: list):
        super().__init__()
        net = []
        n = len(dilation)
        for i in range(n):
            padding = dilation * (kernel_size - 1) // 2
            net.append(nn.LeakyReLU())
            net.append(nn.Conv1d(in_channels=channels_num, out_channels=channels_num,
                                 kernel_size=kernel_size, dilation=dilation[i], padding=padding))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return x + self.net(x)

class ScaleDiscriminator(nn.Module):
    def __init__(self, pooling, prolog_params, downsampler_params, post_downsampler_params, epilog_params):
        super().__init__()
        self.pooling = nn.AvgPool1d(pooling)
        self.prolog = nn.Conv1d(**prolog_params)
        self.downsampler = nn.ModuleList([nn.Conv1d(**params) for params in downsampler_params])
        self.post_downsampler = nn.Conv1d(**post_downsampler_params)
        self.epilog =  nn.Conv1d(**epilog_params)

    def forward(self, x):
        x = self.pooling(x)

        feature_maps = []
        x = F.leaky_relu(self.prolog(x))
        feature_maps.append(x)

        for downsampler in self.downsampler:
            x = F.leaky_relu(downsampler(x))
            feature_maps.append(x)

        x = F.leaky_relu(self.post_downsampler(x))
        feature_maps.append(x)

        x = self.epilog(x)
        feature_maps.append(x)
        return x, feature_maps

class PeriodDiscriminator(nn.Module):
    def __init__(self, period, stem_params, poststem_params, epilog_params):
        super().__init__()
        self.period = period
        self.stem = nn.ModuleList([nn.Conv2d(**stem_param) for stem_param in stem_params])
        self.post_stem = nn.Conv2d(**poststem_params)
        self.epilog = nn.Conv2d(**epilog_params)

    def forward(self, x):
        batch_size, len_t = x.shape
        x = F.pad(x, (0, len_t % self.period))
        x = x.reshape(batch_size, len_t // self.period, self.period)

        feature_maps = []
        for stem in self.stem:
            x = F.leaky_relu(stem(x))
            feature_maps.append(x)

        x = F.leaky_relu(self.post_stem(x))
        feature_maps.append(x)

        x = self.epilog(x)
        feature_maps.append(x)

        return x, feature_maps
