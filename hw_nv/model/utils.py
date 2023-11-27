import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


__all__ = ["UpsamplerBlock", "ScaleDiscriminator", "PeriodDiscriminator"]

class UpsamplerBlock(nn.Module):
    def __init__(self, upsampler_params, res_block_kernels: list, res_block_dilation: list):
        super().__init__()
        self.upsampler = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose1d(**upsampler_params)
        )
        n = len(res_block_kernels)
        res_blocks = []
        for i in range(n):
            res_blocks.append(ResStack(upsampler_params["out_channels"], res_block_kernels[i], res_block_dilation[i]))

        self.mfr = nn.ModuleList(res_blocks)

    def forward(self, x):
        x = self.upsampler(x)
        mfr_out = torch.zeros_like(x, device=x.device)
        for res_block in self.mfr:
            mfr_out = mfr_out + res_block(x)
        return mfr_out

class ResStack(nn.Module):
    def __init__(self, channels_num, kernels: list, block_dilation: list):
        super().__init__()
        n = len(kernels)
        net = []
        for i in range(n):
            net.append(ResBlock(channels_num=channels_num, kernel_size=kernels[i], dilation=block_dilation[i]))

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
    def __init__(self, prolog_params, downsampler_blocks_params, epilog_params):
        super().__init__()
        assert len(epilog_params) == 2
        self.prolog = nn.Sequential(
            nn.Conv1d(**prolog_params),
            nn.LeakyReLU()
        )
        downsamplers = []
        for downsampler_params in downsampler_blocks_params:
            downsamplers.append(nn.Conv1d(**downsampler_params))
            downsamplers.append(nn.LeakyReLU())

        self.downsampler = nn.Sequential(*downsamplers)

        self.epilog = nn.Sequential(
            nn.Conv1d(**epilog_params[0]),
            nn.LeakyReLU(),
            nn.Conv1d(**epilog_params[1])
        )

    def forward(self, x):
        x = self.prolog(x)
        x = self.downsampler(x)
        return self.epilog(x)

class PeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x