import torch.nn as nn

from hw_nv.model.base_model import BaseModel
from hw_nv.model.utils import *


__all__ = ["Generator", "MultiScaleDiscriminator", "MultiPeriodDiscriminator"]

class Generator(BaseModel):
    def __init__(self, prolog_params, upsampler_blocks_params, epilog_params):
        super().__init__()
        self.prolog = nn.Conv1d(**prolog_params)

        upsamplers = []
        for upsampler_params in upsampler_blocks_params:
            upsamplers.append(UpsamplerBlock(**upsampler_params))

        self.upsampler = nn.Sequential(*upsamplers)
        self.epilog = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(**epilog_params),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.prolog(x)
        x = self.upsampler(x)
        return self.epilog(x)

class MultiScaleDiscriminator(BaseModel):
    def __init__(self):
        super().__init__()

    def forward(self, x, **batch):
        return x


class MultiPeriodDiscriminator(BaseModel):
    def __init__(self):
        super().__init__()

    def forward(self, x, **batch):
        return x