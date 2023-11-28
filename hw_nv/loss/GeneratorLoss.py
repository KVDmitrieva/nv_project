import torch
import torch.nn as nn


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, gen_discriminator_out, mel, gen_mel, real_feature_map, gen_feature_map, **batch):
        adv_loss = 0.0
        for d_out in gen_discriminator_out:
            adv_loss += self.mse_loss(d_out, torch.ones_like(d_out, device=gen_discriminator_out.device))

        mel_loss = self.l1_loss(mel, gen_mel)
        feature_loss = 0.0

        for real, gen in zip(real_feature_map, gen_feature_map):
            tmp = 0.0
            for i in range(len(real)):
                tmp += self.l1_loss(real[i], gen[i])
            feature_loss += tmp / len(real)

        return {
            "gen_loss": adv_loss + mel_loss + feature_loss,
            "adv_loss": adv_loss,
            "mel_loss": mel_loss,
            "feature_loss": feature_loss
        }
