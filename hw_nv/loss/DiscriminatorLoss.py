import torch
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, gen_discriminator_out, real_discriminator_out, **batch):
        adv_loss = 0.0
        for d_out in gen_discriminator_out:
            adv_loss += self.mse_loss(d_out, torch.zeros_like(d_out, device=gen_discriminator_out.device))

        for d_out in real_discriminator_out:
            adv_loss += self.mse_loss(d_out, torch.ones_like(d_out, device=gen_discriminator_out.device))

        return {
            "discriminator_loss": adv_loss,
        }
