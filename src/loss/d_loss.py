import torch
from torch import nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_outputs, gen_outputs):
        loss = self.adv_loss(real_outputs, gen_outputs)
        return {"loss": loss, "adv_loss": loss}

    def adv_loss(self, real_outputs, gen_outputs):
        loss = 0
        for real_out, gen_out in zip(real_outputs, gen_outputs):
            loss += torch.mean((real_out - 1) ** 2) + torch.mean(gen_out**2)
        return loss
