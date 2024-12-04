import torch
from torch import nn


class GeneratorLoss(nn.Module):
    def __init__(
        self,
        l_adv,
        l_fm,
        l_mel,
    ):
        super().__init__()

    def forward(self, gen_outputs, **batch):
        loss = self.adv_loss(gen_outputs)
        return {"loss": loss}

    @staticmethod
    def adv_loss(gen_outputs):
        loss = 0
        for gen_output in gen_outputs:
            loss += torch.mean((gen_output - 1) ** 2)
        return loss
