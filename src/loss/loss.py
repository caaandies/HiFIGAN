import torch
from torch import nn

from src.loss.d_loss import DiscriminatorLoss
from src.loss.g_loss import GeneratorLoss


class Loss(nn.Module):
    def __init__(
        self,
        gen_l_fm=2,
        gen_l_mel=45,
    ):
        super().__init__()

        self.mpd_loss = DiscriminatorLoss()
        self.msd_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss(gen_l_fm, gen_l_mel)
