import torch
import torch.nn as nn

from src.model.discriminators import MPD, MSD
from src.model.generator import Generator


class HiFIGAN(nn.Module):
    def __init__(
        self,
        mpd: MPD,
        msd: MSD,
        gen: Generator,
    ):
        super().__init__()
        self.mpd = mpd
        self.msd = msd
        self.gen = gen
