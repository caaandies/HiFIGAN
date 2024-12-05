import torch
import torch.nn as nn
from hydra.utils import instantiate


class HiFIGAN(nn.Module):
    def __init__(
        self,
        mpd_config,
        msd_config,
        gen_config,
    ):
        super().__init__()
        self.mpd = instantiate(mpd_config)
        self.msd = instantiate(msd_config)
        self.gen = instantiate(gen_config)
