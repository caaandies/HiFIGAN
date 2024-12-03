import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm

SLOPE = 0.1


class MSDSubDiscriminator(nn.Module):
    def __init__(self, norm):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                norm(
                    nn.Conv1d(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=15,
                        stride=1,
                        padding=7,
                    )
                ),
                norm(
                    nn.Conv1d(
                        in_channels=16,
                        out_channels=64,
                        kernel_size=41,
                        stride=4,
                        groups=4,
                        padding=20,
                    )
                ),
                norm(
                    nn.Conv1d(
                        in_channels=64,
                        out_channels=256,
                        kernel_size=41,
                        stride=4,
                        groups=16,
                        padding=20,
                    )
                ),
                norm(
                    nn.Conv1d(
                        in_channels=256,
                        out_channels=1024,
                        kernel_size=41,
                        stride=4,
                        groups=64,
                        padding=20,
                    )
                ),
                norm(
                    nn.Conv1d(
                        in_channels=1024,
                        out_channels=1024,
                        kernel_size=41,
                        stride=4,
                        groups=256,
                        padding=20,
                    )
                ),
                norm(
                    nn.Conv1d(
                        in_channels=1024,
                        out_channels=1024,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                    )
                ),
                norm(
                    nn.Conv1d(
                        in_channels=1024,
                        out_channels=1,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                ),
            ]
        )

    def forward(self, x):
        feature_maps = []
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x)
            x = F.leaky_relu(x, SLOPE)
            feature_maps.append(x)
        x = self.convs[-1](x)
        return x, feature_maps


class MSD(nn.Module):
    def __init__(self):
        super().__init__()

        self.subs = nn.ModuleList(
            [
                MSDSubDiscriminator(spectral_norm),
                MSDSubDiscriminator(weight_norm),
                MSDSubDiscriminator(weight_norm),
            ]
        )

        self.pools = nn.ModuleList(
            [
                nn.Identity(),
                nn.AvgPool1d(kernel_size=4, stride=2, padding=1),
                nn.AvgPool1d(kernel_size=4, stride=2, padding=1),
            ]
        )

    def forward(self, x):  # B * 1 * T
        feature_maps = []
        outputs = []
        for i in range(len(self.subs)):
            x = self.pools[i](x)
            output, f_map = self.subs[i](x)
            outputs.append(output)
            feature_maps.append(f_map)
        return outputs, feature_maps
