import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

SLOPE = 0.1


class MPDSubDiscriminator(nn.Module):
    def __init__(self, period, norm):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList(
            [
                norm(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=64,
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                ),
                norm(
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                ),
                norm(
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                ),
                norm(
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                ),
                norm(
                    nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=(5, 1),
                        padding=(2, 0),
                    )
                ),
                norm(
                    nn.Conv2d(
                        in_channels=1024,
                        out_channels=1,
                        kernel_size=(3, 1),
                        padding=(1, 0),
                    )
                ),
            ]
        )

    def forward(self, x):  # B * 1 * T
        batch_size = x.shape[0]
        channels = x.shape[1]
        duration = x.shape[2]
        if duration % self.period != 0:
            x = F.pad(x, (0, self.period - duration % self.period))
        duration = x.shape[2]
        x = x.view(
            batch_size, channels, duration // self.period, self.period
        )  # B * 1 * (T / p) * p

        feature_maps = []
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x)
            x = F.leaky_relu(x, SLOPE)
            feature_maps.append(x)
        x = self.convs[-1](x)
        return x, feature_maps


class MPD(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()

        self.subs = nn.ModuleList()
        for period in periods:
            self.subs.append(MPDSubDiscriminator(period, weight_norm))

    def forward(self, x):
        feature_maps = []
        outputs = []
        for i in range(len(self.subs)):
            output, f_map = self.subs[i](x)
            outputs.append(output)
            feature_maps.append(f_map)
        return outputs, feature_maps
