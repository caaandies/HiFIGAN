import torch.nn as nn
import torch.nn.functional as F

SLOPE = 0.1


def calc_padding_with_dilation(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation_rates):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(dilation_rates)):
            modules = []
            for j in range(len(dilation_rates[i])):
                modules.append(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        dilation=dilation_rates[i][j],
                        padding=calc_padding_with_dilation(
                            kernel_size, dilation_rates[i][j]
                        ),
                    )
                )
                modules.append(nn.LeakyReLU(SLOPE))

            self.blocks.append(nn.Sequential(*modules))

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class MRF(nn.Module):
    def __init__(self, channels, kernel_sizes, dilation_rates):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            self.res_blocks.append(
                ResBlock(channels, kernel_sizes[i], dilation_rates[i])
            )

    def forward(self, x):
        output = self.res_blocks[0](x)
        for i in range(1, len(self.res_blocks)):
            output = output + self.res_blocks[i](x)
        return output


class Generator(nn.Module):
    def __init__(
        self, in_channels, hidden_dim, upsample_kernels, mrf_kernels, mrf_dilation_rates
    ):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, hidden_dim, kernel_size=7)

        conv_out_channels = hidden_dim // (2 ** len(upsample_kernels))
        self.conv_out = nn.Conv1d(conv_out_channels, 1, kernel_size=7)

        modules = []
        for i in range(len(upsample_kernels)):
            modules.append(nn.LeakyReLU(SLOPE))
            modules.append(
                nn.ConvTranspose1d(
                    in_channels=hidden_dim // (2**i),
                    out_channels=hidden_dim // (2 ** (i + 1)),
                    kernel_size=upsample_kernels[i],
                    stride=upsample_kernels[i] // 2,
                    padding=(upsample_kernels[i] - 1) // 2,
                )
            )
            modules.append(
                MRF(
                    channels=hidden_dim // (2 ** (i + 1)),
                    kernel_sizes=mrf_kernels,
                    dilation_rates=mrf_dilation_rates,
                )
            )

        self.processing = nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.processing(x)
        x = F.leaky_relu(x, SLOPE)
        x = self.conv_out(x)
        x = F.tanh(x)
        return x
