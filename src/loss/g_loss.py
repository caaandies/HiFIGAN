import torch
from torch import nn


class GeneratorLoss(nn.Module):
    def __init__(
        self,
        spec_transform,
        l_fm,
        l_mel,
    ):
        super().__init__()

        self.spec_transform = spec_transform
        self.l_fm = l_fm
        self.l_mel = l_mel

    def forward(self, real_features, real_specs, gen_outputs, gen_features, gen_wavs):
        adv_loss = self.adv_loss(gen_outputs)
        fm_loss = self.fm_loss(real_features, gen_features)
        mel_loss = self.mel_loss(real_specs, gen_wavs)
        loss = adv_loss + self.l_fm * fm_loss + self.l_mel * mel_loss
        return {
            "loss": loss,
            "adv_loss": adv_loss,
            "fm_loss": fm_loss,
            "mel_loss": mel_loss,
        }

    def adv_loss(self, gen_outputs):
        loss = 0
        for gen_out in gen_outputs:
            loss += torch.mean((gen_out - 1) ** 2)
        return loss

    def fm_loss(self, real_features, gen_features):
        loss = 0
        for sub_real_features, sub_gen_features in zip(
            real_features, gen_features
        ):  # submodules
            for real_f, gen_f in zip(
                sub_real_features, sub_gen_features
            ):  # levels in submodule
                loss += torch.mean(torch.abs(real_f - gen_f))
        return loss

    def mel_loss(self, real_specs, gen_wavs):
        gen_specs = self.spec_transform(gen_wavs)
        loss = torch.mean(torch.abs(real_specs - gen_specs))
        return loss
