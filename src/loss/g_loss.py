import torch
from torch import nn

from src.loss.pad_tensors import pad_tensors_to_match


class GeneratorLoss(nn.Module):
    def __init__(
        self,
        l_fm,
        l_mel,
    ):
        super().__init__()

        self.l_fm = l_fm
        self.l_mel = l_mel

    def forward(self, real_features, real_specs, gen_outputs, gen_features, gen_specs):
        adv_loss = self.adv_loss(gen_outputs)
        fm_loss = self.fm_loss(real_features, gen_features)
        mel_loss = self.mel_loss(real_specs, gen_specs)
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
                p_real_f, p_gen_f = pad_tensors_to_match(real_f, gen_f)
                loss += torch.mean(torch.abs(p_real_f - p_gen_f))
        return loss

    def mel_loss(self, real_specs, gen_specs):
        p_real_specs, p_gen_specs = pad_tensors_to_match(real_specs, gen_specs)
        loss = torch.mean(torch.abs(real_specs - gen_specs))
        return loss
