import torch
from torch import nn


class DiscriminatorLoss(nn.Module):
    def __init__(
        self,
        l_fm,
        l_mel,
    ):
        super().__init__()

        self.l_fm = l_fm
        self.l_mel = l_mel

    def forward(
        self,
        real_outputs,
        real_features,
        real_specs,
        gen_outputs,
        gen_features,
        gen_specs,
        **batch
    ):
        adv_loss = self.adv_loss(real_outputs, gen_outputs)
        fm_loss = self.fm_loss(real_features, gen_features)
        mel_loss = self.mel_loss(real_specs, gen_specs)
        loss = adv_loss + self.l_fm * fm_loss + self.l_mel * mel_loss
        return {
            "loss": loss,
            "adv_loss": adv_loss,
            "fm_loss": fm_loss,
            "mel_loss": mel_loss,
        }

    @staticmethod
    def adv_loss(real_outputs, gen_outputs):
        loss = 0
        for real_output, gen_output in zip(real_outputs, gen_outputs):
            loss += torch.mean((real_output - 1) ** 2) + torch.mean(gen_output**2)
        return loss

    @staticmethod
    def fm_loss(real_features, gen_features):
        loss = 0
        for real_feature, gen_feature in zip(real_features, gen_features):
            loss += torch.mean(torch.abs(real_feature - gen_feature))
        return loss

    @staticmethod
    def mel_loss(real_specs, gen_specs):
        loss = 0
        for real_spec, gen_spec in zip(real_specs, gen_specs):
            loss += torch.mean(torch.abs(real_spec - gen_spec))
        return loss
