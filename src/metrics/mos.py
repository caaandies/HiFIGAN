import torch
from torchaudio.functional import resample
from wvmos import get_wvmos

from src.metrics.base_metric import BaseMetric


class MOS(BaseMetric):
    def __init__(self, name=None, sr=22050, target_sr=16000, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        self.sr = sr
        self.target_sr = target_sr
        self.model = get_wvmos(cuda=True).eval()

    @torch.no_grad
    def __call__(self, audio):  # (B * 1 *T) / (1 * T)
        r_audio = self.resample(audio, self.sr, self.target_sr)
        self.model(r_audio.cuda())
