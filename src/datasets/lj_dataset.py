from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import read_json, write_json


class LJDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        target_sr=22050,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
        *args,
        **kwargs,
    ):
        if type(data_dir) is str:
            data_dir = Path(data_dir)
        assert data_dir.is_dir(), f"The folder {data_dir} does not exist"

        self.target_sr = target_sr
        self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        self.index_path = data_dir / "index.json"

        if self.index_path.exists():
            index = read_json(str(self.index_path))
        else:
            index = self._create_index(data_dir)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, data_dir):
        index = []

        specs_dir = data_dir / "spectrograms"
        specs_dir.mkdir(parents=True, exist_ok=True)

        for audio_file in tqdm(
            (data_dir / "wavs").iterdir(), desc="Indexing LJspeeeh dataset"
        ):
            entry = {}

            audio_tensor = self.load_audio(audio_file)
            spectrogram = self.get_spectrogram(audio_tensor)
            torch.save(spectrogram, specs_dir / (audio_file.stem + ".pt"))
            entry["spectrogram_path"] = str(specs_dir / (audio_file.stem + ".pt"))

            index.append(entry)

        write_json(index, str(self.index_path))
        return index

    def load_audio(self, path):
        """
        Load audio from disk.

        Args:
            path(str): path to the audio (wav/flac/mp3).
        Returns:
            Audio tensor.
        """
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def get_spectrogram(self, audio_tensor):
        return self.mel_spec_transform(audio_tensor)
