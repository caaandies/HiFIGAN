from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import FastSpeech2ConformerModel, FastSpeech2ConformerTokenizer

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import read_json, write_json

PARTS = ["test", "train", "val"]


class LJDataset(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        if type(data_dir) is str:
            data_dir = Path(data_dir)
        assert data_dir.is_dir(), f"The folder {data_dir} does not exist"

        self.index_path = data_dir / "index.json"

        if self.index_path.exists():
            index = read_json(str(self.index_path))
        else:
            index = self._create_index(data_dir)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, data_dir):
        index = []

        tokenizer = FastSpeech2ConformerTokenizer.from_pretrained(
            "espnet/fastspeech2_conformer"
        )
        model = FastSpeech2ConformerModel.from_pretrained(
            "espnet/fastspeech2_conformer"
        )

        specs_dir = data_dir / "spectrograms"
        specs_dir.mkdir(parents=True, exist_ok=True)

        with open(data_dir / "metadata.csv", "r") as metadata:
            for line in tqdm(metadata, desc="Indexing ljspeech dataset"):
                entry = {}

                id, text, norm_text = line.split("|")
                # text = text.strip()
                # norm_text = text.strip()

                with torch.no_grad():
                    inputs = tokenizer(norm_text, return_tensors="pt")
                    input_ids = inputs["input_ids"]
                    output_dict = model(input_ids, return_dict=True)
                    spectrogram = output_dict["spectrogram"]

                torch.save(spectrogram, specs_dir / (id + ".pt"))
                entry["spectrogram_path"] = str(specs_dir / (id + ".pt"))

                entry["target_audio_path"] = data_dir / "wavs" / (id + ".wav")
                if entry["target_audio_path"].exists():
                    entry["target_audio_path"] = str(entry["target_audio_path"])
                else:
                    entry["target_audio_path"] = None

                index.append(entry)

        write_json(index, str(self.index_path))
        return index
