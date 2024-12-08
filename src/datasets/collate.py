import torch
from torch.nn.utils.rnn import pad_sequence

from src.transforms.spec_transform import MelSpectrogramConfig


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    batch = {}

    batch["spectrogram"] = pad_sequence(
        [item["spectrogram"].squeeze(0).transpose(0, 1) for item in dataset_items],
        padding_value=MelSpectrogramConfig.pad_value,
        batch_first=True,
    ).transpose(1, 2)

    if "wav" in dataset_items[0]:
        batch["wav"] = pad_sequence(
            [item["wav"].squeeze(0) for item in dataset_items], batch_first=True
        )

    return batch
