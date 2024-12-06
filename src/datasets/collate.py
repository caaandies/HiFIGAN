import torch
from torch.nn.utils.rnn import pad_sequence


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
        [item["spectrogram"] for item in dataset_items], batch_first=True
    )
    batch["spectrogram_len"] = torch.tensor(
        [item["spectrogram_len"] for item in dataset_items]
    )

    if "wav" in dataset_items[0]:
        batch["wav"] = pad_sequence(
            [item["wav"] for item in dataset_items], batch_first=True
        )
        batch["wav_len"] = torch.tensor([item["wav_len"] for item in dataset_items])

    return batch
