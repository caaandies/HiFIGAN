from itertools import repeat

from hydra.utils import instantiate
from torch.utils.data import random_split

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def get_dataloaders(config, device, train_size=0.85):
    # dataset partitions init
    dataset = instantiate(config.datasets)  # instance transforms are defined inside

    train_size = int(train_size * len(dataset))
    val_size = len(dataset) - train_size

    datasets = {}
    datasets["train"], dataset["val"] = random_split(dataset, [train_size, val_size])
    # dataloaders init
    dataloaders = {}

    for dataset_partition in datasets.keys():
        dataset = datasets[dataset_partition]

        assert config.dataloader.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders
