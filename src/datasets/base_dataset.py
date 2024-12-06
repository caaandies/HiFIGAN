import logging
import random
from typing import List

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        index,
        limit=None,
        shuffle_index=False,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        self._assert_index_is_valid(index)

        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: List[dict] = index

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]

        spectrogram_path = data_dict["spectrogram_path"]
        spectrogram = torch.load(spectrogram_path, weights_only=True)
        instance_data = {
            "spectrogram": spectrogram,
            "spectrogram_len": spectrogram.shape[-1],
        }

        wav_path = data_dict["wav_path"]
        if wav_path is not None:
            wav = torch.load(wav_path, weights_only=True)
            instance_data["wav"] = wav
            instance_data["wav_len"] = wav.shape[-1]

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert (
                "spectrogram_path" in entry
            ), "Each dataset item should include field 'spectrogram_path' - path to spectrogram tensor to generate audio from."

            assert (
                "wav_path" in entry
            ), "Each dataset item should include field 'wav_path' - path to target wav tensor or None."

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
