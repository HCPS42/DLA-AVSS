import logging
import random
from typing import Any, List

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    _attrs = [
        "mix_path",
        "speaker_1_path",
        "speaker_2_path",
        "mouth_1_path",
        "mouth_2_path",
    ]
    _attrs_mapping = {
        "mix_path": "mix_wav",
        "speaker_1_path": "speaker_1_wav",
        "speaker_2_path": "speaker_2_wav",
        "mouth_1_path": "mouth_1_npz",
        "mouth_2_path": "mouth_2_npz",
    }

    def __init__(
        self, index, limit=None, shuffle_index=False, instance_transforms=None
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
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._assert_index_is_valid(index)

        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: List[dict] = index

        self.instance_transforms = instance_transforms

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
        for path_key in filter(lambda key: key in data_dict, self._attrs):
            object_key = self._attrs_mapping[path_key]
            data_dict[object_key] = self.load_object(data_dict[path_key])

        instance_data = self.preprocess_data(data_dict)

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_object(self, path: str) -> Any:
        """
        Load object from disk.

        Args:
            path (str): path to the object.
        Returns:
            data_object (Any): object loaded from disk.
        """
        if path.endswith((".wav", ".flac", ".mp3")):
            audio, sr = torchaudio.load(path)
            assert sr == 16000
            return audio
        elif path.endswith(".npz"):
            with np.load(path) as data:
                return data
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            if "get_spectrogram" in self.instance_transforms:
                get_spectrogram = self.instance_transforms["get_spectrogram"]

                # apply get_spectrogram transform to all waveforms
                for key in filter(
                    lambda key: key.endswith("_wav"), list(instance_data.keys())
                ):
                    spec_key = key.removesuffix("_wav") + "_spec"
                    instance_data[spec_key] = get_spectrogram(instance_data[key])

            for transform_name in filter(
                lambda name: name != "get_spectrogram",
                list(self.instance_transforms.keys()),
            ):
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name].unsqueeze(0)).squeeze(0)
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        some condition.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
        # Filter logic
        pass

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
        attrs = ["mix_path", "mouth_1_path", "mouth_2_path"]
        for entry in index:
            for attr in attrs:
                if attr not in entry:
                    raise KeyError(f"Each dataset item should include field '{attr}'")

    @staticmethod
    def _sort_index(index):
        """
        Sort index via some rules.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting and after filtering.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): sorted list, containing dict for each element
                of the dataset. The dict has required metadata information,
                such as label and object path.
        """
        return sorted(index, key=lambda x: x["mix_path"])

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
