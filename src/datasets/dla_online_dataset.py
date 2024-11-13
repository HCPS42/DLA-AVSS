import itertools
import os
import random
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class DLAOnlineDataset(BaseDataset):
    """
    Dataset for DLA course assignment.
    """

    def __init__(
        self, dir, part="train", ratio=0.9, shuffle=False, limit=None, *args, **kwargs
    ):
        """
        Args:
            dir (str): Path to the custom directory.
            part (str): Partition name.
        """
        assert part in ("train", "val")

        self.dir = ROOT_PATH / Path(dir)
        index_path = self._get_index_path(part)

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            raise ValueError("No index. Run scripts/create_index.py first.")

        print("preparing dataset ...")
        self._pairs = np.stack(
            np.unravel_index(np.arange(len(index) ** 2), (len(index), len(index)))
        ).T
        self._pairs = self._pairs[self._pairs[:, 0] < self._pairs[:, 1]]

        if limit is not None:
            self.len = limit
        else:
            self.len = len(self._pairs)

        super().__init__(index, *args, **kwargs)

    @staticmethod
    def _assert_index_is_valid(index):
        pass

    def _get_index_path(self, part):
        return self.dir / f"{part}_online_index.json"

    def __getitem__(self, ind):
        i, j = self._pairs[ind]

        def get_object(idx, num):
            data_dict = dict(self._index[idx])
            data_dict[f"speaker_{num}_wav"] = self.load_object(
                data_dict["speaker_path"]
            )
            data_dict[f"mouth_{num}_npz"] = self.load_object(data_dict["mouth_path"])

            return data_dict

        data_dict = {}
        data_dict.update(get_object(i, 1))
        data_dict.update(get_object(j, 2))

        instance_data = self.preprocess_data(data_dict)

        instance_data["mix_wav"] = (
            instance_data["speaker_1_wav"] + instance_data["speaker_2_wav"]
        )

        return instance_data

    def __len__(self):
        return self.len
