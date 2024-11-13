from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json


class DLAOnlineDataset(BaseDataset):
    """
    Dataset for DLA course assignment.
    """

    def __init__(self, dir, part="train", *args, **kwargs):
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

        super().__init__(index, *args, **kwargs)

    @staticmethod
    def _assert_index_is_valid(index):
        pass

    def _get_index_path(self, part):
        return self.dir / f"{part}_online_index.json"

    def __getitem__(self, ind):
        i, j = self._get_pair(ind)

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
        return len(self._index) ** 2

    def _get_pair(self, ind):
        return np.unravel_index(ind, (len(self._index), len(self._index)))
