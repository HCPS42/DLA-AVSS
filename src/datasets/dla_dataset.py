import os

import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class DLADataset(BaseDataset):
    """
    Dataset for DLA course assignment.
    """

    def __init__(self, part="train", *args, **kwargs):
        """
        Args:
            part (str): partition name
        """
        index_path = self._get_index_path(part)

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(part)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, part) -> dict:
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            part (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        audio_path = ROOT_PATH / "data" / "dla_dataset" / "audio" / part
        index_path = self._get_index_path(part)

        for file in tqdm(
            os.listdir(audio_path / "mix"), desc=f"creating {part} index ..."
        ):
            data_dict = {}
            data_dict["mix_path"] = str(audio_path / "mix" / file)
            if part != "test":
                data_dict["speaker1_path"] = str(audio_path / "s1" / file)
                data_dict["speaker2_path"] = str(audio_path / "s2" / file)

            data_dict["landmarks_path"] = str(
                ROOT_PATH
                / "data"
                / "dla_dataset"
                / "mouths"
                / (file.split("_")[0] + ".npz")
            )

            index.append(data_dict)

        write_json(index, str(index_path))

        return index

    def _get_index_path(self, part):
        return ROOT_PATH / "data" / "dla_dataset" / f"{part}_index.json"
