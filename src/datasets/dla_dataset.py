import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class DLADataset(BaseDataset):
    """
    Dataset for DLA course assignment.
    """

    def __init__(self, dir, part="train", *args, **kwargs):
        """
        Args:
            dir (str): Path to the custom directory.
            part (str): Partition name.
        """
        self.dir = ROOT_PATH / Path(dir)
        index_path = self._get_index_path(part)

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(part)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, part) -> list:
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            part (str): Partition name.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        assert part in ("train", "test", "val", "")
        index = []

        audio_path = self._get_audio_path(part)
        index_path = self._get_index_path(part)

        for file in tqdm(
            os.listdir(audio_path / "mix"), desc=f"creating {part} index ..."
        ):
            data_dict = {}

            data_dict["mix_path"] = str(audio_path / "mix" / file)

            if (audio_path / "s1" / file).exists():
                data_dict["speaker_1_path"] = str(audio_path / "s1" / file)

            if (audio_path / "s2" / file).exists():
                data_dict["speaker_2_path"] = str(audio_path / "s2" / file)

            speaker_1_id = file.split("_")[0]
            speaker_2_id = file.split("_")[1].split(".")[0]

            data_dict["visual_1_path"] = str(
                self.dir / "visual_embeddings" / (speaker_1_id + ".npz")
            )

            data_dict["visual_2_path"] = str(
                self.dir / "visual_embeddings" / (speaker_2_id + ".npz")
            )

            index.append(data_dict)

        write_json(index, str(index_path))

        return index

    def _get_audio_path(self, part):
        return self.dir / "audio" / part

    def _get_index_path(self, part):
        return self.dir / f"{part}_index.json"


class CustomDirDataset(DLADataset):
    def __init__(self, dir, *args, **kwargs):
        super().__init__(dir, part="", *args, **kwargs)

    def _get_audio_path(self, part):
        return self.dir / "audio"

    def _get_index_path(self, part):
        return self.dir / "index.json"
