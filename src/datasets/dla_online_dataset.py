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
            index = self._create_index(part, ratio=ratio)

        # TODO utilize torch.data.utils.RandomSampler
        print("preparing dataset ...")
        self._pairs = np.stack(
            np.unravel_index(np.arange(len(index) ** 2), (len(index), len(index)))
        ).T
        self._pairs = self._pairs[self._pairs[:, 0] < self._pairs[:, 1]]

        if shuffle:
            for i in tqdm(
                range(len(self._pairs)), total=len(self._pairs), desc="shuffling ..."
            ):
                j = random.randint(i, len(self._pairs) - 1)
                self._pairs[i], self._pairs[j] = self._pairs[j], self._pairs[i]

        if limit is not None:
            self._pairs = self._pairs[:limit]

        super().__init__(index, *args, **kwargs)

    @staticmethod
    def _assert_index_is_valid(index):
        pass

    def _create_index(self, part: str, ratio=None) -> list:
        """
        Args:
            part (str): Partition name.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        # we mix samples only from original train set
        audio_path = self.dir / "audio" / "train"

        speakers = {}
        for file in tqdm(os.listdir(audio_path / "mix"), desc="creating index ..."):
            speaker_1_id = file.split("_")[0]
            speaker_2_id = file.split("_")[1].split(".")[0]

            speaker_1_path = audio_path / "s1" / file
            speaker_2_path = audio_path / "s2" / file
            mouth_1_path = self.dir / "mouths" / (speaker_1_id + ".npz")
            mouth_2_path = self.dir / "mouths" / (speaker_2_id + ".npz")
            visual_1_path = self.dir / "visual_embeddings" / (speaker_1_id + ".npz")
            visual_2_path = self.dir / "visual_embeddings" / (speaker_2_id + ".npz")

            speakers[speaker_1_id] = {
                "speaker_path": str(speaker_1_path),
                "mouth_path": str(mouth_1_path),
                "visual_path": str(visual_1_path),
            }

            speakers[speaker_2_id] = {
                "speaker_path": str(speaker_2_path),
                "mouth_path": str(mouth_2_path),
                "visual_path": str(visual_2_path),
            }

        print("splitting ...")
        train_index, val_index = self._random_split(list(speakers.values()), ratio)

        write_json(train_index, self._get_index_path("train"))
        write_json(val_index, self._get_index_path("val"))

        if part == "train":
            return train_index
        else:
            return val_index

    def _get_index_path(self, part):
        return self.dir / f"{part}_online_index.json"

    @staticmethod
    def _random_split(index, ratio):
        roll = np.random.uniform(0, 1, len(index))
        train_indeces = roll < ratio

        train_index, val_index = [], []
        for item, flag in zip(index, train_indeces):
            if flag:
                train_index.append(item)
            else:
                val_index.append(item)
        return train_index, val_index

    def __getitem__(self, ind):
        i, j = self._pairs[ind]

        def get_object(idx, num):
            data_dict = dict(self._index[idx])
            data_dict[f"speaker_{num}_wav"] = self.load_object(
                data_dict["speaker_path"]
            )
            data_dict[f"mouth_{num}_npz"] = self.load_object(data_dict["mouth_path"])
            data_dict[f"visual_{num}_emb"] = self.load_object(data_dict["visual_path"])

            return data_dict

        data_dict = {}
        data_dict.update(get_object(i, 1))
        data_dict.update(get_object(j, 2))

        instance_data = self.preprocess_data(data_dict)

        instance_data["mix_wav"] = (
            instance_data["speaker_1_wav"] + instance_data["speaker_2_wav"]
        )

        data_dict["mix_visual"] = np.stack(
            (data_dict["visual_1_emb"], data_dict["visual_2_emb"]), axis=0
        )

        return instance_data

    def __len__(self):
        return len(self._pairs)
