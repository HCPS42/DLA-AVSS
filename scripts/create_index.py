import argparse
import os
import random
import sys
from pathlib import Path

sys.path += [".", ".."]

import numpy as np
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.init_utils import set_random_seed
from src.utils.io_utils import write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to the folder with data"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.98,
        help="Ratio of train set (utterance-wise)",
    )
    parser.add_argument(
        "--val-size", type=int, default=5000, help="Size of val set (mixes-wise)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed")

    args = parser.parse_args()
    set_random_seed(args.seed)

    print("Using parameters:")
    print(args)

    audio_dir = Path(args.data_dir) / "audio" / "train"
    embeddings_dir = Path(args.data_dir) / "visual_embeddings"

    utterances = {}
    for file in os.listdir(audio_dir / "s1"):
        id1 = file.split("_")[0]
        utterances[id1] = audio_dir / "s1" / file

    for file in os.listdir(audio_dir / "s2"):
        id2 = file.split("_")[1].split(".")[0]
        utterances[id2] = audio_dir / "s2" / file

    ids = list(utterances.keys())

    index = []
    for i, file in enumerate(utterances.values()):
        index.append(
            {
                "id": ids[i],
                "speaker_path": str(file),
                "visual_path": str(embeddings_dir / f"{ids[i]}.npz"),
            }
        )

    train_idxs, val_idxs = train_test_split(
        np.arange(len(index)),
        train_size=args.train_ratio,
        random_state=args.seed,
        shuffle=True,
    )

    train_index = [index[i] for i in train_idxs]
    val_index = []

    data_dir = Path(args.data_dir)
    (data_dir / "audio" / "val_online" / "mix").mkdir(parents=True, exist_ok=True)
    (data_dir / "audio" / "val_online" / "s1").mkdir(parents=True, exist_ok=True)
    (data_dir / "audio" / "val_online" / "s2").mkdir(parents=True, exist_ok=True)

    for _ in tqdm(range(args.val_size), total=args.val_size):
        i, j = 0, 0
        while i == j:
            i = random.choice(val_idxs)
            j = random.choice(val_idxs)

        id1 = index[i]["id"]
        id2 = index[j]["id"]
        filename = f"{id1}_{id2}.wav"

        wav1, sr = torchaudio.load(utterances[id1])
        wav2, sr = torchaudio.load(utterances[id2])
        mix = wav1 + wav2

        speaker_1_path = data_dir / "audio" / "val_online" / "s1" / filename
        speaker_2_path = data_dir / "audio" / "val_online" / "s2" / filename
        mix_path = data_dir / "audio" / "val_online" / "mix" / filename

        torchaudio.save(speaker_1_path, wav1, sr)
        torchaudio.save(speaker_2_path, wav2, sr)
        torchaudio.save(mix_path, mix, sr)

        val_index.append(
            {
                "mix_path": str(mix_path),
                "speaker_1_path": str(speaker_1_path),
                "speaker_2_path": str(speaker_2_path),
                "visual_1_path": str(index[i]["visual_path"]),
                "visual_2_path": str(index[j]["visual_path"]),
            }
        )

    write_json(train_index, data_dir / "train_online_index.json")
    write_json(val_index, data_dir / "val_online_index.json")
