import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from pyannote.audio import Inference, Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.init_utils import set_random_seed
from src.utils.io_utils import write_json

sys.path += [".", ".."]
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to the folder with data"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--train-threshold",
        type=float,
        default=0.9,
        help="Discard utterances with the higher cosine similarity to some other utterance in the common (train + val) set",
    )
    parser.add_argument(
        "--val-threshold",
        type=float,
        default=0.7,
        help="Discard utterances from the val set with the higher cosine similarity to some other utterance in the train set",
    )
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
    mouths_dir = Path(args.data_dir) / "mouths"

    utterances = {}
    for file in os.listdir(audio_dir / "s1"):
        id1 = file.split("_")[0]
        utterances[id1] = audio_dir / "s1" / file

    for file in os.listdir(audio_dir / "s2"):
        id2 = file.split("_")[1].split(".")[0]
        utterances[id2] = audio_dir / "s2" / file

    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM").to(
        args.device
    )
    inference = Inference(model, window="whole")

    embeddings = []
    for id, file in tqdm(utterances.items()):
        features = torch.nn.functional.normalize(torch.tensor(inference(file)), dim=0)
        embeddings.append(features)

    embeddings = torch.stack(embeddings)
    embeddings = embeddings.squeeze(1).numpy()
    ids = list(utterances.keys())

    cosine_sim_matrix = cosine_similarity(embeddings)

    index = []
    filtered = 0
    for i, file in enumerate(utterances.values()):
        # leave only one utterance from the similar ones
        if np.all(cosine_sim_matrix[:i, i] < args.train_threshold):
            index.append(
                {
                    "id": ids[i],
                    "speaker_path": str(file),
                    "mouth_path": str(mouths_dir / f"{ids[i]}.npz"),
                }
            )
        else:
            filtered += 1

    print(
        f"Filtered {filtered} utterances from common set ({filtered / len(utterances) * 100:.2f}%)"
    )
    print(f"Total number of utterances left: {len(index)}")

    train_idxs, val_idxs = train_test_split(
        np.arange(len(index)),
        train_size=args.train_ratio,
        random_state=args.seed,
        shuffle=True,
    )

    _val_idxs = []
    filtered = 0
    for val_idx in val_idxs:
        # drop utterance from val set if it is too similar to the one in train set
        if np.all(cosine_sim_matrix[i, train_idxs] < args.val_threshold):
            _val_idxs.append(val_idx)
        else:
            filtered += 1
    val_idxs = np.array(_val_idxs)

    print(
        f"Filtered {filtered} utterances from val set ({filtered / len(val_idxs) * 100:.2f}%)"
    )
    print(f"Total number of utterances left: {len(val_idxs)}")

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

        torchaudio.save(speaker_1_path, mix, sr)
        torchaudio.save(speaker_2_path, wav1, sr)
        torchaudio.save(mix_path, wav2, sr)

        val_index.append(
            {
                "mix_path": str(mix_path),
                "speaker_1_path": str(speaker_1_path),
                "speaker_2_path": str(speaker_2_path),
                "mouth_1_path": str(index[i]["mouth_path"]),
                "mouth_2_path": str(index[j]["mouth_path"]),
            }
        )

    write_json(train_index, data_dir / "train_online_index.json")
    write_json(val_index, data_dir / "val_online_index.json")
