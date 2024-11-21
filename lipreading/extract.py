import os
import urllib

import numpy as np
import torch
from lipreading_model import Lipreading
from torchvision.transforms import CenterCrop, Compose, Normalize
from tqdm import tqdm

from utils import load_model

if __name__ == "__main__":
    # resnet18_dctcn_video_boundary
    model_path = "lipreading/model.pth"

    densetcn_options = {
        "block_config": [3, 3, 3, 3],
        "growth_rate_set": [384, 384, 384, 384],
        "reduced_size": 512,
        "kernel_size_set": [3, 5, 7],
        "dilation_size_set": [1, 2, 5],
        "squeeze_excitation": True,
        "dropout": 0.2,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Lipreading(
        densetcn_options=densetcn_options,
    ).to(device)

    print(model_path)

    model = load_model("lipreading/model.pth", model, allow_size_mismatch=True)

    model.eval()

    landmarks_path = "data/dla_dataset/mouths"
    embeddings_path = "data/dla_dataset/visual_embeddings"
    os.makedirs(embeddings_path, exist_ok=True)

    landmark_files = [f for f in os.listdir(landmarks_path) if f.endswith(".npz")]

    preprocessing = Compose(
        [Normalize(0.0, 255.0), CenterCrop((88, 88)), Normalize(0.421, 0.165)]
    )

    for landmark_file in tqdm(landmark_files):
        landmark_file_path = os.path.join(landmarks_path, landmark_file)
        landmark = torch.from_numpy(np.load(landmark_file_path)["data"]).to(device)
        landmark = landmark.float()
        landmark = preprocessing(landmark)
        landmark = landmark.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            embeddings = model(landmark, [landmark.size(1)])
            embeddings = embeddings.squeeze(0).cpu().numpy()

        embeddings_file_path = os.path.join(embeddings_path, landmark_file)

        if not os.path.exists(embeddings_file_path):
            np.savez(embeddings_file_path, embeddings=embeddings.cpu().numpy())
