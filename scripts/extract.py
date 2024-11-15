import os

import numpy as np
import torch
from lipreading.model import Lipreading
from lipreading.utils import load_json, load_model
from torchvision.transforms import CenterCrop, Compose, Normalize
from tqdm import tqdm


def extract_visual_embeddings():
    """
    Follow these steps to extract visual embeddings:
    1. Clone https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks?tab=readme-ov-file#how-to-extract-embeddings.
    2. For Windows users: replace colons with hyphens in get_save_folder function in lipreading/utils.py.
    3. Install requirements, including pytorch with CUDA support.
    4. Download the pretrained model from the link in the README.
    5. Put the mouth videos in the `landmarks` folder.
    6. Run the script from the root of the repository.
    7. The embeddings will be saved in the `visual_embeddings` folder.
    8. Copy the embeddings to the `data/dla_dataset/visual_embeddings` folder of the current repository.
    """
    model_path = "models/lrw_resnet18_dctcn_video_boundary.pth"
    config_path = "configs/lrw_resnet18_dctcn_boundary.json"

    args_loaded = load_json(config_path)
    backbone_type = args_loaded["backbone_type"]
    width_mult = args_loaded["width_mult"]
    relu_type = args_loaded["relu_type"]
    use_boundary = False
    tcn_options = {}
    densetcn_options = {
        "block_config": args_loaded["densetcn_block_config"],
        "growth_rate_set": args_loaded["densetcn_growth_rate_set"],
        "reduced_size": args_loaded["densetcn_reduced_size"],
        "kernel_size_set": args_loaded["densetcn_kernel_size_set"],
        "dilation_size_set": args_loaded["densetcn_dilation_size_set"],
        "squeeze_excitation": args_loaded["densetcn_se"],
        "dropout": args_loaded["densetcn_dropout"],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Lipreading(
        modality="video",
        num_classes=500,
        tcn_options=tcn_options,
        densetcn_options=densetcn_options,
        backbone_type=backbone_type,
        relu_type=relu_type,
        width_mult=width_mult,
        use_boundary=use_boundary,
        extract_feats=True,
    ).to(device)
    model = load_model(model_path, model, allow_size_mismatch=True)

    model.eval()

    landmarks_path = "landmarks"
    embeddings_path = "visual_embeddings"
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
        np.savez(embeddings_file_path, embeddings=embeddings.cpu().numpy())


if __name__ == "__main__":
    extract_visual_embeddings()
