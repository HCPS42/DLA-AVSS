#!/bin/bash

apt update
apt install -y tmux unzip

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

wandb login

gdown --fuzzy ${DATASET_LINK}
mv dla_dataset.zip data
unzip data/dla_dataset.zip

gdown --fuzzy ${EMBED_LINK}
mv visual_embeddings.zip data/dla_dataset
unzip data/dla_dataset/visual_embeddings.zip

python scripts/create_index.py --data-dir data/dla_dataset
