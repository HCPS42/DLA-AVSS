#!/bin/bash

apt update
apt install -y tmux unzip vim

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

wandb login

mkdir data
gdown --fuzzy ${DATASET_LINK}
unzip dla_dataset.zip
mv dla_dataset data

gdown --fuzzy ${EMBED_LINK}
unzip visual_embeddings.zip
mv visual_embeddings data/dla_dataset

python scripts/create_index.py --data-dir data/dla_dataset
