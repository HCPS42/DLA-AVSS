#!/bin/bash

apt update
apt install -y tmux unzip vim

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

mkdir -p data
gdown --fuzzy ${DATASET_LINK} -O data/dla_dataset.zip
unzip -q data/dla_dataset.zip -d data

gdown --fuzzy "https://drive.google.com/file/d/1TGFG0dW5M3rBErgU8i0N7M1ys9YMIvgm/view" -O lipreading/model.pth

python lipreading/extract.py

python scripts/create_index.py --data-dir data/dla_dataset
