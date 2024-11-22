# Stacking More Layers is All You Need in Speech Separation

### Assignment

This repository holds code for the 'Deep Learning for Audio' course assignment at the Higher School of Economics. The goal of the assignment is to implement and evaluate audio-only and audio-visual speech separation models. The full assignment description can be found [here](https://github.com/markovka17/dla/blob/2024/project_avss/README.md). The contributing team members are: [Danil Sheshenya](https://github.com/bigshishiga0000xDxD), [Temirkhan Zimanov](https://github.com/HCPS42).

### Setup

* Clone this repository

* Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

* Install dependencies
```bash
pip install -r requirements.txt
```

* Run the following commands to download and extract the training dataset
```bash
mkdir -p data
gdown --fuzzy ${DATASET_LINK} -O data/dla_dataset.zip
unzip -q data/dla_dataset.zip -d data
```
`DATASET_LINK` is supposed to be a link to the training dataset supplied with the assignment and stored on Google Drive.

After that, the dataset should be stored in the `data/dla_dataset` directory.

* Extract visual embeddings

```bash
gdown --fuzzy "https://drive.google.com/file/d/1TGFG0dW5M3rBErgU8i0N7M1ys9YMIvgm/view" -O lipreading/model.pth
```

If you have already downloaded the training dataset, the following should be enough
```bash
python lipreading/extract.py
```

If you have your own dataset, run
```bash
python lipreading/extract.py --mouths-path <PATH_TO_MOUTH_FILES> --embeddings-path <PATH_TO_EMBEDDINGS>
```

Embeddings are expected to be in the same directory as the mouth files (i.e. the dataset folder should contain `audio` and `visual_embeddings` subfolders).

* If you are willing to train, you should run

```bash
python scripts/create_index.py --data-dir data/dla_dataset
```

If you are just running a pretrained model, omit this step.

### Training

To train a model, run one of the following commands:

```bash
python train.py --config-name conv-tasnet-train

python train.py --config-name av-conv-tasnet-train

python train.py --config-name sudo-rmrf-train

python train.py --config-name av-sudo-rmrf-train
```

Each of these commands will train a model with the specified configuration. The configuration files are stored in the `src/configs` directory. The logs and model weights will be stored in the `outputs` and `saved` directories, respectively.

### Inference

Download the best pretrained model with

```bash
gdown --fuzzy "https://drive.google.com/file/d/11rcJvOdtnwKz2wiGhiGCXNUarXwVvcZU/view?usp=sharing"
```

To separate mixtures in the test set (`<DATA_DIR>/audio/mix`), run the following command:

```bash
python inference.py datasets.test.dir=<DATA_DIR> inferencer.save_path=<PATH_TO_SAVE_DIR> inferencer.from_pretrained=<PATH_TO_MODEL_WEIGHTS>
```

Results will be saved to `data/saved/<PATH_TO_SAVE_DIR>/s1` and `data/saved/<PATH_TO_SAVE_DIR>/s2` directories with filenames corresponding to the names of the files in `<DATA_DIR>/audio/mix`.

### Evaluation

To calculate metrics on the test set (supposing that the ground truth is stored in `<DATA_DIR>/audio/s1`, `<DATA_DIR>/audio/s2`), run the following command:

```bash
python inference.py --config-name validate datasets.test.dir=<DATA_DIR> inferencer.from_pretrained=<PATH_TO_MODEL_WEIGHTS>
```
