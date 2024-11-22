# Stacking More Layers is All You Need in Speech Separation

### Assignment

This repository holds code for the 'Deep Learning for Audio' course assignment at the Higher School of Economics. The goal of the assignment is to implement and evaluate audio-only and audio-visual speech separation models. The full assignment description can be found [here](https://github.com/markovka17/dla/blob/2024/project_avss/README.md). The contributing team members are: [Danil Sheshenya](https://github.com/bigshishiga0000xDxD), [Temirkhan Zimanov](https://github.com/HCPS42).

### Setup

Set up the environment, download a dataset, and extract visual embeddings by running the following command from the root directory of the repository:

```bash
DATASET_LINK=<WRITE_YOUR_LINK_HERE> ./scripts/init.sh
```

`DATASET_LINK` is expected to be a link to a dataset that complies with the assignment requirements and is stored on Google Drive. The dataset will be stored in the `data/dla_dataset` directory. The extracted visual embeddings will be stored in the `data/dla_dataset/visual_embeddings` directory.

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

TODO: link to model weights and where to put them

To separate mixtures in the test set (`data/dla_dataset/audio/test/mix`), run one of the following commands:

```bash
python inference.py --config-name av-sudo-rmrf-train
```

### Evaluation
