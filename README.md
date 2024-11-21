# Stacking More Layers is All You Need in Speech Separation

### Assignment

This repository holds code for the 'Deep Learning in Audio' course assignment at the Higher School of Economics. The goal of the assignment is to implement and evaluate audio-only and audio-visual speech separation models. The full assignment description can be found [here](https://github.com/markovka17/dla/blob/2024/project_avss/README.md). The contributing team members are: [Danil Sheshenya](https://github.com/bigshishiga0000xDxD), [Temirkhan Zimanov](https://github.com/HCPS42).

### Setup

Set up the environment, download the dataset, and extract visual embeddings by running the following command:

```bash
DATASET_LINK=<WRITE_YOUR_LINK_HERE> ./scripts/init.sh
```

```DATASET_LINK``` is expected to be a link to a dataset that complies with the assignment requirements and is stored on Google Drive.

### Training

To train a model, run one of the following commands:

```bash
python train.py --config-name conv-tasnet-train

python train.py --config-name av-conv-tasnet-train

python train.py --config-name sudo-rmrf-train

python train.py --config-name av-sudo-rmrf-train
```

### Inference

TODO: link to model weights and where to put them

To separate speech from a mixture, run one of the following commands:

```bash
python inference.py --config-name conv-tasnet-train

python inference.py --config-name av-conv-tasnet-train

python train.py --config-name sudo-rmrf-train

python train.py --config-name av-sudo-rmrf-train
```

### Evaluation
