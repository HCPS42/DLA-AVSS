import torch
from torch import nn
from torchaudio import transforms as T


class SpectrogramEncoder(nn.Module):
    def __init__(self, n_fft=1024, n_mels=128, eps=1e-8):
        super().__init__()
        self.transform = T.MelSpectrogram(n_fft=n_fft, n_mels=n_mels)
        self.eps = eps

    def forward(self, wav: torch.Tensor):
        return torch.log(self.transform(wav) + self.eps)
