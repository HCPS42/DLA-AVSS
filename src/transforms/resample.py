import torch
import torchaudio.transforms as T
from torch import nn


class Resample(nn.Module):
    def __init__(self, orig_freq, new_freq):
        super().__init__()

        self.resample = T.Resample(orig_freq, new_freq)
        self.speed = T.Speed(new_freq, new_freq / orig_freq)

    def forward(self, x):
        out = self.resample(x)
        return self.speed(out)[0]
