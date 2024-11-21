import torch
from torch import nn


class NormalizeAudio(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, wav: torch.Tensor):
        """
        Normalize a waveform tensor to have maximum absolute value of 1.

        Args:
            wav (torch.Tensor): The waveform tensor of shape (batch_size, 1, time_steps).

        Returns:
            torch.Tensor: The normalized waveform tensor of shape (batch_size, 1, time_steps).
        """
        return wav / torch.abs(wav).max(dim=-1, keepdim=True).values
