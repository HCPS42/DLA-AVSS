import torch
from torch import nn


class LogStable(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input_tensor: torch.Tensor):
        return torch.log(input_tensor + self.eps)
