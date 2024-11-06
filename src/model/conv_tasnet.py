import torch
from torch import nn
from torchaudio.models import ConvTasNet

from src.model.base_model import BaseModel


class ConvTasNetModel(BaseModel, ConvTasNet):
    def forward(self, mix_wav: torch.Tensor, **batch):
        """
        Model forward method.

        Args:
            mix_spec (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        return {"output_wav": super().forward(mix_wav)}
