import torch
from torch import nn
from torch.nn import functional as F

from src.model.base_model import BaseVisualModel
from src.model.utils import VisualConv1D


class ConvTasNetVisualModel(BaseVisualModel):
    def __init__(self, embedding_size, num_layers):
        super().__init__()
        self.embedding_size = embedding_size

        blocks = [VisualConv1D(embedding_size) for _ in range(num_layers)]
        self.visual_conv = nn.Sequential(*blocks)

    def forward(self, mix_visual: torch.Tensor, audio_len: int):
        batch_size = mix_visual.size(0)

        visual_features = mix_visual.view(batch_size * 2, -1, self.embedding_size)
        visual_features = visual_features.transpose(1, 2)
        visual_features = self.visual_conv(visual_features)
        # upsampling
        visual_features = F.interpolate(visual_features, size=audio_len, mode="linear")
        visual_features = visual_features.view(
            batch_size, 2, self.embedding_size, audio_len
        )
        visual_features = visual_features.view(batch_size, -1, audio_len)
        # visual_features: (batch_size, emb_dim * 2, latent_time_steps)

        return visual_features
