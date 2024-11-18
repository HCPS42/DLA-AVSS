import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.base_model import BaseAudioModel, BaseSeparatorModel
from src.model.utils import TemporalConvNet


class Separator(BaseSeparatorModel):
    def __init__(
        self,
        num_filters,
        bottleneck_channels,
        conv_num_channels,
        conv_kernel_size,
        num_conv_blocks,
        num_tcn_blocks,
    ):
        """
        Args:
            num_filters (int): Number of filters in the encoder and decoder.
            bottleneck_channels (int): Number of channels in the bottleneck.
            conv_num_channels (int): Number of channels in the convolutional blocks.
            conv_kernel_size (int): Kernel size in the convolutional blocks.
            num_conv_blocks (int): Number of convolutional blocks in each TCN block.
            num_audio_tcn_blocks (int): Number of audio TCN blocks.
            num_fusion_tcn_blocks (int): Number of fusion TCN blocks.
            visual_emb_dim (int): Dimension of the visual embeddings.
        """
        super().__init__()
        self.embedding_size = num_filters

        self.tcn = TemporalConvNet(
            num_filters,
            num_filters * 2,
            bottleneck_channels,
            conv_num_channels,
            num_conv_blocks,
            num_tcn_blocks,
            conv_kernel_size,
        )

    def forward(self, audio_features: torch.Tensor, mix_visual: torch.Tensor):
        return self.tcn(audio_features)


class ConvTasNetModel(BaseAudioModel):
    """
    Audio-Visual Conv-TasNet model.
    Papers: https://arxiv.org/abs/1809.07454, https://arxiv.org/abs/1904.03760
    This implementation is based on the ConvTasNetModel.
    """

    def __init__(
        self,
        num_filters,
        filter_length,
        bottleneck_channels,
        conv_num_channels,
        conv_kernel_size,
        num_conv_blocks,
        num_tcn_blocks,
    ):
        """
        Args:
            num_filters (int): Number of filters in the encoder and decoder. (N)
            filter_length (int): Length of the filters in time steps. (L)
            bottleneck_channels (int): Number of channels in the bottleneck. (B)
            conv_num_channels (int): Number of channels in the convolutional blocks. (H)
            conv_kernel_size (int): Kernel size in the convolutional blocks. (P)
            num_conv_blocks (int): Number of convolutional blocks in each TCN block. (D)
            num_audio_tcn_blocks (int): Number of audio TCN blocks. (N_a)
            num_fusion_tcn_blocks (int): Number of fusion TCN blocks. (N_f)
            visual_emb_dim (int): Dimension of the visual embeddings.
        """
        super().__init__()

        self.filter_length = filter_length
        self.stride = filter_length // 2

        self.encoder = nn.Conv1d(
            1, num_filters, self.filter_length, bias=False, stride=self.stride
        )
        self.decoder = nn.ConvTranspose1d(
            num_filters, 1, self.filter_length, bias=False, stride=self.stride
        )

        self.separator = Separator(
            num_filters,
            bottleneck_channels,
            conv_num_channels,
            conv_kernel_size,
            num_conv_blocks,
            num_tcn_blocks,
        )

        self.relu = nn.ReLU()

    def pad(self, x):
        time_steps = x.size(2)
        zero_tail = (
            self.stride
            + self.filter_length
            - (self.stride + time_steps) % self.filter_length
        )
        x = F.pad(x, (self.stride, zero_tail))
        return x, zero_tail

    def forward(self, mix_wav, mix_visual, **batch):
        """
        Args:
            mix_wav (torch.Tensor): Input tensor representing the mixed waveform.
                Shape: (batch_size, 1, time_steps), tested on time_steps = 32000.
            mix_visual (torch.Tensor): Input tensor representing the stacked visual features.
                Shape: (batch_size, 2, num_frames, emb_dim), tested on num_frames = 50, emb_dim = 512.

        Returns:
            dict: A dictionary containing the separated waveforms.
                - output_wav (torch.Tensor): Output tensor representing the separated waveforms.
                    Shape: (batch_size, 2, time_steps)
        """
        batch_size = mix_wav.size(0)

        mix_wav, zero_tail = self.pad(mix_wav)
        # mix_wav: (batch_size, 1, padded_time_steps)

        encoded = self.encoder(mix_wav)
        # encoded: (batch_size, num_filters, latent_time_steps)

        masks = self.separator(encoded, mix_visual)
        # masks: (batch_size, num_filters * 2, latent_time_steps)

        masks = self.relu(masks)
        masks = masks.view(batch_size, 2, -1, encoded.size(2))
        # masks: (batch_size, 2, num_filters, latent_time_steps)

        masked = encoded.unsqueeze(1) * masks
        # masked: (batch_size, 2, num_filters, latent_time_steps)

        masked = masked.view(batch_size * 2, -1, encoded.size(-1))
        # masked: (batch_size * 2, num_filters, latent_time_steps)

        decoded = self.decoder(masked)
        # decoded: (batch_size * 2, 1, padded_time_steps)

        decoded = decoded[:, :, self.stride : -zero_tail]
        # decoded: (batch_size * 2, 1, time_steps)

        decoded = decoded.view(batch_size, 2, -1)
        # decoded: (batch_size, 2, time_steps)

        return {"output_wav": decoded}
