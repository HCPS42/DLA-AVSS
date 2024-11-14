import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.base_model import BaseModel


class DepthwiseConv1d(nn.Module):
    """
    Depthwise convolutional module. (1-D Conv block in the paper)
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size for depthwise convolution.
            padding (int): Padding size for depthwise convolution.
            dilation (int): Dilation rate for depthwise convolution.
        """
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.depth_conv = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=out_channels,
            padding=padding,
        )
        self.residual_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(
            out_channels, in_channels, kernel_size=1
        )  # S_c in the paper

        self.prelu = nn.PReLU()
        self.norm_1 = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.norm_2 = nn.GroupNorm(1, out_channels, eps=1e-8)

    def forward(self, x):
        x = self.norm_1(self.prelu(self.conv(x)))
        x = self.norm_2(self.prelu(self.depth_conv(x)))

        residual = self.residual_conv(x)
        skip = self.skip_conv(x)

        return residual, skip


class VisualConv1D(nn.Module):
    """
    Visual Convolutional module. (Conv1D_v in the paper)
    """

    def __init__(self, emb_dim):
        """
        Args:
            emb_dim (int): Dimension of the visual embeddings.
        """
        super().__init__()

        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(emb_dim)
        self.depth_conv = nn.Conv1d(
            emb_dim,
            emb_dim,
            kernel_size=3,
            padding=1,
            groups=emb_dim,
        )

    def forward(self, x):
        output = self.depth_conv(self.norm(self.relu(x)))

        return output + x


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        hidden_channels,
        num_conv_blocks,
        num_tcn_blocks,
        kernel_size,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): Number of channels in the bottleneck.
            hidden_channels (int): Number of channels in the convolutional blocks.
            num_conv_blocks (int): Number of convolutional blocks in each TCN block.
            num_tcn_blocks (int): Number of TCN blocks.
            kernel_size (int): Kernel size for convolutional layers.
        """
        super().__init__()

        self.norm = nn.GroupNorm(1, in_channels, eps=1e-8)
        self.bottleneck_conv = nn.Conv1d(
            in_channels, bottleneck_channels, kernel_size=1
        )

        self.tcn_blocks = nn.ModuleList()
        for _ in range(num_tcn_blocks):
            for i in range(num_conv_blocks):
                dilation = 2**i
                block = DepthwiseConv1d(
                    bottleneck_channels,
                    hidden_channels,
                    kernel_size,
                    padding=dilation,
                    dilation=dilation,
                )
                self.tcn_blocks.append(block)

        self.prelu = nn.PReLU()
        self.output_conv = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.bottleneck_conv(self.norm(x))

        skip_sum = 0
        for block in self.tcn_blocks:
            residual, skip = block(x)
            x = x + residual
            skip_sum = skip_sum + skip

        x = self.output_conv(self.prelu(skip_sum))

        return x


class SeparationNet(nn.Module):
    """
    Audio-Visual Separation Network.
    """

    def __init__(
        self,
        num_filters,
        bottleneck_channels,
        conv_num_channels,
        conv_kernel_size,
        num_conv_blocks,
        num_audio_tcn_blocks,
        num_fusion_tcn_blocks,
        visual_emb_dim,
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

        self.tcn_audio = TemporalConvNet(
            num_filters,
            num_filters,
            bottleneck_channels,
            conv_num_channels,
            num_conv_blocks,
            num_audio_tcn_blocks,
            conv_kernel_size,
        )

        blocks = [VisualConv1D(visual_emb_dim) for _ in range(5)]
        self.visual_conv = nn.Sequential(*blocks)

        self.projection = nn.Conv1d(
            num_filters + visual_emb_dim * 2, num_filters, kernel_size=1
        )

        self.tcn_fusion = TemporalConvNet(
            num_filters,
            num_filters * 2,
            bottleneck_channels,
            conv_num_channels,
            num_conv_blocks,
            num_fusion_tcn_blocks,
            conv_kernel_size,
        )

    def forward(self, audio, visual):
        batch_size = audio.size(0)

        audio = self.tcn_audio(audio)
        # audio: (batch_size, num_filters, latent_time_steps)

        emb_dim = visual.size(-1)
        visual = visual.view(batch_size * 2, -1, emb_dim)
        visual = visual.transpose(1, 2)
        visual = self.visual_conv(visual)
        visual = F.interpolate(visual, size=audio.size(-1), mode="linear")  # upsampling
        visual = visual.view(batch_size, 2, emb_dim, audio.size(-1))
        visual = visual.view(batch_size, -1, audio.size(-1))
        # visual: (batch_size, emb_dim * 2, latent_time_steps)

        fused = torch.cat((audio, visual), dim=1)
        fused = self.projection(fused)
        # fused: (batch_size, num_filters, latent_time_steps)

        masks = self.tcn_fusion(fused)
        # masks: (batch_size, num_filters * 2, latent_time_steps)

        return masks


class AVConvTasNetModel(BaseModel):
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
        num_audio_tcn_blocks,
        num_fusion_tcn_blocks,
        visual_emb_dim,
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

        self.separator = SeparationNet(
            num_filters,
            bottleneck_channels,
            conv_num_channels,
            conv_kernel_size,
            num_conv_blocks,
            num_audio_tcn_blocks,
            num_fusion_tcn_blocks,
            visual_emb_dim,
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
