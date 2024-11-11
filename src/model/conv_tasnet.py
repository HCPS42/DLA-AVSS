import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.model.base_model import BaseModel


class DepthConv1d(nn.Module):
    """
    Depthwise convolutional module.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for depthwise convolution.
        padding (int): Padding size for depthwise convolution.
        dilation (int): Dilation rate for depthwise convolution. Default is 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
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
        self.skip_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)

        self.prelu = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.norm2 = nn.GroupNorm(1, out_channels, eps=1e-8)

    def forward(self, x):
        x = self.norm1(self.prelu(self.conv(x)))
        x = self.norm2(self.prelu(self.depth_conv(x)))

        residual = self.residual_conv(x)
        skip = self.skip_conv(x)

        return residual, skip


class TCN(nn.Module):
    """
    Temporal Convolutional Network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bottleneck_channels (int): Number of channels in the bottleneck.
        hidden_channels (int): Number of hidden channels.
        num_conv_blocks (int): Number of convolutional blocks in each TCN block.
        num_tcn_blocks (int): Number of TCN blocks.
        kernel_size (int): Kernel size for convolutional layers.
        dilated (bool): If True, uses dilated convolutions. Default is True.
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
        dilated=True,
    ):
        super().__init__()

        self.input_norm = nn.GroupNorm(1, in_channels, eps=1e-8)
        self.bottleneck_conv = nn.Conv1d(
            in_channels, bottleneck_channels, kernel_size=1
        )

        self.tcn_blocks = nn.ModuleList()
        for _ in range(num_tcn_blocks):
            for i in range(num_conv_blocks):
                dilation = 2**i if dilated else 1
                padding = dilation
                self.tcn_blocks.append(
                    DepthConv1d(
                        bottleneck_channels,
                        hidden_channels,
                        kernel_size,
                        dilation=dilation,
                        padding=padding,
                    )
                )

        self.output_conv = nn.Sequential(
            nn.PReLU(), nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.bottleneck_conv(self.input_norm(x))

        skip_sum = 0
        for layer in self.tcn_blocks:
            residual, skip = layer(x)
            x = x + residual
            skip_sum = skip_sum + skip

        x = self.output_conv(skip_sum)

        return x


class ConvTasNetModel(BaseModel):
    """
    Audio-Visual Conv-TasNet Model for source separation.

    Args:
        num_filters (int): Number of filters in the encoder and decoder. (N in the paper)
        filter_length (int): Length of the filters in samples in milliseconds. (L in the paper)
        bottleneck_channels (int): Number of channels in the bottleneck. (B in the paper)
        conv_kernel_size (int): Kernel size in the convolutional blocks. (P in the paper)
        num_conv_blocks (int): Number of convolutional blocks in each TCN block. (X in the paper)
        num_tcn_blocks (int): Number of TCN blocks. (R in the paper)
    """

    def __init__(
        self,
        num_filters,
        filter_length,
        bottleneck_channels,
        conv_kernel_size,
        num_conv_blocks,
        num_tcn_blocks,
    ):
        super().__init__()

        self.num_speakers = 2
        self.sr = 16000

        self.filter_length = int(self.sr * filter_length / 1000)
        self.stride = self.filter_length // 2

        self.encoder = nn.Conv1d(
            1, num_filters, self.filter_length, bias=False, stride=self.stride
        )
        self.decoder = nn.ConvTranspose1d(
            num_filters, 1, self.filter_length, bias=False, stride=self.stride
        )

        self.tcn = TCN(
            num_filters,
            num_filters * self.num_speakers,
            bottleneck_channels,
            bottleneck_channels * 4,
            num_conv_blocks,
            num_tcn_blocks,
            conv_kernel_size,
        )

    def pad_signal(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, _, nsample = x.size()

        remainder = (self.stride + nsample % self.filter_length) % self.filter_length
        rest = (self.filter_length - remainder) % self.filter_length

        pad = torch.zeros(
            batch_size,
            1,
            rest + 2 * self.stride,
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([pad[:, :, : self.stride], x, pad[:, :, self.stride :]], dim=2)

        return x, rest

    def forward(self, mix_wav: torch.Tensor, **batch):
        x, rest = self.pad_signal(mix_wav)
        batch_size = x.size(0)

        encoded = self.encoder(x)

        masks = torch.sigmoid(self.tcn(encoded)).view(
            batch_size, self.num_speakers, -1, encoded.size(2)
        )
        masked = encoded.unsqueeze(1) * masks

        output = self.decoder(
            masked.view(batch_size * self.num_speakers, -1, encoded.size(2))
        )
        output = (
            output[:, :, self.stride : -(rest + self.stride)]
            .contiguous()
            .view(batch_size, self.num_speakers, -1)
        )

        print(output.shape)
        assert 0

        return {"output_wav": output}
