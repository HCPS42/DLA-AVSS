import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.model.base_model import BaseModel


class DepthConv1d(nn.Module):
    """
    Depthwise convolutional module with optional skip connection.

    Args:
        input_channel (int): Number of input channels.
        hidden_channel (int): Number of hidden channels.
        kernel (int): Kernel size for depthwise convolution.
        padding (int): Padding size for depthwise convolution.
        dilation (int): Dilation rate for depthwise convolution. Default is 1.
        skip (bool): If True, includes skip connection. Default is True.
    """

    def __init__(
        self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True
    ):
        super().__init__()

        self.skip = skip
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, kernel_size=1)
        self.dconv1d = nn.Conv1d(
            hidden_channel,
            hidden_channel,
            kernel_size=kernel,
            dilation=dilation,
            groups=hidden_channel,
            padding=padding,
        )
        self.res_out = nn.Conv1d(hidden_channel, input_channel, kernel_size=1)
        self.skip_out = (
            nn.Conv1d(hidden_channel, input_channel, kernel_size=1) if skip else None
        )

        self.nonlinearity = nn.PReLU()
        self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-8)
        self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-8)

    def forward(self, x):
        x = self.reg1(self.nonlinearity(self.conv1d(x)))
        output = self.reg2(self.nonlinearity(self.dconv1d(x)))

        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        return residual


class TCN(nn.Module):
    """
    Temporal Convolutional Network with optional skip connections.

    Args:
        input_dim (int): Input channel dimension.
        output_dim (int): Output channel dimension.
        bn_dim (int): Bottleneck layer dimension.
        hidden_dim (int): Hidden layer dimension.
        layer (int): Number of layers in each stack.
        stack (int): Number of stacks.
        kernel (int): Kernel size for convolutional layers. Default is 3.
        skip (bool): If True, uses skip connections. Default is True.
        dilated (bool): If True, uses dilated convolutions. Default is True.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        bn_dim,
        hidden_dim,
        layer,
        stack,
        kernel=3,
        skip=True,
        dilated=True,
    ):
        super().__init__()

        self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        self.BN = nn.Conv1d(input_dim, bn_dim, kernel_size=1)
        self.skip = skip
        self.dilated = dilated
        self.receptive_field = 0

        self.TCN = nn.ModuleList()
        for s in range(stack):
            for i in range(layer):
                dilation = 2**i if dilated else 1
                padding = dilation
                self.TCN.append(
                    DepthConv1d(
                        bn_dim,
                        hidden_dim,
                        kernel,
                        dilation=dilation,
                        padding=padding,
                        skip=skip,
                    )
                )

                self.receptive_field += (kernel - 1) * dilation if s or i else kernel

        self.output = nn.Sequential(
            nn.PReLU(), nn.Conv1d(bn_dim, output_dim, kernel_size=1)
        )

    def forward(self, x):
        x = self.BN(self.LN(x))

        if self.skip:
            skip_connection = 0
            for layer in self.TCN:
                residual, skip = layer(x)
                x = x + residual
                skip_connection = skip_connection + skip
            x = self.output(skip_connection)
        else:
            for layer in self.TCN:
                residual = layer(x)
                x = x + residual
            x = self.output(x)

        return x


class AVConvTasNetModel(BaseModel):
    """
    Audio-Visual Conv-TasNet Model for source separation.

    Args:
        num_filters (int): Number of filters in the encoder and decoder. (N in the paper)
        filter_length (int): Length of the filters in samples in milliseconds. (L in the paper)
        bottleneck_channels (int): Number of channels in the bottleneck. (B in the paper)
        conv_kernel_size (int): Kernel size in the convolutional blocks. (P in the paper)
        num_conv_blocks (int): Number of convolutional blocks in each repeat. (X in the paper)
        num_repeats (int): Number of repeats. (R in the paper)
    """

    def __init__(
        self,
        num_filters,
        filter_length,
        bottleneck_channels,
        conv_kernel_size,
        num_conv_blocks,
        num_repeats,
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

        self.TCN = TCN(
            num_filters,
            num_filters * self.num_speakers,
            bottleneck_channels,
            bottleneck_channels * 4,
            num_conv_blocks,
            num_repeats,
            conv_kernel_size,
        )
        self.receptive_field = self.TCN.receptive_field

    def pad_signal(self, mix_wav):
        if mix_wav.dim() == 2:
            mix_wav = mix_wav.unsqueeze(1)

        batch_size, _, nsample = mix_wav.size()

        remainder = (self.stride + nsample % self.filter_length) % self.filter_length
        rest = (self.filter_length - remainder) % self.filter_length

        pad = torch.zeros(
            batch_size,
            1,
            rest + 2 * self.stride,
            device=mix_wav.device,
            dtype=mix_wav.dtype,
        )
        mix_wav = torch.cat(
            [pad[:, :, : self.stride], mix_wav, pad[:, :, self.stride :]], dim=2
        )

        return mix_wav, rest

    def forward(self, mix_wav: torch.Tensor, **batch):
        output, rest = self.pad_signal(mix_wav)
        batch_size = output.size(0)

        enc_output = self.encoder(output)
        masks = torch.sigmoid(self.TCN(enc_output)).view(
            batch_size, self.num_speakers, -1, enc_output.size(2)
        )
        masked_output = enc_output.unsqueeze(1) * masks

        output = self.decoder(
            masked_output.view(batch_size * self.num_speakers, -1, enc_output.size(2))
        )
        output = (
            output[:, :, self.stride : -(rest + self.stride)]
            .contiguous()
            .view(batch_size, self.num_speakers, -1)
        )

        return {"output_wav": output}
