import torch
import torch.nn.functional as F
from torch import nn

from src.model.base_model import BaseModel


class DepthwiseConv1d(nn.Module):
    """
    Depthwise convolutional module. (1-D Conv block in the paper)
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size for depthwise convolution.
            padding (int): Padding size for depthwise convolution.
            dilation (int): Dilation rate for depthwise convolution. Default is 1.
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
        self.norm1 = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.norm2 = nn.GroupNorm(1, out_channels, eps=1e-8)

    def forward(self, x):
        x = self.norm1(self.prelu(self.conv(x)))
        x = self.norm2(self.prelu(self.depth_conv(x)))

        residual = self.residual_conv(x)
        skip = self.skip_conv(x)

        return residual, skip


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
        dilated=True,
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
            dilated (bool): If True, uses dilated convolutions. Default is True.
        """
        super().__init__()

        self.norm = nn.GroupNorm(1, in_channels, eps=1e-8)
        self.bottleneck_conv = nn.Conv1d(
            in_channels, bottleneck_channels, kernel_size=1
        )

        self.tcn_blocks = nn.ModuleList()
        for _ in range(num_tcn_blocks):
            for i in range(num_conv_blocks):
                dilation = 2**i if dilated else 1
                padding = dilation
                block = DepthwiseConv1d(
                    bottleneck_channels,
                    hidden_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
                self.tcn_blocks.append(block)

        self.prelu = nn.PReLU()
        self.output_conv = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.bottleneck_conv(self.norm(x))

        skip_sum = 0
        for layer in self.tcn_blocks:
            residual, skip = layer(x)
            x = x + residual
            skip_sum = skip_sum + skip

        x = self.output_conv(self.prelu(skip_sum))

        return x


class ConvTasNetModel(BaseModel):
    """
    Conv-TasNet Model.
    Paper: https://arxiv.org/abs/1809.07454
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
            num_filters (int): Number of filters in the encoder and decoder. (N in the paper)
            filter_length (int): Length of the filters in time steps. (L in the paper)
            bottleneck_channels (int): Number of channels in the bottleneck. (B in the paper)
            conv_num_channels (int): Number of channels in the convolutional blocks. (H in the paper)
            conv_kernel_size (int): Kernel size in the convolutional blocks. (P in the paper)
            num_conv_blocks (int): Number of convolutional blocks in each TCN block. (X in the paper)
            num_tcn_blocks (int): Number of TCN blocks. (R in the paper)
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

        self.tcn = TemporalConvNet(
            num_filters,
            num_filters * 2,
            bottleneck_channels,
            conv_num_channels,
            num_conv_blocks,
            num_tcn_blocks,
            conv_kernel_size,
        )

    def pad(self, x):
        time_steps = x.size(2)
        zero_tail = (
            self.stride
            + self.filter_length
            - (self.stride + time_steps) % self.filter_length
        )
        x = F.pad(x, (self.stride, zero_tail))
        return x, zero_tail

    def forward(self, mix_wav, **batch):
        """
        Args:
            mix_wav (torch.Tensor): Input tensor representing the mixed waveform.
                Shape: (batch_size, 1, time_steps), tested on time_steps = 32000.

        Returns:
            dict: A dictionary containing the separated waveforms.
                - output_wav (torch.Tensor): Output tensor representing the separated waveforms.
                    Shape: (batch_size, 2, time_steps)
        """
        batch_size = mix_wav.size(0)

        x, zero_tail = self.pad(mix_wav)
        # x: (batch_size, 1, 32048)

        encoded = self.encoder(x)
        # encoded: (batch_size, 512, 2002)

        masks = self.tcn(encoded)
        # masks: (batch_size, 1024, 2002)

        masks = torch.sigmoid(masks).view(batch_size, 2, -1, encoded.size(2))
        # masks: (batch_size, 2, 512, 2002)

        masked = encoded.unsqueeze(1) * masks
        # masked: (batch_size, 2, 512, 2002)

        decoded = self.decoder(masked.view(batch_size * 2, -1, encoded.size(2)))
        # decoded: (batch_size * 2, 1, 32048)

        decoded = decoded[:, :, self.stride : -zero_tail].view(batch_size, 2, -1)
        # decoded: (batch_size, 2, 32000)

        return {"output_wav": decoded}
