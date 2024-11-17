import torch
from torch import nn


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


class UConvBlock(nn.Module):
    def __init__(
        self,
        embedding_channels,
        hidden_channels,
        num_layers,
        kernel_size,
        stride,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.stride = stride

        self.expand = self._get_conv_block(
            embedding_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            depthwise=False,
        )

        self.downsamples = nn.ModuleList(
            [
                self._get_conv_block(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    stride=1,
                    depthwise=True,
                )
            ]
        )

        for _ in range(num_layers - 1):
            self.downsamples.append(
                self._get_conv_block(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=stride * 2 + 1,
                    stride=stride,
                    depthwise=True,
                )
            )

        self.upsample = nn.Upsample(scale_factor=2)

        self.collapse = nn.Sequential(
            nn.GroupNorm(1, hidden_channels, eps=1e-8),
            nn.PReLU(),
            self._get_conv(
                hidden_channels,
                embedding_channels,
                kernel_size=1,
                stride=1,
                depthwise=False,
            ),
        )

    def forward(self, x: torch.Tensor):
        assert x.shape[-1] % self.stride ** (self.num_layers - 1) == 0

        residual = x

        hidden = [self.downsamples[0](self.expand(x))]
        for downsample in self.downsamples[1:]:
            hidden.append(downsample(hidden[-1]))

        out = hidden[-1]
        for state in hidden[:-1][::-1]:
            out = self.upsample(out) + state

        out = self.collapse(out)

        return out + residual

    @staticmethod
    def _get_conv(in_channels, out_channels, kernel_size, stride, depthwise):
        return nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=out_channels if depthwise else 1,
            padding=kernel_size // 2,
        )

    @staticmethod
    def _get_conv_block(in_channels, out_channels, kernel_size, stride, depthwise):
        return nn.Sequential(
            UConvBlock._get_conv(
                in_channels, out_channels, kernel_size, stride, depthwise
            ),
            nn.GroupNorm(1, out_channels, eps=1e-8),
            nn.PReLU(),
        )


class StackedUConvBlock(nn.Module):
    def __init__(
        self,
        embedding_channels,
        hidden_channels,
        num_layers,
        kernel_size,
        stride,
        num_blocks,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                UConvBlock(
                    embedding_channels=embedding_channels,
                    hidden_channels=hidden_channels,
                    num_layers=num_layers,
                    kernel_size=kernel_size,
                    stride=stride,
                )
                for _ in range(num_blocks)
            ],
            nn.PReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.blocks(x)
