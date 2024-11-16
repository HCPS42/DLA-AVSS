import torch
import torch.nn.functional as F
from torch import nn

from src.model.base_model import BaseModel


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


class SuDoRMRFModel(BaseModel):
    """ """

    def __init__(
        self,
        enc_kernel_size,
        latent_size,
        bottleneck_size,
        uconv_hidden_size,
        num_uconv_layers,
        uconv_kernel_size,
        uconv_stride,
        num_blocks,
        num_sources,
    ):
        """ """
        super().__init__()

        self.uconv_stride = uconv_stride
        self.num_uconv_layers = num_uconv_layers
        self.num_sources = num_sources
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2

        self.encoder = nn.Conv1d(
            1, latent_size, enc_kernel_size, bias=False, stride=self.enc_stride
        )
        self.decoder = nn.ConvTranspose1d(
            latent_size, 1, enc_kernel_size, bias=False, stride=self.enc_stride
        )

        self.bottleneck = nn.Sequential(
            nn.GroupNorm(1, latent_size, eps=1e-8),
            nn.Conv1d(latent_size, bottleneck_size, kernel_size=1),
        )

        self.separator = nn.Sequential(
            *[
                UConvBlock(
                    embedding_channels=bottleneck_size,
                    hidden_channels=uconv_hidden_size,
                    num_layers=num_uconv_layers,
                    kernel_size=uconv_kernel_size,
                    stride=uconv_stride,
                )
                for _ in range(num_blocks)
            ],
            nn.PReLU()
        )

        self.masker = nn.Sequential(
            nn.Conv1d(bottleneck_size, num_sources * latent_size, kernel_size=1),
            nn.ReLU(),
        )

        torch.nn.init.xavier_uniform(self.encoder.weight)
        torch.nn.init.xavier_uniform(self.decoder.weight)

    def pad(self, x: torch.Tensor):
        time_steps = x.size(2)
        zero_tail = (
            self.enc_stride
            + self.enc_kernel_size
            - (self.enc_stride + time_steps) % self.enc_kernel_size
        )
        enc_time_steps = (
            time_steps + self.enc_stride + zero_tail - self.enc_kernel_size
        ) // self.enc_stride + 1
        mod = self.uconv_stride ** (self.num_uconv_layers - 1)
        zero_tail += (-enc_time_steps % mod) * self.enc_stride
        x = F.pad(x, (self.enc_stride, zero_tail))
        return x, zero_tail

    def forward(self, mix_wav, **batch):
        """
        Args:
            mix_wav (torch.Tensor): Input tensor representing the mixed waveform.
                Shape: (batch_size, 1, time_steps), tested on time_steps = 16000.

        Returns:
            dict: A dictionary containing the separated waveforms.
                - output_wav (torch.Tensor): Output tensor representing the separated waveforms.
                    Shape: (batch_size, 2, time_steps)
        """
        batch_size = mix_wav.size(0)

        x, zero_tail = self.pad(mix_wav)
        # x: (batch_size, 1, 16090)

        encoded = self.encoder(x)
        # encoded: (batch_size, 512, 1608)

        masks = self.bottleneck(encoded)
        # masks: (batch_size, 128, 1608)

        masks = self.separator(masks)
        # masks: (batch_size, 128, 1608)

        masks = self.masker(masks)
        # masks: (batch_size, 1024, 1608)

        masks = masks.view(batch_size, self.num_sources, -1, encoded.size(2))
        # masks: (batch_size, 2, 512, 1608)

        masked = encoded.unsqueeze(1) * masks
        # masked: (batch_size, 2, 512, 1608)

        decoded = self.decoder(masked.view(batch_size * 2, -1, encoded.size(2)))
        # decoded: (batch_size * 2, 1, 16090)

        decoded = decoded[:, :, self.enc_stride : -zero_tail].view(batch_size, 2, -1)
        # decoded: (batch_size, 2, 16000)

        return {"output_wav": decoded}
