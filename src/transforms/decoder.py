import torch
import torchaudio.transforms as T
from torch import nn


class SpectrogramDecoder(nn.Module):
    def __init__(self, n_fft=1024, n_mels=128, eps=1e-8):
        super().__init__()
        self.inv_mel = T.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels)
        self.inv_spec = T.InverseSpectrogram(n_fft=n_fft)
        self.raw_spec = T.Spectrogram(n_fft=n_fft, power=None)
        self.eps = eps

    def forward(self, mix_wav: torch.Tensor, output_spec: torch.Tensor, **batch):
        mel_spec = torch.exp(output_spec.transpose(1, 2).detach().cpu()) - self.eps
        spec = self.inv_mel(mel_spec)

        mix_spec = self.raw_spec(mix_wav)
        # normalize complex numbers to have magnitude 1
        mix_spec /= torch.abs(mix_spec)
        # get magnitude from estimated spectrogram
        mix_spec *= spec

        output_wav = self.inv_spec(mix_spec)
        return {"output_wav": output_wav}
