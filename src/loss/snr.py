import torch
from torch import nn


class SNRLoss(nn.Module):
    """
    Signal to noise ratio a a loss function
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        output_wav: torch.Tensor,
        speaker_1_wav: torch.Tensor,
        speaker_2_wav: torch.Tensor,
        **batch
    ):
        """
        Calculate the signal-to-noise ratio (SNR) loss for a given batch.

        The loss is computed as the negative mean of the SNR for two permutations
        of the output and reference waveforms. The permutation that yields the
        lower SNR is chosen.

        Args:
            output_wav (torch.Tensor): The output waveform of shape (B x 2 x L).
            speaker_1_wav (torch.Tensor): The reference waveform for speaker 1 of shape (B x 1 x L).
            speaker_2_wav (torch.Tensor): The reference waveform for speaker 2 of shape (B x 1 x L).

        Returns:
            dict: A dictionary containing the computed loss.
        """
        loss1 = self.calculate(
            output_wav[:, 0], speaker_1_wav.squeeze(1)
        ) + self.calculate(output_wav[:, 1], speaker_2_wav.squeeze(1))

        loss2 = self.calculate(
            output_wav[:, 0], speaker_2_wav.squeeze(1)
        ) + self.calculate(output_wav[:, 1], speaker_1_wav.squeeze(1))

        return {"loss": torch.mean(torch.minimum(loss1, loss2))}

    def calculate(self, output_wav: torch.Tensor, target_wav: torch.Tensor):
        """
        Calculate the signal-to-noise ratio (SNR) between the output and target waveforms.

        Args:
            output_wav (torch.Tensor): The output waveform of shape (B x L).
            target_wav (torch.Tensor): The target (reference) waveform of shape (B x L).

        Returns:
            torch.Tensor: minus SNR values for each example in the batch (i.e. lower is better).
        """

        def dot(x, y, axis):
            return torch.sum(x * y, axis=axis).unsqueeze(axis)

        def norm(x, axis):
            return torch.sum(x**2, axis=axis).unsqueeze(axis)

        signal = (
            dot(output_wav, target_wav, axis=1) * target_wav / norm(target_wav, axis=1)
        )
        noise = output_wav - signal

        snr = 10 * torch.log10(norm(signal, axis=1) / norm(noise, axis=1))
        return -snr
