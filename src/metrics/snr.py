import torch

from src.metrics.base_metric import BaseMetric


class SNRMetric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        """
        Args:
            metric (Callable): instance of torchmetrics.audio.SignalNoiseRatio or
                torchmetrics.audio.ScaleInvariantSignalNoiseRatio
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(
        self,
        output_wav: torch.Tensor,
        speaker_1_wav: torch.Tensor,
        speaker_2_wav: torch.Tensor,
        **batch
    ):
        """
        Calculate permutation-invariant signal-to-noise ratio (SNR).

        Args:
            output_wav (torch.Tensor): The output waveform of shape (B x 2 x L).
            speaker_1_wav (torch.Tensor): The reference waveform for speaker 1 of shape (B x 1 x L)
            speaker_2_wav (torch.Tensor): The reference waveform for speaker 2 of shape (B x 1 x L)

        Returns:
            float: The best SNR value between two permutations of speakers.
        """
        # we have to use for loop since we want to get unreduced values for batch
        perm1, perm2 = [], []
        for output, speaker_1, speaker_2 in zip(
            output_wav, speaker_1_wav, speaker_2_wav
        ):
            value1 = self.calculate(
                output[0],
                output[1],
                speaker_1.squeeze(0),
                speaker_2.squeeze(0),
            )

            value2 = self.calculate(
                output[0],
                output[1],
                speaker_2.squeeze(0),
                speaker_1.squeeze(0),
            )

            perm1.append(value1)
            perm2.append(value2)

        perm1, perm2 = torch.tensor(perm1), torch.tensor(perm2)
        result = torch.maximum(perm1, perm2)

        return result.mean()

    def calculate(
        self,
        output_1_wav: torch.Tensor,
        output_2_wav: torch.Tensor,
        speaker_1_wav: torch.Tensor,
        speaker_2_wav: torch.Tensor,
    ):
        """
        Calculate SNR for one permutation of speakers.

        Args:
            output_1_wav (torch.Tensor): The output waveform for speaker 1 of shape (L)
            output_2_wav (torch.Tensor): The output waveform for speaker 2 of shape (L)
            speaker_1_wav (torch.Tensor): The reference waveform for speaker 1 of shape (L)
            speaker_2_wav (torch.Tensor): The reference waveform for speaker 2 of shape (L)

        Returns:
            float: The average SNR value between the two speakers.
        """
        return (
            self.metric(output_1_wav, speaker_1_wav)
            + self.metric(output_2_wav, speaker_2_wav)
        ) / 2


class SNRiMetric(SNRMetric):
    def __call__(
        self,
        mix_wav: torch.Tensor,
        output_wav: torch.Tensor,
        speaker_1_wav: torch.Tensor,
        speaker_2_wav: torch.Tensor,
        **batch
    ):
        """
        Calculate SNR improvement (SNRi) by comparing the output SNR to the
        baseline SNR calculated on the mixed waveform.

        Args:
            mix_wav (torch.Tensor): The mixed waveform of shape (B x 1 x L)
            output_wav (torch.Tensor): The output waveform of shape (B x 2 x L)
            speaker_1_wav (torch.Tensor): The reference waveform for speaker 1 of shape (B x 1 x L)
            speaker_2_wav (torch.Tensor): The reference waveform for speaker 2 of shape (B x 1 x L)

        Returns:
            float: The SNR improvement between the output and the mixed waveform.
        """
        baseline = super().__call__(
            mix_wav.repeat(1, 2, 1), speaker_1_wav, speaker_2_wav
        )
        improved = super().__call__(output_wav, speaker_1_wav, speaker_2_wav)
        return improved - baseline
