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
        speaker1_wav: torch.Tensor,
        speaker2_wav: torch.Tensor,
        **batch
    ):
        """
        Calculate permutation-invariant signal-to-noise ratio (SNR).

        Args:
            output_wav (torch.Tensor): The output waveform of shape (B x 2 x L).
            speaker1_wav (torch.Tensor): The reference waveform for speaker 1 of shape (B x 1 x L)
            speaker2_wav (torch.Tensor): The reference waveform for speaker 2 of shape (B x 1 x L)

        Returns:
            float: The best SNR value between two permutations of speakers.
        """
        value1 = self.calculate(
            output_wav[:, 0],
            output_wav[:, 1],
            speaker1_wav.squeeze(1),
            speaker2_wav.squeeze(1),
        )

        value2 = self.calculate(
            output_wav[:, 0],
            output_wav[:, 1],
            speaker2_wav.squeeze(1),
            speaker1_wav.squeeze(1),
        )

        return max(value1, value2)

    def calculate(
        self,
        output1_wav: torch.Tensor,
        output2_wav: torch.Tensor,
        speaker1_wav: torch.Tensor,
        speaker2_wav: torch.Tensor,
    ):
        """
        Calculate SNR for one permutation of speakers.

        Args:
            output1_wav (torch.Tensor): The output waveform for speaker 1 of shape (B x L)
            output2_wav (torch.Tensor): The output waveform for speaker 2 of shape (B x L)
            speaker1_wav (torch.Tensor): The reference waveform for speaker 1 of shape (B x L)
            speaker2_wav (torch.Tensor): The reference waveform for speaker 2 of shape (B x L)

        Returns:
            float: The average SNR value between the two speakers.
        """
        return (
            self.metric(output1_wav, speaker1_wav)
            + self.metric(output2_wav, speaker2_wav)
        ) / 2


class SNRiMetric(SNRMetric):
    def __call__(
        self,
        mix_wav: torch.Tensor,
        output_wav: torch.Tensor,
        speaker1_wav: torch.Tensor,
        speaker2_wav: torch.Tensor,
        **batch
    ):
        """
        Calculate SNR improvement (SNRi) by comparing the output SNR to the
        baseline SNR calculated on the mixed waveform.

        Args:
            mix_wav (torch.Tensor): The mixed waveform of shape (B x 1 x L)
            output_wav (torch.Tensor): The output waveform of shape (B x 2 x L)
            speaker1_wav (torch.Tensor): The reference waveform for speaker 1 of shape (B x 1 x L)
            speaker2_wav (torch.Tensor): The reference waveform for speaker 2 of shape (B x 1 x L)

        Returns:
            float: The SNR improvement between the output and the mixed waveform.
        """
        baseline = super().__call__(mix_wav.repeat(1, 2, 1), speaker1_wav, speaker2_wav)
        improved = super().__call__(output_wav, speaker1_wav, speaker2_wav)
        return improved - baseline
