import torch

from src.metrics.base_metric import BaseMetric


class SNRMetric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        """
        Args:
            metric (Callable): function to calculate metrics.
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

        return min(value1, value2)

    def calculate(
        self,
        output1_wav: torch.Tensor,
        output2_wav: torch.Tensor,
        speaker1_wav: torch.Tensor,
        speaker2_wav: torch.Tensor,
    ):
        return (
            self.metric(output1_wav, speaker1_wav)
            + self.metric(output2_wav, speaker2_wav)
        ) / 2
