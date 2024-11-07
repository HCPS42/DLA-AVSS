from typing import Literal

import torch
from torch import nn


class Loss(nn.Module):
    def __init__(
        self,
        loss: Literal["mse", "mae"],
        output: Literal["spec", "wav"],
    ):
        """
        Initialize the loss function.

        Args:
            loss (Literal["mse", "mae"]): Type of loss function to use.
                "mse" for Mean Squared Error, "mae" for Mean Absolute Error.
            output (Literal["spec", "wav"]): Type of output to evaluate.
                "spec" for spectrogram, "wav" for waveform.
        """
        assert loss in ["mse", "mae"]
        assert output in ["spec", "wav"]

        super().__init__()

        if loss == "mse":
            self.criterion = nn.MSELoss()
        elif loss == "mae":
            self.criterion = nn.L1Loss()

        if output == "spec":
            self.output_key, self.speaker_1_key, self.speaker_2_key = (
                "output_spec",
                "speaker_1_spec",
                "speaker_2_spec",
            )
        elif output == "wav":
            self.output_key, self.speaker_1_key, self.speaker_2_key = (
                "output_wav",
                "speaker_1_wav",
                "speaker_2_wav",
            )

    def forward(self, **batch):
        """
        Compute the loss for a given batch.

        The loss is computed as the mean of two permutations of the loss
        between the output and the two speakers. The permutation with the
        lower loss is chosen.

        Args:
            **batch: dict containing the batch data.

        Returns:
            dict: a dict containing the loss.
        """
        loss1 = self.criterion(
            batch[self.output_key][:, 0], batch[self.speaker_1_key]
        ) + self.criterion(batch[self.output_key][:, 1], batch[self.speaker_2_key])

        loss2 = self.criterion(
            batch[self.output_key][:, 0], batch[self.speaker_2_key]
        ) + self.criterion(batch[self.output_key][:, 1], batch[self.speaker_1_key])

        return {"loss": min(loss1, loss2)}
