import torch
from torch import nn


class MSELoss(nn.MSELoss):
    """
    nn.MSELoss wrapper
    """

    def forward(
        self,
        speaker1_spec: torch.Tensor,
        speaker2_spec: torch.Tensor,
        output_spec: torch.Tensor,
        **batch
    ):
        loss1 = super().forward(output_spec, speaker1_spec)
        loss2 = super().forward(output_spec, speaker2_spec)

        return {"loss": min(loss1, loss2)}
