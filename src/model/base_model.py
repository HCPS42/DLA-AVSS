from abc import abstractmethod

import torch
from torch import nn


class BaseModel(nn.Module):
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


class BaseVisualModel(BaseModel):
    embedding_size: int

    @abstractmethod
    def forward(self, mix_visual: torch.Tensor, audio_len: int):
        raise NotImplementedError()


class BaseSeparatorModel(BaseModel):
    embedding_size: int

    @abstractmethod
    def forward(self, audio_features: torch.Tensor, mix_visual: torch.Tensor):
        raise NotImplementedError()


class BaseAudioModel(BaseModel):
    encoder: nn.Module
    separator: BaseSeparatorModel
    decoder: nn.Module

    @abstractmethod
    def forward(self, mix_wav: torch.Tensor, mix_visual: torch.Tensor, **batch):
        """
        Args:
            mix_wav (torch.Tensor): Input tensor representing the mixed waveform.
                Shape: (batch_size, 1, time_steps), tested on time_steps = 32000.

        Returns:
            dict: A dictionary containing the separated waveforms.
                - output_wav (torch.Tensor): Output tensor representing the separated waveforms.
                    Shape: (batch_size, 2, time_steps)
        """
        raise NotImplementedError()

    @abstractmethod
    def pad(self, x: torch.Tensor):
        raise NotImplementedError()
