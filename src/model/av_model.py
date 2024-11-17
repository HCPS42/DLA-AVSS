import torch
from torch import nn

from src.model.base_model import (
    BaseAudioModel,
    BaseModel,
    BaseSeparatorModel,
    BaseVisualModel,
)


class SeparatorWrapper(BaseSeparatorModel):
    def __init__(
        self,
        separator: BaseSeparatorModel,
        visual_model: BaseVisualModel,
        pre_audio_encoder: nn.Module | None,
    ):
        super().__init__()
        self.embedding_size = separator.embedding_size

        self.separator = separator
        self.visual_model = visual_model
        self.pre_audio_encoder = pre_audio_encoder

        self.projection = nn.Conv1d(
            self.separator.embedding_size + self.visual_model.embedding_size * 2,
            self.separator.embedding_size,
            kernel_size=1,
        )

    def forward(self, audio_features: torch.Tensor, mix_visual: torch.Tensor):
        if self.pre_audio_encoder is not None:
            audio_features = self.pre_audio_encoder(audio_features)

        visual_features = self.visual_model(mix_visual, audio_features.size(-1))

        fused_features = torch.cat((audio_features, visual_features), dim=1)
        fused_features = self.projection(fused_features)

        return self.separator(fused_features, None)


class AudioVisualModel(BaseModel):
    def __init__(
        self,
        audio_model: BaseAudioModel,
        visual_model: BaseVisualModel,
        pre_audio_encoder: nn.Module | None = None,
        **kwargs
    ):
        super().__init__()

        self.audio_model = audio_model
        self.audio_model.separator = SeparatorWrapper(
            audio_model.separator, visual_model, pre_audio_encoder, **kwargs
        )

    def forward(self, **batch):
        return self.audio_model(**batch)
