from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.gelu(inputs + self.block(inputs))


class BEVDecoder(nn.Module):
    """Small 2D refinement decoder producing occupancy and observation logits."""

    def __init__(self, input_channels: int, hidden_channels: int, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden_channels), hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(ResidualBlock(hidden_channels), ResidualBlock(hidden_channels))
        self.output = nn.Conv2d(hidden_channels, 2, 1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.blocks(self.stem(inputs))
        logits = self.output(features)
        if logits.shape[-2:] != (self.output_size, self.output_size):
            logits = F.interpolate(
                logits,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return {
            "occupancy_logit": logits[:, 0],
            "observed_logit": logits[:, 1],
        }

