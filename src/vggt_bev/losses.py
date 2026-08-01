from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from vggt_bev.config import LabelValues


@dataclass(frozen=True)
class Method2LossWeights:
    occupancy: float = 1.0
    observation: float = 0.5
    dice: float = 0.25


DEFAULT_LOSS_WEIGHTS = Method2LossWeights()
DEFAULT_LABEL_VALUES = LabelValues()


def method2_loss(
    predictions: dict[str, torch.Tensor],
    target_labels: torch.Tensor,
    *,
    weights: Method2LossWeights = DEFAULT_LOSS_WEIGHTS,
    labels: LabelValues = DEFAULT_LABEL_VALUES,
) -> dict[str, torch.Tensor]:
    """Observed-map loss with a hard no-leakage mask on occupancy supervision."""

    occupancy_logit = predictions["occupancy_logit"]
    observed_logit = predictions["observed_logit"]
    if occupancy_logit.shape != target_labels.shape or observed_logit.shape != target_labels.shape:
        raise ValueError("prediction and target shapes must match")

    observed_target = target_labels != labels.unknown
    occupied_target = target_labels == labels.occupied
    observation_loss = F.binary_cross_entropy_with_logits(
        observed_logit, observed_target.to(observed_logit.dtype)
    )

    if observed_target.any():
        occupancy_loss = F.binary_cross_entropy_with_logits(
            occupancy_logit[observed_target],
            occupied_target[observed_target].to(occupancy_logit.dtype),
        )
        probability = occupancy_logit[observed_target].sigmoid()
        truth = occupied_target[observed_target].to(probability.dtype)
        intersection = (probability * truth).sum()
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (probability.sum() + truth.sum() + 1.0)
    else:
        occupancy_loss = occupancy_logit.sum() * 0.0
        dice_loss = occupancy_logit.sum() * 0.0

    total = (
        weights.occupancy * occupancy_loss
        + weights.observation * observation_loss
        + weights.dice * dice_loss
    )
    return {
        "loss": total,
        "occupancy_loss": occupancy_loss,
        "observation_loss": observation_loss,
        "dice_loss": dice_loss,
    }
