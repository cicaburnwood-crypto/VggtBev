from __future__ import annotations

import torch

from vggt_bev.config import LabelValues

DEFAULT_LABEL_VALUES = LabelValues()


def _iou(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (prediction & target).sum(dtype=torch.float64)
    union = (prediction | target).sum(dtype=torch.float64)
    if union == 0:
        return torch.ones((), dtype=torch.float64, device=prediction.device)
    return intersection / union


def method2_metrics(
    predictions: dict[str, torch.Tensor],
    target_labels: torch.Tensor,
    *,
    threshold: float = 0.5,
    labels: LabelValues = DEFAULT_LABEL_VALUES,
) -> dict[str, torch.Tensor]:
    observed_target = target_labels != labels.unknown
    occupied_target = target_labels == labels.occupied
    observed_prediction = predictions["observed_logit"].sigmoid() >= threshold
    occupied_prediction = predictions["occupancy_logit"].sigmoid() >= threshold
    occupied_prediction = occupied_prediction & observed_target
    return {
        "observed_iou": _iou(observed_prediction, observed_target),
        "occupied_iou_on_observed": _iou(occupied_prediction, occupied_target),
    }
