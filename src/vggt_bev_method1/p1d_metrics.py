from __future__ import annotations

import torch
from torch.nn import functional as F

from vggt_bev_method1.data.p1b_targets import p1b_region_masks


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius == 0:
        return mask.bool()
    return F.max_pool2d(
        mask.float().unsqueeze(1),
        2 * radius + 1,
        stride=1,
        padding=radius,
    ).squeeze(1) > 0.5


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return ~_dilate(~mask.bool(), radius)


def _edge(mask: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
    return (_dilate(mask, 1) ^ _erode(mask, 1)) & _erode(domain, 1)


def _boundary_metrics(
    probability: torch.Tensor,
    truth: torch.Tensor,
    domain: torch.Tensor,
    *,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = (probability >= 0.5) & domain
    truth = truth.bool() & domain
    prediction_edge = _edge(prediction, domain)
    truth_edge = _edge(truth, domain)
    prediction_band = _dilate(prediction_edge, radius) & domain
    truth_band = _dilate(truth_edge, radius) & domain
    intersection = (prediction_band & truth_band).sum().to(torch.float32)
    union = (prediction_band | truth_band).sum().to(torch.float32)
    iou = intersection / union.clamp_min(1.0)
    precision = (
        prediction_edge & _dilate(truth_edge, radius)
    ).sum().to(torch.float32) / prediction_edge.sum().clamp_min(1).to(torch.float32)
    recall = (
        truth_edge & _dilate(prediction_edge, radius)
    ).sum().to(torch.float32) / truth_edge.sum().clamp_min(1).to(torch.float32)
    return iou, precision, recall


@torch.no_grad()
def p1d_validation_metrics(
    prediction: dict,
    complete_target: torch.Tensor,
    visible_target: torch.Tensor,
    support_target: torch.Tensor,
    *,
    latest_observed_free_target: torch.Tensor,
    latest_support_target: torch.Tensor,
    gt_valid_mask: torch.Tensor,
    boundary_radii: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[str, torch.Tensor]:
    masks = p1b_region_masks(complete_target, visible_target, support_target)
    valid = gt_valid_mask.bool()
    semantic_domain = masks.valid & valid
    gate_probability = torch.sigmoid(prediction["observed_gate_logit"].float())
    support_probability = torch.sigmoid(prediction["fov_support_logit"].float())
    values: dict[str, torch.Tensor] = {}
    for radius in boundary_radii:
        gate_iou, gate_precision, gate_recall = _boundary_metrics(
            gate_probability,
            masks.observed_free,
            semantic_domain,
            radius=radius,
        )
        support_iou, support_precision, support_recall = _boundary_metrics(
            support_probability,
            masks.valid,
            valid,
            radius=radius,
        )
        values.update(
            {
                f"gate_boundary_iou_r{radius}": gate_iou,
                f"gate_boundary_precision_r{radius}": gate_precision,
                f"gate_boundary_recall_r{radius}": gate_recall,
                f"support_boundary_iou_r{radius}": support_iou,
                f"support_boundary_precision_r{radius}": support_precision,
                f"support_boundary_recall_r{radius}": support_recall,
            }
        )
    history_support = (
        masks.valid & ~latest_support_target.bool() & valid
    )
    history_observed = (
        masks.observed_free & ~latest_observed_free_target.bool() & valid
    )
    predicted_gate = gate_probability >= 0.5
    predicted_support = support_probability >= 0.5
    values["temporal_gain_support_recall"] = (
        predicted_support & history_support
    ).sum().to(torch.float32) / history_support.sum().clamp_min(1).to(torch.float32)
    values["temporal_gain_observed_recall"] = (
        predicted_gate & history_observed
    ).sum().to(torch.float32) / history_observed.sum().clamp_min(1).to(torch.float32)
    values["temporal_gain_support_fraction"] = history_support.float().mean()
    values["temporal_gain_observed_fraction"] = history_observed.float().mean()
    return values
