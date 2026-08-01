from __future__ import annotations

import torch

from vggt_bev_method1.config import LabelValues
from vggt_bev_method1.data.p2b_targets import (
    build_packed_ray_bank,
    p2b_region_masks,
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    return float(numerator) / denominator if denominator > 0.0 else 0.0


def _surface_counts(
    probability: torch.Tensor,
    observed_free: torch.Tensor,
    observed_surface: torch.Tensor,
    support: torch.Tensor,
    *,
    tolerance_cells: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = probability.device
    bank = build_packed_ray_bank(support, device=device)
    zero = torch.zeros((), device=device, dtype=torch.float64)
    if bank.indices.shape[0] == 0:
        return zero, zero, zero, zero, zero
    indices = bank.indices
    valid = bank.valid
    positions = torch.arange(indices.shape[1], device=device).unsqueeze(0)
    surface = observed_surface.flatten()[indices] & valid
    observed = (observed_free | observed_surface).flatten()[indices] & valid
    sentinel = indices.shape[1]
    gt_hit = torch.where(
        surface,
        positions,
        torch.full_like(positions, sentinel),
    ).min(dim=1).values
    has_gt = gt_hit < sentinel
    last_observed = torch.where(
        observed,
        positions,
        torch.full_like(positions, -1),
    ).max(dim=1).values
    active = last_observed >= 0
    predicted_occupied = (probability.flatten()[indices] >= 0.5) & valid
    predicted_occupied &= positions <= last_observed[:, None]
    pred_hit = torch.where(
        predicted_occupied,
        positions,
        torch.full_like(positions, sentinel),
    ).min(dim=1).values
    has_pred = pred_hit < sentinel
    matched = (
        active
        & has_gt
        & has_pred
        & ((pred_hit - gt_hit).abs() <= int(tolerance_cells))
    )
    mismatch = active & has_gt & has_pred & ~matched
    true_positive = matched.sum().to(torch.float64)
    false_positive = (
        (active & ~has_gt & has_pred).sum() + mismatch.sum()
    ).to(torch.float64)
    false_negative = (
        (active & has_gt & ~has_pred).sum() + mismatch.sum()
    ).to(torch.float64)
    distance_error = (
        (pred_hit[matched] - gt_hit[matched]).abs().sum().to(torch.float64)
    )
    distance_count = matched.sum().to(torch.float64)
    return (
        true_positive,
        false_positive,
        false_negative,
        distance_error,
        distance_count,
    )


def p2b_metric_totals(
    prediction: dict,
    complete_target: torch.Tensor,
    visible_target: torch.Tensor,
    support_target: torch.Tensor,
    *,
    surface_tolerance_cells: int = 1,
    labels: LabelValues = LabelValues(),
) -> dict[str, torch.Tensor]:
    masks = p2b_region_masks(
        complete_target,
        visible_target,
        support_target,
        labels=labels,
    )
    device = complete_target.device
    surface_values = [
        _surface_counts(
            prediction["observed"]["occupancy_probability"][index].float(),
            masks.observed_free[index],
            masks.observed_surface[index],
            masks.valid[index],
            tolerance_cells=surface_tolerance_cells,
        )
        for index in range(complete_target.shape[0])
    ]
    surface = [torch.stack(values).sum() for values in zip(*surface_values, strict=True)]

    guessed_probability = prediction["guessed"]["occupancy_probability"].float()
    guessed_predicted = guessed_probability >= 0.5
    guessed_truth = masks.occupied
    guessed_tp = (masks.guessed & guessed_predicted & guessed_truth).sum()
    guessed_fp = (masks.guessed & guessed_predicted & ~guessed_truth).sum()
    guessed_fn = (masks.guessed & ~guessed_predicted & guessed_truth).sum()
    guessed_tn = (masks.guessed & ~guessed_predicted & ~guessed_truth).sum()

    observed_predicted = (
        prediction["observed"]["occupancy_probability"].float() >= 0.5
    )
    observed_free_fp = (masks.observed_free & observed_predicted).sum()
    observed_free_count = masks.observed_free.sum()

    gate_predicted = prediction["gate_probability"] >= 0.5
    gate_tp = (masks.valid & gate_predicted & masks.observed).sum()
    gate_fp = (masks.valid & gate_predicted & masks.guessed).sum()
    gate_fn = (masks.valid & ~gate_predicted & masks.observed).sum()

    support_predicted = prediction["fov_support_probability"] >= 0.5
    support_tp = (support_predicted & masks.valid).sum()
    support_fp = (support_predicted & ~masks.valid).sum()
    support_fn = (~support_predicted & masks.valid).sum()

    fused_probability = prediction["fused"]["occupancy_probability"].float()
    fused_predicted = fused_probability >= 0.5
    correct = fused_predicted == masks.occupied
    confidence = prediction["fused"]["navigation_confidence"].float()
    confidence_error = (
        confidence - correct.to(confidence.dtype)
    ).square()
    confidence_brier_sum = confidence_error[masks.valid].sum()
    confidence_count = masks.valid.sum()
    high_confidence_wrong = (
        masks.valid & ~correct & (confidence >= 0.8)
    ).sum()

    def count(value: torch.Tensor) -> torch.Tensor:
        return value.to(device=device, dtype=torch.float64)

    return {
        "surface_tp": surface[0],
        "surface_fp": surface[1],
        "surface_fn": surface[2],
        "surface_hit_distance_error_cells": surface[3],
        "surface_hit_distance_count": surface[4],
        "guessed_tp": count(guessed_tp),
        "guessed_fp": count(guessed_fp),
        "guessed_fn": count(guessed_fn),
        "guessed_tn": count(guessed_tn),
        "observed_free_fp": count(observed_free_fp),
        "observed_free_count": count(observed_free_count),
        "gate_tp": count(gate_tp),
        "gate_fp": count(gate_fp),
        "gate_fn": count(gate_fn),
        "support_tp": count(support_tp),
        "support_fp": count(support_fp),
        "support_fn": count(support_fn),
        "confidence_brier_sum": count(confidence_brier_sum),
        "confidence_count": count(confidence_count),
        "high_confidence_wrong": count(high_confidence_wrong),
    }


def finalize_p2b_metrics(totals: dict[str, float]) -> dict[str, float]:
    surface_precision = _safe_ratio(
        totals["surface_tp"], totals["surface_tp"] + totals["surface_fp"]
    )
    surface_recall = _safe_ratio(
        totals["surface_tp"], totals["surface_tp"] + totals["surface_fn"]
    )
    guessed_precision = _safe_ratio(
        totals["guessed_tp"], totals["guessed_tp"] + totals["guessed_fp"]
    )
    guessed_recall = _safe_ratio(
        totals["guessed_tp"], totals["guessed_tp"] + totals["guessed_fn"]
    )
    return {
        "surface_precision": surface_precision,
        "surface_recall": surface_recall,
        "surface_f1": _safe_ratio(
            2.0 * surface_precision * surface_recall,
            surface_precision + surface_recall,
        ),
        "surface_first_hit_mae_cells": _safe_ratio(
            totals["surface_hit_distance_error_cells"],
            totals["surface_hit_distance_count"],
        ),
        "observed_free_false_occupied_rate": _safe_ratio(
            totals["observed_free_fp"], totals["observed_free_count"]
        ),
        "guessed_pixelwise_precision": guessed_precision,
        "guessed_pixelwise_recall": guessed_recall,
        "guessed_pixelwise_f1": _safe_ratio(
            2.0 * guessed_precision * guessed_recall,
            guessed_precision + guessed_recall,
        ),
        "gate_iou": _safe_ratio(
            totals["gate_tp"],
            totals["gate_tp"] + totals["gate_fp"] + totals["gate_fn"],
        ),
        "support_iou": _safe_ratio(
            totals["support_tp"],
            totals["support_tp"] + totals["support_fp"] + totals["support_fn"],
        ),
        "confidence_brier_correctness": _safe_ratio(
            totals["confidence_brier_sum"], totals["confidence_count"]
        ),
        "high_confidence_wrong_rate": _safe_ratio(
            totals["high_confidence_wrong"], totals["confidence_count"]
        ),
    }
