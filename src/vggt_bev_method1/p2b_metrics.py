from __future__ import annotations

import torch

from vggt_bev_method1.config import LabelValues
from vggt_bev_method1.data.p2b_targets import (
    ROUTING_GUESSED_FREE,
    ROUTING_GUESSED_OCCUPIED,
    ROUTING_OBSERVED_FREE,
    p2b_region_masks,
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    return float(numerator) / denominator if denominator > 0.0 else 0.0


def _pixel_class_counts(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    class_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_class = valid & (predicted == class_index)
    target_class = valid & (target == class_index)
    return (
        (predicted_class & target_class).sum(),
        (predicted_class & ~target_class).sum(),
        (~predicted_class & target_class).sum(),
    )


def p2b_metric_totals(
    prediction: dict,
    complete_target: torch.Tensor,
    visible_target: torch.Tensor,
    support_target: torch.Tensor,
    *,
    labels: LabelValues = LabelValues(),
) -> dict[str, torch.Tensor]:
    masks = p2b_region_masks(
        complete_target,
        visible_target,
        support_target,
        labels=labels,
    )
    device = complete_target.device
    routing_predicted = prediction["routing_probability"].argmax(dim=1)
    routing_counts = {
        "routing_free": _pixel_class_counts(
            routing_predicted,
            masks.routing_target,
            masks.valid,
            ROUTING_OBSERVED_FREE,
        ),
        "routing_guessed_free": _pixel_class_counts(
            routing_predicted,
            masks.routing_target,
            masks.valid,
            ROUTING_GUESSED_FREE,
        ),
        "routing_guessed_occupied": _pixel_class_counts(
            routing_predicted,
            masks.routing_target,
            masks.valid,
            ROUTING_GUESSED_OCCUPIED,
        ),
    }
    predicted_guessed = masks.valid & (
        routing_predicted != ROUTING_OBSERVED_FREE
    )
    routing_guessed_tp = (predicted_guessed & masks.guessed).sum()
    routing_guessed_fp = (predicted_guessed & ~masks.guessed).sum()
    routing_guessed_fn = (~predicted_guessed & masks.guessed).sum()

    # Monitor the Observed Gate itself, independently of the final three-way
    # routing argmax. Its GT is exactly the masked-BEV observed-free region.
    observed_gate_predicted = (
        prediction["observed_gate_probability"].float() >= 0.5
    )
    observed_gate_truth = masks.observed_free
    observed_gate_tp = (
        masks.valid & observed_gate_predicted & observed_gate_truth
    ).sum()
    observed_gate_fp = (
        masks.valid & observed_gate_predicted & ~observed_gate_truth
    ).sum()
    observed_gate_fn = (
        masks.valid & ~observed_gate_predicted & observed_gate_truth
    ).sum()
    surface_gate_predicted = ~observed_gate_predicted
    direct_band = masks.observed_free | masks.visible_surface
    surface_gate_tp = (
        direct_band & surface_gate_predicted & masks.visible_surface
    ).sum()
    surface_gate_fp = (
        direct_band & surface_gate_predicted & masks.observed_free
    ).sum()
    surface_gate_fn = (
        direct_band & ~surface_gate_predicted & masks.visible_surface
    ).sum()

    guessed_probability = prediction["guessed"]["occupancy_probability"].float()
    guessed_predicted = guessed_probability >= 0.5
    guessed_truth = masks.occupied
    guessed_tp = (masks.guessed & guessed_predicted & guessed_truth).sum()
    guessed_fp = (masks.guessed & guessed_predicted & ~guessed_truth).sum()
    guessed_fn = (masks.guessed & ~guessed_predicted & guessed_truth).sum()
    guessed_tn = (masks.guessed & ~guessed_predicted & ~guessed_truth).sum()
    hidden_guessed_tp = (
        masks.hidden_guessed & guessed_predicted & masks.hidden_guessed_occupied
    ).sum()
    hidden_guessed_fp = (
        masks.hidden_guessed & guessed_predicted & ~masks.hidden_guessed_occupied
    ).sum()
    hidden_guessed_fn = (
        masks.hidden_guessed & ~guessed_predicted & masks.hidden_guessed_occupied
    ).sum()

    fused_probability = prediction["fused"]["occupancy_probability"].float()
    fused_predicted = fused_probability >= 0.5
    observed_free_fp = (masks.observed_free & fused_predicted).sum()
    observed_free_count = masks.observed_free.sum()
    fused_surface_tp = (masks.visible_surface & fused_predicted).sum()
    fused_surface_fp = (masks.observed_free & fused_predicted).sum()
    fused_surface_fn = (masks.visible_surface & ~fused_predicted).sum()

    support_predicted = prediction["fov_support_probability"] >= 0.5
    support_tp = (support_predicted & masks.valid).sum()
    support_fp = (support_predicted & ~masks.valid).sum()
    support_fn = (~support_predicted & masks.valid).sum()

    correct = fused_predicted == masks.occupied
    confidence = prediction["fused"]["navigation_confidence"].float()
    confidence_error = (confidence - correct.to(confidence.dtype)).square()
    confidence_brier_sum = confidence_error[masks.valid].sum()
    confidence_count = masks.valid.sum()
    high_confidence_wrong = (masks.valid & ~correct & (confidence >= 0.8)).sum()

    def count(value: torch.Tensor) -> torch.Tensor:
        return value.to(device=device, dtype=torch.float64)

    output: dict[str, torch.Tensor] = {
        "guessed_tp": count(guessed_tp),
        "guessed_fp": count(guessed_fp),
        "guessed_fn": count(guessed_fn),
        "guessed_tn": count(guessed_tn),
        "observed_free_fp": count(observed_free_fp),
        "observed_free_count": count(observed_free_count),
        "routing_guessed_tp": count(routing_guessed_tp),
        "routing_guessed_fp": count(routing_guessed_fp),
        "routing_guessed_fn": count(routing_guessed_fn),
        "observed_gate_tp": count(observed_gate_tp),
        "observed_gate_fp": count(observed_gate_fp),
        "observed_gate_fn": count(observed_gate_fn),
        "surface_gate_tp": count(surface_gate_tp),
        "surface_gate_fp": count(surface_gate_fp),
        "surface_gate_fn": count(surface_gate_fn),
        "fused_surface_tp": count(fused_surface_tp),
        "fused_surface_fp": count(fused_surface_fp),
        "fused_surface_fn": count(fused_surface_fn),
        "hidden_guessed_tp": count(hidden_guessed_tp),
        "hidden_guessed_fp": count(hidden_guessed_fp),
        "hidden_guessed_fn": count(hidden_guessed_fn),
        "support_tp": count(support_tp),
        "support_fp": count(support_fp),
        "support_fn": count(support_fn),
        "confidence_brier_sum": count(confidence_brier_sum),
        "confidence_count": count(confidence_count),
        "high_confidence_wrong": count(high_confidence_wrong),
    }
    for name, (true_positive, false_positive, false_negative) in routing_counts.items():
        output[f"{name}_tp"] = count(true_positive)
        output[f"{name}_fp"] = count(false_positive)
        output[f"{name}_fn"] = count(false_negative)
    return output


def finalize_p2b_metrics(totals: dict[str, float]) -> dict[str, float]:
    def precision(prefix: str) -> float:
        return _safe_ratio(
            totals[f"{prefix}_tp"],
            totals[f"{prefix}_tp"] + totals[f"{prefix}_fp"],
        )

    def recall(prefix: str) -> float:
        return _safe_ratio(
            totals[f"{prefix}_tp"],
            totals[f"{prefix}_tp"] + totals[f"{prefix}_fn"],
        )

    def iou(prefix: str) -> float:
        return _safe_ratio(
            totals[f"{prefix}_tp"],
            totals[f"{prefix}_tp"]
            + totals[f"{prefix}_fp"]
            + totals[f"{prefix}_fn"],
        )

    guessed_precision = _safe_ratio(
        totals["guessed_tp"], totals["guessed_tp"] + totals["guessed_fp"]
    )
    guessed_recall = _safe_ratio(
        totals["guessed_tp"], totals["guessed_tp"] + totals["guessed_fn"]
    )
    surface_precision = precision("fused_surface")
    surface_recall = recall("fused_surface")
    hidden_precision = precision("hidden_guessed")
    hidden_recall = recall("hidden_guessed")
    routing_ious = [
        iou("routing_free"),
        iou("routing_guessed_free"),
        iou("routing_guessed_occupied"),
    ]
    return {
        "observed_gate_precision": precision("observed_gate"),
        "observed_gate_recall": recall("observed_gate"),
        "observed_gate_f1": _safe_ratio(
            2.0 * precision("observed_gate") * recall("observed_gate"),
            precision("observed_gate") + recall("observed_gate"),
        ),
        "observed_gate_iou": iou("observed_gate"),
        "surface_gate_precision": precision("surface_gate"),
        "surface_gate_recall": recall("surface_gate"),
        "visible_surface_precision": surface_precision,
        "visible_surface_recall": surface_recall,
        "visible_surface_f1": _safe_ratio(
            2.0 * surface_precision * surface_recall,
            surface_precision + surface_recall,
        ),
        "visible_surface_iou": iou("fused_surface"),
        "observed_free_false_occupied_rate": _safe_ratio(
            totals["observed_free_fp"], totals["observed_free_count"]
        ),
        "guessed_free_precision": precision("routing_guessed_free"),
        "guessed_free_recall": recall("routing_guessed_free"),
        "guessed_free_iou": iou("routing_guessed_free"),
        "guessed_occupied_precision": precision("routing_guessed_occupied"),
        "guessed_occupied_recall": recall("routing_guessed_occupied"),
        "guessed_occupied_iou": iou("routing_guessed_occupied"),
        "guessed_pixelwise_precision": guessed_precision,
        "guessed_pixelwise_recall": guessed_recall,
        "guessed_pixelwise_f1": _safe_ratio(
            2.0 * guessed_precision * guessed_recall,
            guessed_precision + guessed_recall,
        ),
        "hidden_occupied_precision": hidden_precision,
        "hidden_occupied_recall": hidden_recall,
        "hidden_occupied_f1": _safe_ratio(
            2.0 * hidden_precision * hidden_recall,
            hidden_precision + hidden_recall,
        ),
        "hidden_occupied_iou": iou("hidden_guessed"),
        "guessed_region_iou": iou("routing_guessed"),
        "routing_mean_iou": sum(routing_ious) / len(routing_ious),
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
