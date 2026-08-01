from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from vggt_bev_method1.config import LabelValues

DEFAULT_LABELS = LabelValues()


def _binary_iou(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    intersection = (prediction & truth & valid).sum()
    union = ((prediction | truth) & valid).sum()
    return (intersection + 1).float() / (union + 1)


def _binary_f1(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    true_positive = (prediction & truth & valid).sum()
    false_positive = (prediction & ~truth & valid).sum()
    false_negative = (~prediction & truth & valid).sum()
    return (2 * true_positive + 1).float() / (
        2 * true_positive + false_positive + false_negative + 1
    )


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    value = mask[:, None].float()
    eroded = -F.max_pool2d(-value, kernel_size=3, stride=1, padding=1)
    return mask & (eroded[:, 0] < 0.5)


def _local_maximum(values: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 0:
        raise ValueError("surface tolerance radius cannot be negative")
    if radius == 0:
        return values
    return F.max_pool2d(
        values[:, None],
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )[:, 0]


def _points(mask: torch.Tensor, maximum: int = 1024) -> torch.Tensor:
    points = torch.nonzero(mask, as_tuple=False).float()
    if points.shape[0] > maximum:
        selection = torch.linspace(
            0, points.shape[0] - 1, maximum, device=points.device
        ).round().long()
        points = points[selection]
    return points


def _boundary_chamfer_m(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    extent_m: torch.Tensor,
) -> torch.Tensor:
    distances = []
    height, width = prediction.shape[-2:]
    for index in range(prediction.shape[0]):
        predicted = _points(_boundary(prediction[index : index + 1])[0])
        expected = _points(_boundary(truth[index : index + 1])[0])
        cell = extent_m[index] / max(height, width)
        if predicted.numel() == 0 and expected.numel() == 0:
            distances.append(cell.new_zeros(()))
        elif predicted.numel() == 0 or expected.numel() == 0:
            distances.append(cell * math.sqrt(height**2 + width**2))
        else:
            pairwise = torch.cdist(predicted, expected)
            distances.append(
                0.5
                * (
                    pairwise.amin(dim=1).mean()
                    + pairwise.amin(dim=0).mean()
                )
                * cell
            )
    return torch.stack(distances).mean()


def _expected_calibration_error(
    probability: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    *,
    bins: int = 15,
) -> torch.Tensor:
    if not bool(valid.any()):
        return probability.sum() * 0.0
    selected_probability = probability[valid]
    selected_truth = truth[valid].to(selected_probability.dtype)
    error = selected_probability.new_zeros(())
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        inside = (selected_probability >= lower) & (
            selected_probability <= upper
            if index == bins - 1
            else selected_probability < upper
        )
        if bool(inside.any()):
            error = error + inside.float().mean() * (
                selected_probability[inside].mean()
                - selected_truth[inside].mean()
            ).abs()
    return error


def fov_complete_evidential_metrics(
    prediction: dict[str, torch.Tensor],
    fov_complete_target: torch.Tensor,
    visible_target: torch.Tensor,
    fov_support_target: torch.Tensor,
    *,
    target_extent_m: torch.Tensor,
    surface_tolerance_pixels: int,
    labels: LabelValues = DEFAULT_LABELS,
) -> dict[str, torch.Tensor]:
    """Evaluate FOV support, conditional occupancy and evidence confidence."""

    alpha = prediction["alpha_occupied"].float()
    beta = prediction["beta_free"].float()
    support_probability = prediction["fov_support_probability"].float()
    strength = alpha + beta
    occupancy_probability = alpha / strength
    confidence = (1.0 - 2.0 / strength).clamp(0.0, 1.0)
    predicted_occupied = occupancy_probability >= 0.5
    truth_occupied = fov_complete_target == labels.occupied
    truth_free = fov_complete_target == labels.free
    valid = fov_complete_target != labels.unknown
    support_truth = fov_support_target.bool()
    if not torch.equal(support_truth, valid):
        raise ValueError("FOV support must equal valid FOV-complete cells")
    predicted_support = support_probability >= 0.5
    observed = (visible_target != labels.unknown) & valid
    guessed = (visible_target == labels.unknown) & valid
    observed_free = observed & truth_free
    observed_surface = observed & truth_occupied
    surface_tolerance_band = (
        _local_maximum(
            observed_surface.float(),
            int(surface_tolerance_pixels),
        )
        > 0
    )
    observed_free_outside_surface_band = (
        observed_free & ~surface_tolerance_band
    )
    correct = predicted_occupied == truth_occupied

    def region_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if bool(mask.any()):
            return values[mask].mean()
        return values.sum() * 0.0

    true_positive = (predicted_occupied & truth_occupied & valid).sum()
    false_positive = (predicted_occupied & ~truth_occupied & valid).sum()
    false_negative = (~predicted_occupied & truth_occupied & valid).sum()
    precision = (true_positive + 1).float() / (
        true_positive + false_positive + 1
    )
    recall = (true_positive + 1).float() / (
        true_positive + false_negative + 1
    )
    occupied_f1 = _binary_f1(predicted_occupied, truth_occupied, valid)
    free_f1 = _binary_f1(~predicted_occupied, truth_free, valid)
    observed_free_confidence = region_mean(
        confidence,
        observed_free_outside_surface_band,
    )
    predicted_occupied_float = predicted_occupied.float()
    tolerant_surface_correct = _local_maximum(
        predicted_occupied_float,
        int(surface_tolerance_pixels),
    )
    correct_surface_confidence = _local_maximum(
        confidence * predicted_occupied_float,
        int(surface_tolerance_pixels),
    )
    incorrect_surface_confidence = _local_maximum(
        confidence * (1.0 - predicted_occupied_float),
        int(surface_tolerance_pixels),
    )
    tolerant_surface_confidence = torch.where(
        tolerant_surface_correct > 0,
        correct_surface_confidence,
        incorrect_surface_confidence,
    )
    observed_surface_confidence = region_mean(
        tolerant_surface_confidence,
        observed_surface,
    )
    if bool(observed_free.any()) and bool(observed_surface.any()):
        observed_confidence = 0.5 * (
            observed_free_confidence + observed_surface_confidence
        )
    elif bool(observed_surface.any()):
        observed_confidence = observed_surface_confidence
    else:
        observed_confidence = observed_free_confidence
    guessed_confidence = region_mean(confidence, guessed)
    guessed_true_positive = (
        predicted_occupied & truth_occupied & guessed
    ).sum()
    guessed_false_positive = (
        predicted_occupied & ~truth_occupied & guessed
    ).sum()
    guessed_false_negative = (
        ~predicted_occupied & truth_occupied & guessed
    ).sum()
    guessed_occupied_precision = (
        guessed_true_positive + 1
    ).float() / (
        guessed_true_positive + guessed_false_positive + 1
    )
    guessed_occupied_recall = (
        guessed_true_positive + 1
    ).float() / (
        guessed_true_positive + guessed_false_negative + 1
    )
    observed_free_accuracy = region_mean(
        (~predicted_occupied).float(),
        observed_free,
    )
    observed_free_outside_surface_band_accuracy = region_mean(
        (~predicted_occupied).float(),
        observed_free_outside_surface_band,
    )
    observed_surface_recall = region_mean(
        predicted_occupied.float(),
        observed_surface,
    )
    tolerant_surface_hit = _local_maximum(
        predicted_occupied.float(),
        int(surface_tolerance_pixels),
    )
    observed_surface_recall_with_tolerance = region_mean(
        tolerant_surface_hit,
        observed_surface,
    )
    observed_free_correct = (~predicted_occupied).float()
    observed_free_confidence_brier = region_mean(
        (confidence - observed_free_correct).square(),
        observed_free_outside_surface_band,
    )
    observed_surface_confidence_brier = region_mean(
        (
            tolerant_surface_confidence
            - tolerant_surface_hit.detach()
        ).square(),
        observed_surface,
    )
    observed_free_confidence_ece = _expected_calibration_error(
        confidence,
        ~predicted_occupied,
        observed_free_outside_surface_band,
    )
    observed_surface_confidence_ece = _expected_calibration_error(
        tolerant_surface_confidence,
        tolerant_surface_hit.bool(),
        observed_surface,
    )
    if bool(observed_free_outside_surface_band.any()) and bool(
        observed_surface.any()
    ):
        observed_direct_balanced_accuracy = 0.5 * (
            observed_free_outside_surface_band_accuracy
            + observed_surface_recall_with_tolerance
        )
    elif bool(observed_surface.any()):
        observed_direct_balanced_accuracy = (
            observed_surface_recall_with_tolerance
        )
    else:
        observed_direct_balanced_accuracy = (
            observed_free_outside_surface_band_accuracy
        )
    if bool(observed_free_outside_surface_band.any()) and bool(
        observed_surface.any()
    ):
        observed_confidence_brier = 0.5 * (
            observed_free_confidence_brier
            + observed_surface_confidence_brier
        )
    elif bool(observed_surface.any()):
        observed_confidence_brier = observed_surface_confidence_brier
    else:
        observed_confidence_brier = observed_free_confidence_brier
    if bool(observed_free_outside_surface_band.any()) and bool(
        observed_surface.any()
    ):
        observed_confidence_ece = 0.5 * (
            observed_free_confidence_ece
            + observed_surface_confidence_ece
        )
    elif bool(observed_surface.any()):
        observed_confidence_ece = observed_surface_confidence_ece
    else:
        observed_confidence_ece = observed_free_confidence_ece
    guessed_confidence_brier = region_mean(
        (confidence - correct.to(confidence.dtype)).square(),
        guessed,
    )
    guessed_confidence_ece = _expected_calibration_error(
        confidence,
        correct,
        guessed,
    )
    guessed_occupied_iou = _binary_iou(
        predicted_occupied,
        truth_occupied,
        guessed,
    )
    guessed_all_occupied_baseline_iou = _binary_iou(
        torch.ones_like(predicted_occupied),
        truth_occupied,
        guessed,
    )
    support_true_positive = (predicted_support & support_truth).sum()
    support_false_positive = (predicted_support & ~support_truth).sum()
    support_false_negative = (~predicted_support & support_truth).sum()
    support_precision = (support_true_positive + 1).float() / (
        support_true_positive + support_false_positive + 1
    )
    support_recall = (support_true_positive + 1).float() / (
        support_true_positive + support_false_negative + 1
    )
    return {
        "support_iou": _binary_iou(
            predicted_support,
            support_truth,
            torch.ones_like(support_truth),
        ),
        "support_f1": _binary_f1(
            predicted_support,
            support_truth,
            torch.ones_like(support_truth),
        ),
        "support_precision": support_precision,
        "support_recall": support_recall,
        "inside_mean_support_probability": region_mean(
            support_probability,
            support_truth,
        ),
        "outside_mean_support_probability": region_mean(
            support_probability,
            ~support_truth,
        ),
        "occupied_iou": _binary_iou(
            predicted_occupied,
            truth_occupied,
            valid,
        ),
        "free_iou": _binary_iou(~predicted_occupied, truth_free, valid),
        "occupied_precision": precision,
        "occupied_recall": recall,
        "occupied_f1": occupied_f1,
        "free_f1": free_f1,
        "known_macro_f1": 0.5 * (occupied_f1 + free_f1),
        "occupancy_ece": _expected_calibration_error(
            occupancy_probability,
            truth_occupied,
            valid,
        ),
        "confidence_brier_correctness": region_mean(
            (confidence - correct.to(confidence.dtype)).square(),
            valid,
        ),
        "observed_confidence_brier_correctness": observed_confidence_brier,
        "guessed_confidence_brier_correctness": guessed_confidence_brier,
        "observed_confidence_ece": observed_confidence_ece,
        "guessed_confidence_ece": guessed_confidence_ece,
        "observed_accuracy": region_mean(correct.float(), observed),
        "observed_free_accuracy": observed_free_accuracy,
        "observed_free_false_occupied_rate": region_mean(
            predicted_occupied.float(),
            observed_free,
        ),
        "observed_free_outside_surface_band_false_occupied_rate": region_mean(
            predicted_occupied.float(),
            observed_free_outside_surface_band,
        ),
        "observed_surface_recall": observed_surface_recall,
        "observed_surface_recall_with_tolerance": (
            observed_surface_recall_with_tolerance
        ),
        "observed_direct_balanced_accuracy": (
            observed_direct_balanced_accuracy
        ),
        "guessed_accuracy": region_mean(correct.float(), guessed),
        "guessed_occupied_iou": guessed_occupied_iou,
        "guessed_all_occupied_baseline_iou": (
            guessed_all_occupied_baseline_iou
        ),
        "guessed_occupied_iou_gain_over_all_occupied": (
            guessed_occupied_iou - guessed_all_occupied_baseline_iou
        ),
        "guessed_occupied_precision": guessed_occupied_precision,
        "guessed_occupied_recall": guessed_occupied_recall,
        "guessed_occupied_f1": _binary_f1(
            predicted_occupied,
            truth_occupied,
            guessed,
        ),
        "guessed_target_occupied_fraction": region_mean(
            truth_occupied.float(),
            guessed,
        ),
        "guessed_predicted_occupied_fraction": region_mean(
            occupancy_probability,
            guessed,
        ),
        "guessed_predicted_occupied_binary_fraction": region_mean(
            predicted_occupied.float(),
            guessed,
        ),
        "observed_mean_confidence": observed_confidence,
        "observed_free_mean_confidence": observed_free_confidence,
        "observed_surface_mean_confidence": observed_surface_confidence,
        "guessed_mean_confidence": guessed_confidence,
        "observed_confidence_bias": (
            observed_confidence - observed_direct_balanced_accuracy
        ),
        "guessed_confidence_bias": (
            guessed_confidence - region_mean(correct.float(), guessed)
        ),
        "confidence_gap": observed_confidence - guessed_confidence,
        "observed_fraction": observed.float().mean(),
        "observed_free_fraction": observed_free.float().mean(),
        "observed_free_outside_surface_band_fraction": (
            observed_free_outside_surface_band.float().mean()
        ),
        "observed_surface_fraction": observed_surface.float().mean(),
        "surface_tolerance_band_fraction": (
            surface_tolerance_band.float().mean()
        ),
        "surface_tolerance_pixels": torch.tensor(
            int(surface_tolerance_pixels),
            device=alpha.device,
        ),
        "guessed_fraction": guessed.float().mean(),
        "valid_fraction": valid.float().mean(),
        "predicted_occupied_fraction": region_mean(
            occupancy_probability,
            valid,
        ),
        "target_occupied_fraction": region_mean(
            truth_occupied.float(),
            valid,
        ),
        "boundary_chamfer_m": _boundary_chamfer_m(
            predicted_occupied & predicted_support,
            truth_occupied,
            target_extent_m,
        ),
    }
