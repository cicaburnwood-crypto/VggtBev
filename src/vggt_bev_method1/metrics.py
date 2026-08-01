from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from vggt_bev_method1.config import LabelValues, Supervision

DEFAULT_LABELS = LabelValues()


def _binary_iou(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    intersection = (prediction & truth & valid).sum()
    union = ((prediction | truth) & valid).sum()
    return (intersection + 1).to(torch.float32) / (union + 1)


def _binary_f1(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    true_positive = (prediction & truth & valid).sum()
    false_positive = (prediction & ~truth & valid).sum()
    false_negative = (~prediction & truth & valid).sum()
    return (2 * true_positive + 1).to(torch.float32) / (
        2 * true_positive + false_positive + false_negative + 1
    )


def _expected_calibration_error(
    probability: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    *,
    bins: int = 10,
) -> torch.Tensor:
    probability = probability[valid]
    truth = truth[valid].to(probability.dtype)
    if probability.numel() == 0:
        return probability.new_zeros(())
    result = probability.new_zeros(())
    boundaries = torch.linspace(
        0.0,
        1.0,
        bins + 1,
        device=probability.device,
        dtype=probability.dtype,
    )
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= boundaries[index]) & (
                probability <= boundaries[index + 1]
            )
        else:
            selected = (probability >= boundaries[index]) & (
                probability < boundaries[index + 1]
            )
        if selected.any():
            weight = selected.to(probability.dtype).mean()
            result = result + weight * torch.abs(
                probability[selected].mean() - truth[selected].mean()
            )
    return result


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask[:, None].to(torch.float32)
    eroded = -F.max_pool2d(-mask_float, kernel_size=3, stride=1, padding=1)
    return mask & (eroded[:, 0] < 0.5)


def _sample_coordinates(mask: torch.Tensor, maximum_points: int = 1024) -> torch.Tensor:
    coordinates = torch.nonzero(mask, as_tuple=False).to(torch.float32)
    if coordinates.shape[0] > maximum_points:
        indices = torch.linspace(
            0,
            coordinates.shape[0] - 1,
            maximum_points,
            device=coordinates.device,
        ).round().long()
        coordinates = coordinates[indices]
    return coordinates


def _boundary_chamfer_m(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    target_extent_m: torch.Tensor,
) -> torch.Tensor:
    distances = []
    height, width = prediction.shape[-2:]
    for index in range(prediction.shape[0]):
        predicted_boundary = _boundary(
            prediction[index : index + 1] & valid[index : index + 1]
        )[0]
        truth_boundary = _boundary(
            truth[index : index + 1] & valid[index : index + 1]
        )[0]
        predicted_points = _sample_coordinates(predicted_boundary)
        truth_points = _sample_coordinates(truth_boundary)
        cell_size = target_extent_m[index] / max(height, width)
        if predicted_points.numel() == 0 and truth_points.numel() == 0:
            distances.append(cell_size.new_zeros(()))
        elif predicted_points.numel() == 0 or truth_points.numel() == 0:
            diagonal = math.sqrt(height**2 + width**2)
            distances.append(cell_size * diagonal)
        else:
            pairwise = torch.cdist(predicted_points, truth_points)
            chamfer_cells = 0.5 * (
                pairwise.amin(dim=1).mean() + pairwise.amin(dim=0).mean()
            )
            distances.append(chamfer_cells * cell_size)
    return torch.stack(distances).mean()


def branch_metrics(
    prediction: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    supervision: Supervision,
    target_extent_m: torch.Tensor,
    threshold: float = 0.5,
    labels: LabelValues = DEFAULT_LABELS,
) -> dict[str, torch.Tensor]:
    del supervision  # Both joint branches explicitly predict target validity.
    valid = target != labels.unknown
    truth_occupied = target == labels.occupied
    truth_free = target == labels.free
    occupancy_probability = prediction["occupancy_logit"].sigmoid()
    predicted_occupied = occupancy_probability >= threshold
    predicted_free = ~predicted_occupied

    occupied_f1 = _binary_f1(predicted_occupied, truth_occupied, valid)
    free_f1 = _binary_f1(predicted_free, truth_free, valid)
    output = {
        "occupied_iou": _binary_iou(predicted_occupied, truth_occupied, valid),
        "free_iou": _binary_iou(predicted_free, truth_free, valid),
        "occupied_f1": occupied_f1,
        "free_f1": free_f1,
        "known_macro_f1": 0.5 * (occupied_f1 + free_f1),
        "occupancy_ece": _expected_calibration_error(
            occupancy_probability,
            truth_occupied,
            valid,
        ),
        "boundary_chamfer_m": _boundary_chamfer_m(
            predicted_occupied,
            truth_occupied,
            valid,
            target_extent_m,
        ),
    }

    observed_probability = prediction["observed_logit"].sigmoid()
    predicted_valid = observed_probability >= threshold
    truth_unknown = ~valid
    predicted_unknown = ~predicted_valid
    unknown_f1 = _binary_f1(
        predicted_unknown,
        truth_unknown,
        torch.ones_like(valid),
    )
    output.update(
        {
            "unknown_iou": _binary_iou(
                predicted_unknown,
                truth_unknown,
                torch.ones_like(valid),
            ),
            "unknown_f1": unknown_f1,
            "three_class_macro_f1": (
                occupied_f1 + free_f1 + unknown_f1
            )
            / 3.0,
            "observation_ece": _expected_calibration_error(
                observed_probability,
                valid,
                torch.ones_like(valid),
            ),
        }
    )
    return output


def observed_categorical_metrics(
    prediction: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    target_extent_m: torch.Tensor,
    labels: LabelValues = DEFAULT_LABELS,
) -> dict[str, torch.Tensor]:
    logits = prediction["class_logits"]
    predicted_class = logits.argmax(dim=1)
    predicted_unknown = predicted_class == 0
    predicted_free = predicted_class == 1
    predicted_occupied = predicted_class == 2
    truth_unknown = target == labels.unknown
    truth_free = target == labels.free
    truth_occupied = target == labels.occupied
    everywhere = torch.ones_like(truth_unknown)
    occupied_f1 = _binary_f1(predicted_occupied, truth_occupied, everywhere)
    free_f1 = _binary_f1(predicted_free, truth_free, everywhere)
    unknown_f1 = _binary_f1(predicted_unknown, truth_unknown, everywhere)
    return {
        "occupied_iou": _binary_iou(
            predicted_occupied,
            truth_occupied,
            everywhere,
        ),
        "free_iou": _binary_iou(predicted_free, truth_free, everywhere),
        "unknown_iou": _binary_iou(
            predicted_unknown,
            truth_unknown,
            everywhere,
        ),
        "occupied_f1": occupied_f1,
        "free_f1": free_f1,
        "unknown_f1": unknown_f1,
        "three_class_macro_f1": (occupied_f1 + free_f1 + unknown_f1) / 3.0,
        "boundary_chamfer_m": _boundary_chamfer_m(
            predicted_occupied,
            truth_occupied,
            everywhere,
            target_extent_m,
        ),
    }


def evidential_complete_metrics(
    prediction: dict[str, torch.Tensor],
    complete_target: torch.Tensor,
    observed_target: torch.Tensor,
    fov_support_target: torch.Tensor,
    *,
    target_extent_m: torch.Tensor,
    surface_tolerance_pixels: int,
    threshold: float = 0.5,
    labels: LabelValues = DEFAULT_LABELS,
) -> dict[str, torch.Tensor]:
    alpha = prediction["alpha_occupied"].float()
    beta = prediction["beta_free"].float()
    strength = alpha + beta
    occupancy_probability = alpha / strength
    predicted_occupied = occupancy_probability >= threshold
    truth_occupied = complete_target == labels.occupied
    truth_free = complete_target == labels.free
    valid = complete_target != labels.unknown
    if not torch.equal(fov_support_target.bool(), valid):
        raise ValueError("FOV support must equal valid FOV-complete cells")
    observed = (observed_target != labels.unknown) & valid
    guessed = (~observed) & valid
    observed_free = observed & truth_free
    observed_surface = observed & truth_occupied
    surface_band = (
        F.max_pool2d(
            observed_surface[:, None].float(),
            kernel_size=2 * int(surface_tolerance_pixels) + 1,
            stride=1,
            padding=int(surface_tolerance_pixels),
        )[:, 0]
        > 0
    )
    observed_free_supervised = observed_free & ~surface_band
    correct = predicted_occupied == truth_occupied
    confidence = (1.0 - 2.0 / strength).clamp(0.0, 1.0)
    runtime_support = prediction["runtime_fov_support"].bool()
    if runtime_support.shape != valid.shape:
        raise ValueError("runtime FOV support shape does not match target")

    def region_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.any():
            return values[mask].mean()
        return values.sum() * 0.0

    occupied_f1 = _binary_f1(predicted_occupied, truth_occupied, valid)
    free_f1 = _binary_f1(~predicted_occupied, truth_free, valid)
    tolerant_surface_hit = F.max_pool2d(
        predicted_occupied[:, None].float(),
        kernel_size=2 * int(surface_tolerance_pixels) + 1,
        stride=1,
        padding=int(surface_tolerance_pixels),
    )[:, 0]
    guessed_iou = _binary_iou(predicted_occupied, truth_occupied, guessed)
    guessed_baseline = _binary_iou(
        torch.ones_like(predicted_occupied),
        truth_occupied,
        guessed,
    )
    everywhere = torch.ones_like(valid)
    return {
        "runtime_support_iou": _binary_iou(
            runtime_support,
            valid,
            everywhere,
        ),
        "runtime_support_f1": _binary_f1(
            runtime_support,
            valid,
            everywhere,
        ),
        "occupied_iou": _binary_iou(predicted_occupied, truth_occupied, valid),
        "free_iou": _binary_iou(~predicted_occupied, truth_free, valid),
        "occupied_f1": occupied_f1,
        "free_f1": free_f1,
        "known_macro_f1": 0.5 * (occupied_f1 + free_f1),
        "occupancy_ece": _expected_calibration_error(
            occupancy_probability,
            truth_occupied,
            valid,
        ),
        "confidence_brier": region_mean(
            (confidence - correct.to(confidence.dtype)).square(),
            valid,
        ),
        "observed_accuracy": region_mean(correct.float(), observed),
        "observed_free_accuracy": region_mean(
            (~predicted_occupied).float(),
            observed_free_supervised,
        ),
        "observed_surface_recall_with_tolerance": region_mean(
            tolerant_surface_hit,
            observed_surface,
        ),
        "guessed_accuracy": region_mean(correct.float(), guessed),
        "guessed_occupied_iou": guessed_iou,
        "guessed_all_occupied_baseline_iou": guessed_baseline,
        "guessed_occupied_iou_gain_over_all_occupied": (
            guessed_iou - guessed_baseline
        ),
        "observed_confidence_ece": _expected_calibration_error(
            confidence,
            correct,
            observed,
        ),
        "guessed_confidence_ece": _expected_calibration_error(
            confidence,
            correct,
            guessed,
        ),
        "observed_mean_evidence_confidence": region_mean(confidence, observed),
        "guessed_mean_evidence_confidence": region_mean(confidence, guessed),
        "confidence_gap": (
            region_mean(confidence, observed)
            - region_mean(confidence, guessed)
        ),
        "observed_mean_uncertainty": region_mean(2.0 / strength, observed),
        "guessed_mean_uncertainty": region_mean(2.0 / strength, guessed),
        "boundary_chamfer_m": _boundary_chamfer_m(
            predicted_occupied,
            truth_occupied,
            valid,
            target_extent_m,
        ),
    }
