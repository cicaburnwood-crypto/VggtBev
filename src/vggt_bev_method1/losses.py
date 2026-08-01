from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from vggt_bev_method1.config import LabelValues, Supervision

DEFAULT_LABELS = LabelValues()


@dataclass(frozen=True)
class LossWeights:
    occupancy: float = 1.0
    observation: float = 0.5
    dice: float = 0.25
    single: float = 0.5
    merged: float = 0.5
    observed: float = 0.5
    complete: float = 0.5


@dataclass(frozen=True)
class ObservedModelLossWeights:
    known: float = 1.0
    free: float = 1.0
    surface: float = 0.35
    single: float = 0.5
    merged: float = 0.5


@dataclass(frozen=True)
class EvidentialModelLossWeights:
    observed_free: float = 1.0
    observed_surface: float = 0.35
    guessed_nll: float = 1.0
    guessed_overlap: float = 0.20
    observed_region: float = 1.0
    guessed_region: float = 0.25
    incorrect_evidence: float = 0.05
    observation_relation: float = 0.10
    calibration: float = 0.25
    guessed_incorrect_evidence_multiplier: float = 4.0
    guessed_calibration_multiplier: float = 8.0
    relation_margin: float = 0.10
    single: float = 0.5
    merged: float = 0.5


def _occupied_dice(
    logits: torch.Tensor,
    occupied: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if not valid.any():
        return logits.sum() * 0.0
    probability = logits[valid].sigmoid()
    truth = occupied[valid].to(probability.dtype)
    intersection = (probability * truth).sum()
    return 1.0 - (2.0 * intersection + 1.0) / (
        probability.sum() + truth.sum() + 1.0
    )


def branch_loss(
    prediction: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    supervision: Supervision,
    weights: LossWeights,
    labels: LabelValues = DEFAULT_LABELS,
) -> dict[str, torch.Tensor]:
    occupancy_logit = prediction["occupancy_logit"]
    if occupancy_logit.shape != target.shape:
        raise ValueError("occupancy prediction and target shapes do not match")
    occupied = target == labels.occupied
    valid = target != labels.unknown
    if valid.any():
        occupancy_loss = F.binary_cross_entropy_with_logits(
            occupancy_logit[valid],
            occupied[valid].to(occupancy_logit.dtype),
        )
    else:
        occupancy_loss = occupancy_logit.sum() * 0.0
    dice_loss = _occupied_dice(occupancy_logit, occupied, valid)

    observed_logit = prediction["observed_logit"]
    if observed_logit.shape != target.shape:
        raise ValueError("observation prediction and target shapes do not match")
    observation_loss = F.binary_cross_entropy_with_logits(
        observed_logit,
        valid.to(observed_logit.dtype),
    )

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


def dual_method1_loss(
    prediction: dict,
    batch: dict,
    *,
    supervision: Supervision,
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    if supervision == "joint":
        output: dict[str, torch.Tensor] = {}
        task_totals = {}
        for task in ("observed", "complete"):
            single = branch_loss(
                prediction["single"][task],
                batch[f"single_{task}_target"],
                supervision=task,
                weights=weights,
            )
            merged = branch_loss(
                prediction["merged"][task],
                batch[f"merged_{task}_target"],
                supervision=task,
                weights=weights,
            )
            task_total = (
                weights.single * single["loss"]
                + weights.merged * merged["loss"]
            )
            task_totals[task] = task_total
            output.update(
                {
                    f"{task}_single_{key}": value
                    for key, value in single.items()
                }
            )
            output.update(
                {
                    f"{task}_merged_{key}": value
                    for key, value in merged.items()
                }
            )
            output[f"{task}_loss"] = task_total
        output["loss"] = (
            weights.observed * task_totals["observed"]
            + weights.complete * task_totals["complete"]
        )
        return output

    single = branch_loss(
        prediction["single"],
        batch["single_target"],
        supervision=supervision,
        weights=weights,
    )
    merged = branch_loss(
        prediction["merged"],
        batch["merged_target"],
        supervision=supervision,
        weights=weights,
    )
    total = weights.single * single["loss"] + weights.merged * merged["loss"]
    output = {"loss": total}
    output.update({f"single_{key}": value for key, value in single.items()})
    output.update({f"merged_{key}": value for key, value in merged.items()})
    return output


def masked_target_classes(
    target: torch.Tensor,
    *,
    labels: LabelValues = DEFAULT_LABELS,
) -> torch.Tensor:
    """Map raster values to direct classes: unknown=0, free=1, occupied=2."""

    classes = torch.empty_like(target, dtype=torch.long)
    classes[target == labels.unknown] = 0
    classes[target == labels.free] = 1
    classes[target == labels.occupied] = 2
    valid = (
        (target == labels.unknown)
        | (target == labels.free)
        | (target == labels.occupied)
    )
    if not valid.all():
        raise ValueError("masked target contains an unsupported label value")
    return classes


def _categorical_occupied_dice(
    class_logits: torch.Tensor,
    target_classes: torch.Tensor,
) -> torch.Tensor:
    occupied_probability = class_logits.float().softmax(dim=1)[:, 2]
    truth = target_classes == 2
    intersection = (occupied_probability * truth).sum()
    return 1.0 - (2.0 * intersection + 1.0) / (
        occupied_probability.sum() + truth.sum() + 1.0
    )


def _local_minimum(values: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 0:
        raise ValueError("surface tolerance radius cannot be negative")
    if radius == 0:
        return values
    return -F.max_pool2d(
        -values[:, None],
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )[:, 0]


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


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return _local_maximum(mask.float(), radius) > 0


def _weighted_available_pair(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    first_available: bool,
    second_available: bool,
    first_weight: float,
    second_weight: float,
) -> torch.Tensor:
    denominator = (
        float(first_weight) * float(first_available)
        + float(second_weight) * float(second_available)
    )
    if denominator <= 0:
        return (first + second) * 0.0
    return (
        float(first_weight) * first * float(first_available)
        + float(second_weight) * second * float(second_available)
    ) / denominator


def _region_weighted_pair(
    observed_value: torch.Tensor,
    guessed_value: torch.Tensor,
    *,
    observed_weight: float,
    guessed_weight: float,
    guessed_scale: float,
) -> torch.Tensor:
    denominator = max(float(observed_weight + guessed_weight), 1e-6)
    return (
        float(observed_weight) * observed_value
        + float(guessed_weight) * float(guessed_scale) * guessed_value
    ) / denominator


def observed_extent_loss(
    prediction: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    weights: ObservedModelLossWeights,
    surface_tolerance_pixels: int,
) -> dict[str, torch.Tensor]:
    logits = prediction["class_logits"]
    if logits.shape != (target.shape[0], 3, *target.shape[1:]):
        raise ValueError("observed class logits and target shapes do not match")
    classes = masked_target_classes(target)
    log_probability = logits.float().log_softmax(dim=1)
    known = classes != 0
    free = classes == 1
    surface = classes == 2
    surface_band = _dilate(surface, int(surface_tolerance_pixels))
    free_supervised = free & ~surface_band

    known_logit = torch.logsumexp(logits.float()[:, 1:], dim=1) - logits.float()[:, 0]
    known_loss = F.binary_cross_entropy_with_logits(
        known_logit,
        known.to(known_logit.dtype),
    )
    free_nll = _mean_or_zero(-log_probability[:, 1], free_supervised)
    surface_nll = _mean_or_zero(
        _local_minimum(-log_probability[:, 2], int(surface_tolerance_pixels)),
        surface,
    )
    direct_loss = _weighted_available_pair(
        free_nll,
        surface_nll,
        first_available=bool(free_supervised.any()),
        second_available=bool(surface.any()),
        first_weight=weights.free,
        second_weight=weights.surface,
    )
    return {
        "loss": weights.known * known_loss + direct_loss,
        "known_loss": known_loss,
        "observed_free_nll": free_nll,
        "observed_surface_nll": surface_nll,
        "direct_content_loss": direct_loss,
        "known_fraction": known.float().mean(),
        "surface_tolerance_pixels": torch.tensor(
            int(surface_tolerance_pixels),
            device=logits.device,
        ),
    }


def observed_model_loss(
    prediction: dict,
    batch: dict,
    *,
    weights: ObservedModelLossWeights,
    surface_tolerance_single_pixels: int,
    surface_tolerance_merged_pixels: int,
) -> dict[str, torch.Tensor]:
    single = observed_extent_loss(
        prediction["single"],
        batch["single_fov_visible_target"],
        weights=weights,
        surface_tolerance_pixels=surface_tolerance_single_pixels,
    )
    merged = observed_extent_loss(
        prediction["merged"],
        batch["merged_fov_visible_target"],
        weights=weights,
        surface_tolerance_pixels=surface_tolerance_merged_pixels,
    )
    output = {
        "loss": weights.single * single["loss"] + weights.merged * merged["loss"]
    }
    output.update({f"single_{key}": value for key, value in single.items()})
    output.update({f"merged_{key}": value for key, value in merged.items()})
    return output


def _beta_kl_to_uniform(
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    strength = alpha + beta
    return (
        torch.lgamma(strength)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(strength))
        + (beta - 1.0) * (torch.digamma(beta) - torch.digamma(strength))
    )


def _mean_or_zero(
    values: torch.Tensor,
    mask: torch.Tensor,
    cell_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not bool(mask.any()):
        return values.sum() * 0.0
    if cell_weights is None:
        return values[mask].mean()
    selected_weights = cell_weights[mask]
    return (values[mask] * selected_weights).sum() / selected_weights.sum().clamp_min(
        1e-6
    )


def _target_regions(
    complete_target: torch.Tensor,
    observed_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if complete_target.shape != observed_target.shape:
        raise ValueError("complete and masked-observed BEV shapes do not match")
    allowed = (
        (complete_target == DEFAULT_LABELS.unknown)
        | (complete_target == DEFAULT_LABELS.free)
        | (complete_target == DEFAULT_LABELS.occupied)
    )
    observed_allowed = (
        (observed_target == DEFAULT_LABELS.unknown)
        | (observed_target == DEFAULT_LABELS.free)
        | (observed_target == DEFAULT_LABELS.occupied)
    )
    if not bool(allowed.all()) or not bool(observed_allowed.all()):
        raise ValueError("FOV target contains an unsupported label")
    valid = complete_target != DEFAULT_LABELS.unknown
    observed = (observed_target != DEFAULT_LABELS.unknown) & valid
    guessed = (observed_target == DEFAULT_LABELS.unknown) & valid
    if bool(((observed_target != DEFAULT_LABELS.unknown) & ~valid).any()):
        raise ValueError("masked-observed BEV has known cells outside complete GT")
    if bool((observed & (observed_target != complete_target)).any()):
        raise ValueError("masked-observed labels disagree with complete GT")
    occupied = complete_target == DEFAULT_LABELS.occupied
    return valid, observed, guessed, occupied


def evidential_extent_loss(
    prediction: dict[str, torch.Tensor],
    complete_target: torch.Tensor,
    observed_target: torch.Tensor,
    *,
    guessed_class_weights: torch.Tensor,
    weights: EvidentialModelLossWeights,
    regularizer_scale: float,
    guessed_supervision_scale: float,
    surface_tolerance_pixels: int,
) -> dict[str, torch.Tensor]:
    alpha = prediction["alpha_occupied"].float()
    beta = prediction["beta_free"].float()
    if alpha.shape != complete_target.shape or beta.shape != complete_target.shape:
        raise ValueError("evidential parameters and complete target shapes do not match")
    valid, observed, guessed, occupied = _target_regions(
        complete_target,
        observed_target,
    )
    strength = alpha + beta
    probability = alpha / strength

    if guessed_class_weights.numel() != 2:
        raise ValueError("guessed class weights must be [free, occupied]")
    guessed_class_weights = guessed_class_weights.to(alpha)
    guessed_cell_weights = torch.where(
        occupied,
        guessed_class_weights[1],
        guessed_class_weights[0],
    )

    occupied_nll_map = torch.digamma(strength) - torch.digamma(alpha)
    free_nll_map = torch.digamma(strength) - torch.digamma(beta)
    expected_nll_map = torch.where(occupied, occupied_nll_map, free_nll_map)
    observed_free = observed & ~occupied
    observed_surface = observed & occupied
    surface_band = _dilate(observed_surface, int(surface_tolerance_pixels))
    observed_free_supervised = observed_free & ~surface_band
    observed_free_nll = _mean_or_zero(free_nll_map, observed_free_supervised)
    observed_surface_nll = _mean_or_zero(
        _local_minimum(occupied_nll_map, int(surface_tolerance_pixels)),
        observed_surface,
    )
    guessed_nll = _mean_or_zero(
        expected_nll_map,
        guessed,
        guessed_cell_weights,
    )
    observed_direct_loss = _weighted_available_pair(
        observed_free_nll,
        observed_surface_nll,
        first_available=bool(observed_free_supervised.any()),
        second_available=bool(observed_surface.any()),
        first_weight=weights.observed_free,
        second_weight=weights.observed_surface,
    )

    def region_dice(probability_map: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        if not bool(guessed.any()):
            return probability_map.sum() * 0.0
        selected_probability = probability_map[guessed]
        selected_truth = truth[guessed].to(selected_probability.dtype)
        return 1.0 - (
            2.0 * (selected_probability * selected_truth).sum() + 1.0
        ) / (selected_probability.sum() + selected_truth.sum() + 1.0)

    guessed_occupied_dice = region_dice(probability, occupied)
    guessed_free_dice = region_dice(1.0 - probability, ~occupied)
    guessed_macro_dice = _weighted_available_pair(
        guessed_occupied_dice,
        guessed_free_dice,
        first_available=bool((guessed & occupied).any()),
        second_available=bool((guessed & ~occupied).any()),
        first_weight=1.0,
        second_weight=1.0,
    )
    guessed_completion_loss = (
        weights.guessed_nll * guessed_nll
        + weights.guessed_overlap * guessed_macro_dice
    )
    guessed_supervision_scale = float(
        max(0.0, min(1.0, guessed_supervision_scale))
    )
    content_loss = _region_weighted_pair(
        observed_direct_loss,
        guessed_completion_loss,
        observed_weight=weights.observed_region,
        guessed_weight=weights.guessed_region,
        guessed_scale=guessed_supervision_scale,
    )

    free_wrong_map = _beta_kl_to_uniform(alpha, torch.ones_like(beta))
    occupied_wrong_map = _beta_kl_to_uniform(torch.ones_like(alpha), beta)
    wrong_map = torch.where(occupied, occupied_wrong_map, free_wrong_map)
    observed_free_wrong = _mean_or_zero(
        free_wrong_map,
        observed_free_supervised,
    )
    observed_surface_wrong = _mean_or_zero(
        _local_minimum(occupied_wrong_map, int(surface_tolerance_pixels)),
        observed_surface,
    )
    observed_wrong = _weighted_available_pair(
        observed_free_wrong,
        observed_surface_wrong,
        first_available=bool(observed_free_supervised.any()),
        second_available=bool(observed_surface.any()),
        first_weight=weights.observed_free,
        second_weight=weights.observed_surface,
    )
    guessed_wrong = _mean_or_zero(wrong_map, guessed)

    predicted_occupied = probability >= 0.5
    correct = (predicted_occupied == occupied).detach().to(alpha.dtype)
    confidence = (1.0 - 2.0 / strength).clamp(0.0, 1.0)
    calibration_map = (confidence - correct).square()
    predicted_occupied_float = predicted_occupied.detach().to(confidence.dtype)
    observed_free_calibration = _mean_or_zero(
        (
            confidence
            - (~predicted_occupied).detach().to(confidence.dtype)
        ).square(),
        observed_free_supervised,
    )
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
    observed_surface_calibration = _mean_or_zero(
        (
            tolerant_surface_confidence
            - tolerant_surface_correct.detach()
        ).square(),
        observed_surface,
    )
    observed_calibration = _weighted_available_pair(
        observed_free_calibration,
        observed_surface_calibration,
        first_available=bool(observed_free_supervised.any()),
        second_available=bool(observed_surface.any()),
        first_weight=weights.observed_free,
        second_weight=weights.observed_surface,
    )
    guessed_calibration = _mean_or_zero(calibration_map, guessed)
    observed_free_confidence = _mean_or_zero(
        confidence,
        observed_free_supervised,
    )
    observed_surface_confidence = _mean_or_zero(
        tolerant_surface_confidence,
        observed_surface,
    )
    observed_confidence = _weighted_available_pair(
        observed_free_confidence,
        observed_surface_confidence,
        first_available=bool(observed_free_supervised.any()),
        second_available=bool(observed_surface.any()),
        first_weight=1.0,
        second_weight=1.0,
    )
    guessed_confidence = _mean_or_zero(confidence, guessed)
    if bool(observed.any()) and bool(guessed.any()):
        relation = guessed_supervision_scale * F.relu(
            weights.relation_margin
            - (observed_confidence - guessed_confidence)
        )
    else:
        relation = confidence.sum() * 0.0

    regularizer_scale = float(max(0.0, min(1.0, regularizer_scale)))
    denominator = max(weights.observed_region + weights.guessed_region, 1e-6)
    direct_confidence_loss = regularizer_scale * (
        weights.observed_region
        / denominator
        * (
            weights.incorrect_evidence * observed_wrong
            + weights.calibration * observed_calibration
        )
    )
    guessed_confidence_loss = regularizer_scale * (
        weights.guessed_region
        * guessed_supervision_scale
        / denominator
        * (
            weights.incorrect_evidence
            * weights.guessed_incorrect_evidence_multiplier
            * guessed_wrong
            + weights.calibration
            * weights.guessed_calibration_multiplier
            * guessed_calibration
        )
        + weights.observation_relation * relation
    )
    direct_objective = (
        weights.observed_region / denominator * observed_direct_loss
        + direct_confidence_loss
    )
    guessed_objective = (
        weights.guessed_region
        * guessed_supervision_scale
        / denominator
        * guessed_completion_loss
        + guessed_confidence_loss
    )
    total = content_loss + direct_confidence_loss + guessed_confidence_loss
    return {
        "loss": total,
        "direct_objective": direct_objective,
        "guessed_objective": guessed_objective,
        "content_loss": content_loss,
        "observed_direct_loss": observed_direct_loss,
        "guessed_completion_loss": guessed_completion_loss,
        "observed_free_nll": observed_free_nll,
        "observed_surface_nll": observed_surface_nll,
        "guessed_evidential_nll": guessed_nll,
        "guessed_macro_dice_loss": guessed_macro_dice,
        "incorrect_evidence_loss": observed_wrong + guessed_wrong,
        "observed_incorrect_evidence_loss": observed_wrong,
        "guessed_incorrect_evidence_loss": guessed_wrong,
        "observation_relation_loss": relation,
        "confidence_calibration_loss": observed_calibration + guessed_calibration,
        "observed_confidence_calibration_loss": observed_calibration,
        "observed_free_confidence_calibration_loss": (
            observed_free_calibration
        ),
        "observed_surface_confidence_calibration_loss": (
            observed_surface_calibration
        ),
        "guessed_confidence_calibration_loss": guessed_calibration,
        "observed_mean_strength": _mean_or_zero(strength, observed),
        "guessed_mean_strength": _mean_or_zero(strength, guessed),
        "observed_mean_uncertainty": _mean_or_zero(2.0 / strength, observed),
        "guessed_mean_uncertainty": _mean_or_zero(2.0 / strength, guessed),
        "observed_mean_confidence": observed_confidence,
        "observed_free_mean_confidence": observed_free_confidence,
        "observed_surface_mean_confidence": observed_surface_confidence,
        "guessed_mean_confidence": guessed_confidence,
        "confidence_gap": observed_confidence - guessed_confidence,
        "observed_fraction": observed.float().mean(),
        "guessed_fraction": guessed.float().mean(),
        "valid_fraction": valid.float().mean(),
    }


def complete_evidential_model_loss(
    prediction: dict,
    batch: dict,
    *,
    guessed_class_weights: torch.Tensor,
    weights: EvidentialModelLossWeights,
    regularizer_scale: float,
    guessed_supervision_scale: float,
    surface_tolerance_single_pixels: int,
    surface_tolerance_merged_pixels: int,
) -> dict[str, torch.Tensor]:
    for branch in ("single", "merged"):
        valid = batch[f"{branch}_fov_complete_target"] != DEFAULT_LABELS.unknown
        if not torch.equal(
            batch[f"{branch}_fov_support_target"].bool(),
            valid,
        ):
            raise ValueError(
                f"{branch} FOV support must equal valid FOV-complete cells"
            )
    single = evidential_extent_loss(
        prediction["single"],
        batch["single_fov_complete_target"],
        batch["single_fov_visible_target"],
        guessed_class_weights=guessed_class_weights,
        weights=weights,
        regularizer_scale=regularizer_scale,
        guessed_supervision_scale=guessed_supervision_scale,
        surface_tolerance_pixels=surface_tolerance_single_pixels,
    )
    merged = evidential_extent_loss(
        prediction["merged"],
        batch["merged_fov_complete_target"],
        batch["merged_fov_visible_target"],
        guessed_class_weights=guessed_class_weights,
        weights=weights,
        regularizer_scale=regularizer_scale,
        guessed_supervision_scale=guessed_supervision_scale,
        surface_tolerance_pixels=surface_tolerance_merged_pixels,
    )
    output = {}
    for key in ("loss", "direct_objective", "guessed_objective"):
        output[key] = (
            weights.single * single[key] + weights.merged * merged[key]
        )
    output.update({f"single_{key}": value for key, value in single.items()})
    output.update({f"merged_{key}": value for key, value in merged.items()})
    return output
