from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from vggt_bev_method1.config import LabelValues
from vggt_bev_method1.data.p1b_targets import p1b_region_masks
from vggt_bev_method1.models.p1b_probability import ProbabilityModel


@dataclass(frozen=True)
class P1BLossWeights:
    observed_gate_pixel: float = 1.0
    guessed_pixel: float = 1.0
    guessed_surface: float = 0.5
    guessed_free: float = 0.35
    guessed_visible_surface: float = 0.40
    guessed_hidden_occupied: float = 0.25
    wrong_evidence_kl: float = 0.005
    support_bce: float = 0.5
    support_dice: float = 0.5


DEFAULT_P1B_LOSS_WEIGHTS = P1BLossWeights()
DEFAULT_LABEL_VALUES = LabelValues()


def _per_sample_ratio_mean(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    """Average one scalar ratio per available sample."""

    numerator = numerator.flatten(1).sum(dim=1)
    denominator = denominator.flatten(1).sum(dim=1)
    available = denominator > 0
    ratios = numerator / denominator.clamp_min(1e-6)
    weights = available.to(ratios.dtype)
    return (ratios * weights).sum() / weights.sum().clamp_min(1.0)


def _binary_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 0:
        raise ValueError("morphology radius cannot be negative")
    if radius == 0:
        return mask.bool()
    kernel = 2 * radius + 1
    return F.max_pool2d(
        mask.to(torch.float32).unsqueeze(1),
        kernel,
        stride=1,
        padding=radius,
    ).squeeze(1) > 0.5


def _binary_erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 0:
        raise ValueError("morphology radius cannot be negative")
    if radius == 0:
        return mask.bool()
    return ~_binary_dilate(~mask.bool(), radius)


def _binary_boundary_band(
    mask: torch.Tensor,
    radius: int,
    domain: torch.Tensor,
) -> torch.Tensor:
    """Return a symmetric GT boundary band restricted to reliable pixels."""

    if not (mask.shape == domain.shape):
        raise ValueError("boundary mask and domain must align")
    stable_domain = _binary_erode(domain.bool(), radius)
    return (_binary_dilate(mask, radius) ^ _binary_erode(mask, radius)) & (
        stable_domain
    )


def _soft_region_dice(
    probability: torch.Tensor,
    truth: torch.Tensor,
    domain: torch.Tensor,
) -> torch.Tensor:
    if not (probability.shape == truth.shape == domain.shape):
        raise ValueError("Dice probability, truth and domain must align")
    weight = domain.to(probability.dtype)
    target = truth.to(probability.dtype)
    numerator = (2.0 * probability * target * weight).flatten(1).sum(dim=1)
    denominator = ((probability + target) * weight).flatten(1).sum(dim=1)
    return (1.0 - (numerator + 1.0) / (denominator + 1.0)).mean()


def _soft_contour_dice(
    probability: torch.Tensor,
    truth: torch.Tensor,
    domain: torch.Tensor,
) -> torch.Tensor:
    """Align predicted and GT contours without adding an output channel."""

    probability_4d = probability.unsqueeze(1)
    predicted_dilated = F.max_pool2d(probability_4d, 3, stride=1, padding=1)
    predicted_eroded = -F.max_pool2d(
        -probability_4d,
        3,
        stride=1,
        padding=1,
    )
    predicted_edge = (predicted_dilated - predicted_eroded).squeeze(1)
    target_edge = _binary_boundary_band(truth.bool(), 1, domain.bool())
    return _soft_region_dice(
        predicted_edge,
        target_edge,
        _binary_erode(domain.bool(), 1),
    )


def p1b_fov_support_loss(
    prediction: dict,
    support_target: torch.Tensor,
    *,
    gt_valid_mask: torch.Tensor | None = None,
    variant: str = "balanced_bce_dice",
    bce_weight: float = 0.65,
    dice_weight: float = 0.35,
    tversky_weight: float = 0.0,
    boundary_weight: float = 0.0,
    boundary_radius: int = 2,
    interior_weight: float = 0.25,
    edge_weight: float = 0.45,
    contour_weight: float = 0.20,
    region_dice_weight: float = 0.10,
    tversky_false_positive_weight: float = 0.70,
    tversky_false_negative_weight: float = 0.30,
) -> dict[str, torch.Tensor]:
    """Isolated Merged FOV-support objective.

    This function never reads complete/visible semantics, Observed Gate or the
    Guessed Expert.  ``balanced_bce_dice`` is the throughput-oriented scheme;
    ``boundary_tversky`` adds an explicit target-boundary term and asymmetric
    overlap penalty while preserving the same FOV-only supervision contract.
    """

    if variant not in (
        "balanced_bce_dice",
        "balanced_bce_dice_boundary",
        "boundary_tversky",
        "role_balanced_contour",
    ):
        raise ValueError(f"unsupported FOV support loss variant: {variant}")
    support_logit = prediction["fov_support_logit"].float()
    if support_logit.shape != support_target.shape:
        raise ValueError("FOV support prediction and target must align")
    valid = (
        torch.ones_like(support_target, dtype=torch.bool)
        if gt_valid_mask is None
        else gt_valid_mask.bool()
    )
    if valid.shape != support_target.shape:
        raise ValueError("FOV support validity mask must align with target")
    truth = support_target.bool() & valid
    positive = truth & valid
    negative = ~truth & valid
    target_float = truth.to(support_logit.dtype)
    pixel_bce = F.binary_cross_entropy_with_logits(
        support_logit,
        target_float,
        reduction="none",
    )
    positive_bce = _per_sample_masked_mean(pixel_bce, positive)
    negative_bce = _per_sample_masked_mean(pixel_bce, negative)
    balanced_bce = 0.5 * (positive_bce + negative_bce)

    probability = torch.sigmoid(support_logit)
    valid_float = valid.to(probability.dtype)
    truth_float = target_float * valid_float
    probability_valid = probability * valid_float
    intersection = probability_valid * truth_float
    dice = 1.0 - _per_sample_ratio_mean(
        2.0 * intersection + valid_float / valid.flatten(1).sum(1)
        .clamp_min(1)
        .view(-1, 1, 1),
        probability_valid + truth_float + valid_float / valid.flatten(1).sum(1)
        .clamp_min(1)
        .view(-1, 1, 1),
    )

    false_positive = probability_valid * (1.0 - truth_float)
    false_negative = (1.0 - probability) * truth_float
    tversky = 1.0 - _per_sample_ratio_mean(
        intersection + 1e-6 * valid_float,
        intersection
        + tversky_false_positive_weight * false_positive
        + tversky_false_negative_weight * false_negative
        + 1e-6 * valid_float,
    )

    if boundary_radius < 1:
        raise ValueError("support_boundary_radius must be at least one")
    kernel = 2 * int(boundary_radius) + 1
    truth_4d = target_float.unsqueeze(1)
    dilated = F.max_pool2d(truth_4d, kernel, stride=1, padding=boundary_radius)
    eroded = -F.max_pool2d(
        -truth_4d,
        kernel,
        stride=1,
        padding=boundary_radius,
    )
    boundary = ((dilated - eroded) > 0).squeeze(1) & valid
    boundary_bce = _per_sample_masked_mean(pixel_bce, boundary)

    role_boundary = _binary_boundary_band(truth, boundary_radius, valid)
    interior_bce, interior_negative_bce, interior_positive_bce = (
        _class_balanced_binary_bce(
            support_logit,
            positive & ~role_boundary,
            negative & ~role_boundary,
        )
    )
    edge_bce, edge_negative_bce, edge_positive_bce = (
        _class_balanced_binary_bce(
            support_logit,
            positive & role_boundary,
            negative & role_boundary,
        )
    )
    contour_dice = _soft_contour_dice(probability, truth, valid)
    region_dice = _soft_region_dice(probability, truth, valid)

    if variant == "balanced_bce_dice":
        loss = bce_weight * balanced_bce + dice_weight * dice
    elif variant == "balanced_bce_dice_boundary":
        loss = (
            bce_weight * balanced_bce
            + dice_weight * dice
            + boundary_weight * boundary_bce
        )
    elif variant == "boundary_tversky":
        loss = (
            bce_weight * balanced_bce
            + tversky_weight * tversky
            + boundary_weight * boundary_bce
        )
    else:
        loss = (
            interior_weight * interior_bce
            + edge_weight * edge_bce
            + contour_weight * contour_dice
            + region_dice_weight * region_dice
        )

    hard = probability >= 0.5
    tp = (hard & truth & valid).sum().to(torch.float32)
    fp = (hard & ~truth & valid).sum().to(torch.float32)
    fn = (~hard & truth & valid).sum().to(torch.float32)
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    iou = tp / (tp + fp + fn).clamp_min(1.0)
    return {
        "loss": loss,
        "support_objective": loss,
        "support_balanced_bce_loss": balanced_bce,
        "support_positive_bce_loss": positive_bce,
        "support_negative_bce_loss": negative_bce,
        "support_dice_loss": dice,
        "support_tversky_loss": tversky,
        "support_boundary_bce_loss": boundary_bce,
        "support_interior_bce_loss": interior_bce,
        "support_interior_positive_bce_loss": interior_positive_bce,
        "support_interior_negative_bce_loss": interior_negative_bce,
        "support_edge_bce_loss": edge_bce,
        "support_edge_positive_bce_loss": edge_positive_bce,
        "support_edge_negative_bce_loss": edge_negative_bce,
        "support_contour_dice_loss": contour_dice,
        "support_region_dice_loss": region_dice,
        "support_edge_fraction": role_boundary.float().mean(),
        "support_precision": precision,
        "support_recall": recall,
        "support_iou": iou,
        "support_fraction": truth.float().mean(),
        "support_valid_fraction": valid.float().mean(),
    }


def p1b_routing_geometry_loss(
    prediction: dict,
    complete_target: torch.Tensor,
    visible_target: torch.Tensor,
    support_target: torch.Tensor,
    *,
    gt_valid_mask: torch.Tensor | None = None,
    observed_gate_weight: float = 1.0,
    support_weight: float = 1.0,
    gate_loss_variant: str = "balanced_bce",
    gate_boundary_radius: int = 3,
    gate_interior_weight: float = 0.20,
    gate_edge_weight: float = 0.50,
    gate_contour_weight: float = 0.20,
    gate_dice_weight: float = 0.10,
    **support_arguments,
) -> dict[str, torch.Tensor]:
    """Train only the two Merged routing-geometry outputs.

    The Observed Gate target is the exact masked-GT observed-free mask.  Its
    positive and negative domains are averaged 1:1 per sample.  No Guessed,
    occupancy, evidence, confidence or surface objective is evaluated.
    """

    masks = p1b_region_masks(
        complete_target,
        visible_target,
        support_target,
    )
    valid = (
        torch.ones_like(masks.valid)
        if gt_valid_mask is None
        else gt_valid_mask.bool()
    )
    if valid.shape != masks.valid.shape:
        raise ValueError("routing-geometry validity mask must align with target")
    support = p1b_fov_support_loss(
        prediction,
        support_target,
        gt_valid_mask=valid,
        **support_arguments,
    )
    if observed_gate_weight == 0.0:
        # During the FOV-only curriculum the Gate objective is exactly zero.
        # Keep a zero-valued graph anchor for DDP, but avoid the otherwise
        # wasted full-resolution morphology, BCE, Dice and metric kernels.
        zero = prediction["observed_gate_logit"].float().sum() * 0.0
        total = support_weight * support["loss"] + zero
        return {
            **support,
            "loss": total,
            "routing_geometry_objective": total,
            "observed_gate_objective": zero,
            "observed_gate_balanced_bce": zero,
            "observed_gate_guessed_bce": zero,
            "observed_gate_free_bce": zero,
            "observed_gate_interior_bce": zero,
            "observed_gate_interior_positive_bce": zero,
            "observed_gate_interior_negative_bce": zero,
            "observed_gate_edge_bce": zero,
            "observed_gate_edge_positive_bce": zero,
            "observed_gate_edge_negative_bce": zero,
            "observed_gate_contour_dice": zero,
            "observed_gate_region_dice": zero,
            "observed_gate_edge_fraction": zero,
            "observed_gate_precision": zero,
            "observed_gate_recall": zero,
            "observed_gate_iou": zero,
        }
    observed_free = masks.observed_free & masks.valid & valid
    guessed = masks.guessed & masks.valid & valid
    gate_bce, gate_negative_bce, gate_positive_bce = (
        _class_balanced_binary_bce(
            prediction["observed_gate_logit"].float(),
            observed_free,
            guessed,
        )
    )
    if gate_loss_variant not in ("balanced_bce", "role_balanced_contour"):
        raise ValueError(f"unsupported Observed Gate loss: {gate_loss_variant}")
    support_domain = masks.valid & valid
    fov_boundary = _binary_boundary_band(
        support_domain,
        gate_boundary_radius,
        valid,
    )
    fov_exclusion = _binary_dilate(fov_boundary, gate_boundary_radius)
    gate_domain = support_domain & ~fov_exclusion
    gate_boundary = _binary_boundary_band(
        observed_free,
        gate_boundary_radius,
        valid,
    ) & gate_domain
    gate_interior_bce, gate_interior_negative_bce, gate_interior_positive_bce = (
        _class_balanced_binary_bce(
            prediction["observed_gate_logit"].float(),
            observed_free & gate_domain & ~gate_boundary,
            guessed & gate_domain & ~gate_boundary,
        )
    )
    gate_edge_bce, gate_edge_negative_bce, gate_edge_positive_bce = (
        _class_balanced_binary_bce(
            prediction["observed_gate_logit"].float(),
            observed_free & gate_boundary,
            guessed & gate_boundary,
        )
    )
    gate_probability = torch.sigmoid(prediction["observed_gate_logit"].float())
    gate_contour_dice = _soft_contour_dice(
        gate_probability,
        observed_free,
        gate_domain,
    )
    gate_region_dice = _soft_region_dice(
        gate_probability,
        observed_free,
        support_domain,
    )
    if gate_loss_variant == "role_balanced_contour":
        gate_loss = (
            gate_interior_weight * gate_interior_bce
            + gate_edge_weight * gate_edge_bce
            + gate_contour_weight * gate_contour_dice
            + gate_dice_weight * gate_region_dice
        )
    else:
        gate_loss = gate_bce
    total = observed_gate_weight * gate_loss + support_weight * support["loss"]

    gate_hard = gate_probability >= 0.5
    gate_tp = (gate_hard & observed_free).sum().to(torch.float32)
    gate_fp = (gate_hard & guessed).sum().to(torch.float32)
    gate_fn = (~gate_hard & observed_free).sum().to(torch.float32)
    return {
        **support,
        "loss": total,
        "routing_geometry_objective": total,
        "observed_gate_objective": observed_gate_weight * gate_loss,
        "observed_gate_balanced_bce": gate_bce,
        "observed_gate_guessed_bce": gate_negative_bce,
        "observed_gate_free_bce": gate_positive_bce,
        "observed_gate_interior_bce": gate_interior_bce,
        "observed_gate_interior_positive_bce": gate_interior_positive_bce,
        "observed_gate_interior_negative_bce": gate_interior_negative_bce,
        "observed_gate_edge_bce": gate_edge_bce,
        "observed_gate_edge_positive_bce": gate_edge_positive_bce,
        "observed_gate_edge_negative_bce": gate_edge_negative_bce,
        "observed_gate_contour_dice": gate_contour_dice,
        "observed_gate_region_dice": gate_region_dice,
        "observed_gate_edge_fraction": gate_boundary.float().mean(),
        "observed_gate_precision": gate_tp / (gate_tp + gate_fp).clamp_min(1.0),
        "observed_gate_recall": gate_tp / (gate_tp + gate_fn).clamp_min(1.0),
        "observed_gate_iou": gate_tp
        / (gate_tp + gate_fp + gate_fn).clamp_min(1.0),
    }


def wrong_evidence_kl_weight(
    step: int,
    total_steps: int,
    *,
    maximum: float = 0.01,
    zero_fraction: float = 0.20,
    ramp_fraction: float = 0.10,
) -> float:
    if total_steps <= 0 or step < 0:
        raise ValueError("training steps must be non-negative and total positive")
    progress = min(float(step) / float(total_steps), 1.0)
    if progress <= zero_fraction:
        return 0.0
    if ramp_fraction <= 0.0:
        return float(maximum)
    return float(maximum) * min(
        (progress - zero_fraction) / ramp_fraction,
        1.0,
    )


def hidden_occupied_supervision_weight(
    step: int,
    total_steps: int,
    *,
    zero_fraction: float = 0.10,
    ramp_fraction: float = 0.15,
) -> float:
    """Delay only hidden occupied completion while preserving direct edges."""

    return wrong_evidence_kl_weight(
        step,
        total_steps,
        maximum=1.0,
        zero_fraction=zero_fraction,
        ramp_fraction=ramp_fraction,
    )


def _per_sample_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    flat_values = values.flatten(1)
    flat_mask = mask.flatten(1).to(values.dtype)
    denominator = flat_mask.sum(dim=1)
    available = denominator > 0
    per_sample = (flat_values * flat_mask).sum(dim=1) / denominator.clamp_min(1.0)
    available_weight = available.to(values.dtype)
    return (per_sample * available_weight).sum() / available_weight.sum().clamp_min(
        1.0
    )


def _assert_no_overlap(overlap: torch.Tensor, message: str) -> None:
    """Keep target-contract checks without synchronizing CUDA with the host."""

    valid = ~overlap.any()
    if valid.device.type == "cuda":
        torch._assert_async(valid, message)
    elif not bool(valid):
        raise ValueError(message)


def _weighted_per_sample_group_mean(
    values: torch.Tensor,
    masks: tuple[torch.Tensor, ...],
    weights: tuple[float, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Average each exact GT subset before applying explicit task weights.

    Missing subsets are skipped and the remaining weights are normalized per
    sample.  In particular, a frame with no visible occupied surface creates
    neither a positive surface loss nor a false surface-negative target.
    """

    if not masks or len(masks) != len(weights):
        raise ValueError("group masks and weights must be non-empty and aligned")
    if any(mask.shape != values.shape for mask in masks):
        raise ValueError("group masks and loss values must align")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("group weights cannot be negative")
    stacked = torch.stack(masks, dim=1)
    _assert_no_overlap(
        stacked.sum(dim=1) > 1,
        "loss groups must be mutually exclusive",
    )
    flat_values = values.flatten(1)
    flat_masks = stacked.flatten(2).to(values.dtype)
    counts = flat_masks.sum(dim=2)
    present = counts > 0
    group_means = (
        (flat_masks * flat_values[:, None, :]).sum(dim=2)
        / counts.clamp_min(1.0)
    )
    configured = torch.tensor(
        weights,
        device=values.device,
        dtype=values.dtype,
    ).unsqueeze(0)
    effective = configured * present.to(values.dtype)
    denominator = effective.sum(dim=1)
    per_sample = (
        group_means * effective
    ).sum(dim=1) / denominator.clamp_min(torch.finfo(values.dtype).eps)
    available = denominator > 0
    available_weight = available.to(values.dtype)
    combined = (per_sample * available_weight).sum() / available_weight.sum().clamp_min(
        1.0
    )
    diagnostics = []
    for group_index in range(len(masks)):
        group_available = present[:, group_index]
        group_available_weight = group_available.to(values.dtype)
        diagnostics.append(
            (
                group_means[:, group_index] * group_available_weight
            ).sum()
            / group_available_weight.sum().clamp_min(1.0)
        )
    return combined, tuple(diagnostics)


def _expert_cell_losses(
    expert: dict[str, torch.Tensor],
    probability_model: ProbabilityModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    if probability_model == "evidential":
        alpha = expert["alpha_occupied"].float()
        beta = expert["beta_free"].float()
        strength = alpha + beta
        return (
            torch.digamma(strength) - torch.digamma(alpha),
            torch.digamma(strength) - torch.digamma(beta),
        )
    if probability_model == "bce":
        logit = expert["occupancy_logit"].float()
        return F.softplus(-logit), F.softplus(logit)
    raise ValueError(f"unsupported probability model: {probability_model}")


def _beta_kl_to_uniform(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    strength = alpha + beta
    return (
        torch.lgamma(strength)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(strength))
        + (beta - 1.0) * (torch.digamma(beta) - torch.digamma(strength))
    )


def _wrong_evidence_map(
    expert: dict[str, torch.Tensor],
    occupied: torch.Tensor,
) -> torch.Tensor:
    alpha = expert["alpha_occupied"].float()
    beta = expert["beta_free"].float()
    wrong_if_free = _beta_kl_to_uniform(alpha, torch.ones_like(beta))
    wrong_if_occupied = _beta_kl_to_uniform(torch.ones_like(alpha), beta)
    return torch.where(occupied, wrong_if_occupied, wrong_if_free)


def _class_balanced_binary_bce(
    logit: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-pixel Gate BCE balanced only across its two original classes."""

    if not (logit.shape == positive.shape == negative.shape):
        raise ValueError("binary Gate logits and masks must align")
    _assert_no_overlap(
        positive & negative,
        "binary Gate positive and negative masks overlap",
    )
    target = positive.to(logit.dtype)
    pixel_bce = F.binary_cross_entropy_with_logits(
        logit.float(), target.float(), reduction="none"
    )
    class_masks = torch.stack((negative, positive), dim=1)
    flat_bce = pixel_bce.flatten(1)
    flat_masks = class_masks.flatten(2).to(pixel_bce.dtype)
    denominators = flat_masks.sum(dim=2)
    present = denominators > 0
    per_class_per_sample = (
        flat_masks * flat_bce[:, None, :]
    ).sum(dim=2) / denominators.clamp_min(1.0)
    per_sample = (
        per_class_per_sample * present.to(pixel_bce.dtype)
    ).sum(dim=1) / present.sum(dim=1).clamp_min(1).to(pixel_bce.dtype)
    available_samples = present.any(dim=1)
    available_weight = available_samples.to(pixel_bce.dtype)
    balanced = (per_sample * available_weight).sum() / available_weight.sum().clamp_min(
        1.0
    )
    negative_loss = _per_sample_masked_mean(pixel_bce, negative)
    positive_loss = _per_sample_masked_mean(pixel_bce, positive)
    return balanced, negative_loss, positive_loss


def p1b_bev_loss(
    prediction: dict,
    complete_target: torch.Tensor,
    visible_target: torch.Tensor,
    support_target: torch.Tensor,
    *,
    gt_valid_mask: torch.Tensor | None = None,
    probability_model: ProbabilityModel,
    weights: P1BLossWeights = DEFAULT_P1B_LOSS_WEIGHTS,
    wrong_evidence_scale: float = 1.0,
    hidden_occupied_scale: float = 1.0,
    labels: LabelValues = DEFAULT_LABEL_VALUES,
) -> dict[str, torch.Tensor]:
    masks = p1b_region_masks(
        complete_target,
        visible_target,
        support_target,
        labels=labels,
    )
    if gt_valid_mask is None:
        gt_valid = torch.ones_like(masks.valid)
    else:
        if gt_valid_mask.shape != complete_target.shape:
            raise ValueError("gt_valid_mask must align with the BEV target")
        gt_valid = gt_valid_mask.bool()
    # Scene geometry determines the hard loss domain only. Camera-FOV support,
    # masked visibility, Surface and Guessed targets are built exactly as in
    # the pre-Void pipeline, then every BEV loss is restricted by this mask.
    semantic_gt_valid = masks.valid & gt_valid
    valid_observed_free = masks.observed_free & semantic_gt_valid
    valid_guessed = masks.guessed & semantic_gt_valid
    valid_guessed_free = masks.guessed_free & semantic_gt_valid
    valid_visible_surface = masks.visible_surface & semantic_gt_valid
    valid_hidden_guessed_occupied = (
        masks.hidden_guessed_occupied & semantic_gt_valid
    )
    guessed = prediction["guessed"]
    guessed_occupied_loss, guessed_free_loss = _expert_cell_losses(
        guessed, probability_model
    )
    guessed_map = torch.where(
        masks.occupied,
        guessed_occupied_loss,
        guessed_free_loss,
    )
    guessed_group_weights = (
        weights.guessed_free,
        weights.guessed_visible_surface,
        weights.guessed_hidden_occupied * float(hidden_occupied_scale),
    )
    guessed_pixel_loss, guessed_group_losses = _weighted_per_sample_group_mean(
        guessed_map,
        (
            valid_guessed_free,
            valid_visible_surface,
            valid_hidden_guessed_occupied,
        ),
        guessed_group_weights,
    )
    (
        guessed_free_pixel_loss,
        visible_surface_occupied_pixel_loss,
        hidden_guessed_occupied_pixel_loss,
    ) = guessed_group_losses
    if probability_model == "evidential":
        wrong_evidence, wrong_evidence_groups = _weighted_per_sample_group_mean(
            _wrong_evidence_map(guessed, masks.occupied),
            (
                valid_guessed_free,
                valid_visible_surface,
                valid_hidden_guessed_occupied,
            ),
            guessed_group_weights,
        )
    else:
        wrong_evidence = guessed_pixel_loss * 0.0
        wrong_evidence_groups = (wrong_evidence, wrong_evidence, wrong_evidence)
    guessed_loss = weights.guessed_pixel * guessed_pixel_loss + (
        weights.wrong_evidence_kl
        * float(wrong_evidence_scale)
        * wrong_evidence
    )

    observed_gate_bce, observed_guessed_bce, observed_free_bce = (
        _class_balanced_binary_bce(
            prediction["observed_gate_logit"].float(),
            valid_observed_free,
            valid_guessed,
        )
    )
    routing_loss = weights.observed_gate_pixel * observed_gate_bce

    # Surface emphasis belongs exclusively to the Guessed Expert.  The Gate
    # still learns that these cells are not observed-free through its original
    # binary target, but this added surface term has no path into Gate logits.
    guessed_surface_map = -torch.log(
        prediction["guessed"]["occupancy_probability"].float().clamp_min(1e-6)
    )
    guessed_surface_loss = _per_sample_masked_mean(
        guessed_surface_map,
        valid_visible_surface,
    )
    guessed_surface_objective = weights.guessed_surface * guessed_surface_loss

    support_logit = prediction["fov_support_logit"].float()
    support_truth = masks.valid.to(support_logit.dtype)
    support_bce_map = F.binary_cross_entropy_with_logits(
        support_logit,
        support_truth,
        reduction="none",
    )
    inside_bce = _per_sample_masked_mean(
        support_bce_map,
        masks.valid & gt_valid,
    )
    outside_bce = _per_sample_masked_mean(
        support_bce_map,
        ~masks.valid & gt_valid,
    )
    support_bce = 0.5 * (inside_bce + outside_bce)
    support_probability = prediction["fov_support_probability"].float()
    support_loss_domain = gt_valid.to(support_probability.dtype)
    support_probability_for_loss = support_probability * support_loss_domain
    support_truth_for_loss = support_truth * support_loss_domain
    support_dice = 1.0 - (
        2.0
        * (support_probability_for_loss * support_truth_for_loss).sum()
        + 1.0
    ) / (
        support_probability_for_loss.sum()
        + support_truth_for_loss.sum()
        + 1.0
    )
    support_loss = weights.support_bce * support_bce + (
        weights.support_dice * support_dice
    )

    total = routing_loss + guessed_loss + guessed_surface_objective + support_loss
    return {
        "loss": total,
        "routing_objective": routing_loss,
        "guessed_objective": guessed_loss,
        "guessed_surface_objective": guessed_surface_objective,
        "support_objective": support_loss,
        "observed_gate_pixel_bce": observed_gate_bce,
        "observed_gate_guessed_bce": observed_guessed_bce,
        "observed_gate_free_bce": observed_free_bce,
        "guessed_pixel_loss": guessed_pixel_loss,
        "guessed_free_pixel_loss": guessed_free_pixel_loss,
        "visible_surface_occupied_pixel_loss": visible_surface_occupied_pixel_loss,
        "hidden_guessed_occupied_pixel_loss": hidden_guessed_occupied_pixel_loss,
        "guessed_surface_loss": guessed_surface_loss,
        "wrong_evidence_kl": wrong_evidence,
        "wrong_evidence_guessed_free": wrong_evidence_groups[0],
        "wrong_evidence_visible_surface": wrong_evidence_groups[1],
        "wrong_evidence_hidden_occupied": wrong_evidence_groups[2],
        "wrong_evidence_scale": torch.tensor(
            float(wrong_evidence_scale), device=complete_target.device
        ),
        "hidden_occupied_scale": torch.tensor(
            float(hidden_occupied_scale), device=complete_target.device
        ),
        "support_bce_loss": support_bce,
        "support_dice_loss": support_dice,
        "observed_free_fraction": masks.observed_free.float().mean(),
        "visible_surface_fraction": masks.visible_surface.float().mean(),
        "guessed_fraction": masks.guessed.float().mean(),
        "gt_valid_fraction": gt_valid.float().mean(),
        "gt_void_fraction": (~gt_valid).float().mean(),
        "semantic_gt_valid_fraction": semantic_gt_valid.float().mean(),
        "gt_void_inside_support_fraction": (
            masks.valid & ~gt_valid
        ).float().mean(),
        "hidden_guessed_fraction": masks.hidden_guessed.float().mean(),
        "guessed_free_fraction": masks.guessed_free.float().mean(),
        "guessed_occupied_fraction": masks.guessed_occupied.float().mean(),
        "hidden_guessed_occupied_fraction": (
            masks.hidden_guessed_occupied.float().mean()
        ),
        "support_fraction": masks.valid.float().mean(),
    }
