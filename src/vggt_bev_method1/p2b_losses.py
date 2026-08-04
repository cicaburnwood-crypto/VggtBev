from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from vggt_bev_method1.config import LabelValues
from vggt_bev_method1.data.p2b_targets import p2b_region_masks
from vggt_bev_method1.models.p2b_probability import ProbabilityModel


@dataclass(frozen=True)
class P2BLossWeights:
    observed_gate_pixel: float = 1.0
    guessed_pixel: float = 1.0
    guessed_surface: float = 0.5
    guessed_free: float = 0.35
    guessed_visible_surface: float = 0.40
    guessed_hidden_occupied: float = 0.25
    wrong_evidence_kl: float = 0.005
    support_bce: float = 0.5
    support_dice: float = 0.5


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
    if bool(available.any()):
        return per_sample[available].mean()
    return values.sum() * 0.0


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
    if bool((stacked.sum(dim=1) > 1).any()):
        raise ValueError("loss groups must be mutually exclusive")
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
    combined = (
        per_sample[available].mean() if bool(available.any()) else values.sum() * 0.0
    )
    diagnostics = []
    for group_index in range(len(masks)):
        group_available = present[:, group_index]
        diagnostics.append(
            group_means[group_available, group_index].mean()
            if bool(group_available.any())
            else values.sum() * 0.0
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
    if bool((positive & negative).any()):
        raise ValueError("binary Gate positive and negative masks overlap")
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
    balanced = (
        per_sample[available_samples].mean()
        if bool(available_samples.any())
        else pixel_bce.sum() * 0.0
    )
    negative_loss = _per_sample_masked_mean(pixel_bce, negative)
    positive_loss = _per_sample_masked_mean(pixel_bce, positive)
    return balanced, negative_loss, positive_loss


def p2b_bev_loss(
    prediction: dict,
    complete_target: torch.Tensor,
    visible_target: torch.Tensor,
    support_target: torch.Tensor,
    *,
    probability_model: ProbabilityModel,
    weights: P2BLossWeights = P2BLossWeights(),
    wrong_evidence_scale: float = 1.0,
    hidden_occupied_scale: float = 1.0,
    labels: LabelValues = LabelValues(),
) -> dict[str, torch.Tensor]:
    masks = p2b_region_masks(
        complete_target,
        visible_target,
        support_target,
        labels=labels,
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
            masks.guessed_free,
            masks.visible_surface,
            masks.hidden_guessed_occupied,
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
                masks.guessed_free,
                masks.visible_surface,
                masks.hidden_guessed_occupied,
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
            masks.observed_free,
            masks.guessed,
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
        masks.visible_surface,
    )
    guessed_surface_objective = weights.guessed_surface * guessed_surface_loss

    support_logit = prediction["fov_support_logit"].float()
    support_truth = masks.valid.to(support_logit.dtype)
    support_bce_map = F.binary_cross_entropy_with_logits(
        support_logit,
        support_truth,
        reduction="none",
    )
    inside_bce = _per_sample_masked_mean(support_bce_map, masks.valid)
    outside_bce = _per_sample_masked_mean(support_bce_map, ~masks.valid)
    support_bce = 0.5 * (inside_bce + outside_bce)
    support_probability = prediction["fov_support_probability"].float()
    support_dice = 1.0 - (
        2.0 * (support_probability * support_truth).sum() + 1.0
    ) / (support_probability.sum() + support_truth.sum() + 1.0)
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
        "hidden_guessed_fraction": masks.hidden_guessed.float().mean(),
        "guessed_free_fraction": masks.guessed_free.float().mean(),
        "guessed_occupied_fraction": masks.guessed_occupied.float().mean(),
        "hidden_guessed_occupied_fraction": (
            masks.hidden_guessed_occupied.float().mean()
        ),
        "support_fraction": masks.valid.float().mean(),
    }
