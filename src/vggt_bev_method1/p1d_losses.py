from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from vggt_bev_method1.data.p1b_targets import p1b_region_masks
from vggt_bev_method1.models.p1b_probability import ProbabilityModel
from vggt_bev_method1.p1b_losses import (
    P1BLossWeights,
    _class_balanced_binary_bce,
    _expert_cell_losses,
    _weighted_per_sample_group_mean,
    p1b_bev_loss,
)


@dataclass(frozen=True)
class P1DAdditionalLossWeights:
    history_observed_gate: float = 0.25
    history_support: float = 0.20
    history_guessed: float = 0.30
    guessed_hard_pixel: float = 0.25
    guessed_hard_fraction: float = 0.10
    guessed_hard_minimum: int = 128
    guessed_hard_maximum_per_group: int = 8192

    def __post_init__(self) -> None:
        scalar_weights = (
            self.history_observed_gate,
            self.history_support,
            self.history_guessed,
            self.guessed_hard_pixel,
        )
        if any(value < 0.0 for value in scalar_weights):
            raise ValueError("P1D additional loss weights cannot be negative")
        if not 0.0 < self.guessed_hard_fraction <= 1.0:
            raise ValueError("hard-pixel fraction must be inside (0,1]")
        if self.guessed_hard_minimum <= 0:
            raise ValueError("hard-pixel minimum must be positive")
        if self.guessed_hard_maximum_per_group < self.guessed_hard_minimum:
            raise ValueError("hard-pixel maximum must not be below its minimum")


def _fixed_budget_topk_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    fraction: float,
    minimum: int,
    maximum: int,
) -> torch.Tensor:
    """Mean the hardest valid pixels with a bounded, GPU-stable budget."""

    if values.shape != mask.shape:
        raise ValueError("hard-pixel values and mask must align")
    pixels = values.shape[-2] * values.shape[-1]
    budget = min(maximum, max(minimum, math.ceil(pixels * fraction)))
    budget = min(budget, pixels)
    candidates = torch.where(
        mask.flatten(1),
        values.flatten(1),
        torch.full_like(values.flatten(1), float("-inf")),
    )
    selected = candidates.topk(budget, dim=1, sorted=False).values
    valid = torch.isfinite(selected)
    counts = valid.sum(dim=1)
    per_sample = torch.where(valid, selected, 0.0).sum(dim=1) / counts.clamp_min(1)
    available = counts > 0
    return (
        per_sample * available.to(per_sample.dtype)
    ).sum() / available.sum().clamp_min(1).to(per_sample.dtype)


def _weighted_hard_group_loss(
    values: torch.Tensor,
    masks: tuple[torch.Tensor, ...],
    group_weights: tuple[float, ...],
    settings: P1DAdditionalLossWeights,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    losses = tuple(
        _fixed_budget_topk_mean(
            values,
            mask,
            fraction=settings.guessed_hard_fraction,
            minimum=settings.guessed_hard_minimum,
            maximum=settings.guessed_hard_maximum_per_group,
        )
        for mask in masks
    )
    present = torch.stack([mask.any() for mask in masks])
    configured = values.new_tensor(group_weights)
    effective = configured * present.to(configured.dtype)
    combined = sum(loss * weight for loss, weight in zip(losses, effective))
    return combined / effective.sum().clamp_min(1e-6), losses


def _binary_metrics(
    probability: torch.Tensor,
    truth: torch.Tensor,
    domain: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = probability >= 0.5
    truth = truth.bool()
    domain = domain.bool()
    true_positive = (prediction & truth & domain).sum().to(torch.float32)
    false_positive = (prediction & ~truth & domain).sum().to(torch.float32)
    false_negative = (~prediction & truth & domain).sum().to(torch.float32)
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    iou = true_positive / (
        true_positive + false_positive + false_negative
    ).clamp_min(1.0)
    return precision, recall, iou


def p1d_bev_loss(
    prediction: dict,
    complete_target: torch.Tensor,
    visible_target: torch.Tensor,
    support_target: torch.Tensor,
    *,
    latest_observed_free_target: torch.Tensor,
    latest_support_target: torch.Tensor,
    gt_valid_mask: torch.Tensor,
    probability_model: ProbabilityModel,
    base_weights: P1BLossWeights,
    additional_weights: P1DAdditionalLossWeights,
    wrong_evidence_scale: float,
    hidden_occupied_scale: float,
) -> dict[str, torch.Tensor]:
    """P1C/WTBD loss plus temporal-region and same-NLL hard supervision.

    Every added objective is training-only and uses an existing output.  No
    surface channel, post-processing target, ray loss, or runtime branch is
    introduced.
    """

    base = p1b_bev_loss(
        prediction,
        complete_target,
        visible_target,
        support_target,
        gt_valid_mask=gt_valid_mask,
        probability_model=probability_model,
        weights=base_weights,
        wrong_evidence_scale=wrong_evidence_scale,
        hidden_occupied_scale=hidden_occupied_scale,
    )
    masks = p1b_region_masks(complete_target, visible_target, support_target)
    gt_valid = gt_valid_mask.bool()
    semantic_valid = masks.valid & gt_valid
    if not (
        latest_observed_free_target.shape
        == latest_support_target.shape
        == complete_target.shape
    ):
        raise ValueError("latest temporal targets must align with Merged GT")
    latest_support = latest_support_target.bool() & semantic_valid
    latest_observed = latest_observed_free_target.bool() & latest_support
    history_support = masks.valid & ~latest_support & gt_valid
    history_observed = masks.observed_free & ~latest_observed & gt_valid

    occupied_nll, free_nll = _expert_cell_losses(
        prediction["guessed"], probability_model
    )
    semantic_nll = torch.where(masks.occupied, occupied_nll, free_nll)
    guessed_groups = (
        masks.guessed_free & semantic_valid,
        masks.visible_surface & semantic_valid,
        masks.hidden_guessed_occupied & semantic_valid,
    )
    group_weights = (
        base_weights.guessed_free,
        base_weights.guessed_visible_surface,
        base_weights.guessed_hidden_occupied * float(hidden_occupied_scale),
    )
    hard_loss, hard_groups = _weighted_hard_group_loss(
        semantic_nll,
        guessed_groups,
        group_weights,
        additional_weights,
    )

    history_guessed_groups = tuple(
        group & history_support for group in guessed_groups
    )
    history_guessed_loss, history_guessed_groups_loss = (
        _weighted_per_sample_group_mean(
            semantic_nll,
            history_guessed_groups,
            group_weights,
        )
    )
    history_gate_domain = history_support
    history_gate_loss, _, _ = _class_balanced_binary_bce(
        prediction["observed_gate_logit"].float(),
        history_observed,
        history_gate_domain & ~masks.observed_free,
    )
    history_support_loss = F.softplus(
        -prediction["fov_support_logit"].float()
    )
    history_support_loss = (
        history_support_loss * history_support.to(history_support_loss.dtype)
    ).flatten(1).sum(dim=1) / history_support.flatten(1).sum(dim=1).clamp_min(1)
    history_available = history_support.flatten(1).any(dim=1)
    history_support_loss = (
        history_support_loss * history_available.to(history_support_loss.dtype)
    ).sum() / history_available.sum().clamp_min(1).to(history_support_loss.dtype)

    hard_objective = additional_weights.guessed_hard_pixel * hard_loss
    history_objective = (
        additional_weights.history_observed_gate * history_gate_loss
        + additional_weights.history_support * history_support_loss
        + additional_weights.history_guessed * history_guessed_loss
    )
    total = base["loss"] + hard_objective + history_objective

    gate_probability = torch.sigmoid(prediction["observed_gate_logit"].float())
    support_probability = torch.sigmoid(prediction["fov_support_logit"].float())
    occupancy_probability = prediction["guessed"][
        "occupancy_probability"
    ].float()
    gate_precision, gate_recall, gate_iou = _binary_metrics(
        gate_probability,
        masks.observed_free,
        semantic_valid,
    )
    support_precision, support_recall, support_iou = _binary_metrics(
        support_probability,
        masks.valid,
        gt_valid,
    )
    guessed_precision, guessed_recall, guessed_iou = _binary_metrics(
        occupancy_probability,
        masks.occupied,
        masks.guessed & semantic_valid,
    )
    history_gate_precision, history_gate_recall, history_gate_iou = (
        _binary_metrics(
            gate_probability,
            masks.observed_free,
            history_gate_domain,
        )
    )
    history_guess_precision, history_guess_recall, history_guess_iou = (
        _binary_metrics(
            occupancy_probability,
            masks.occupied,
            masks.guessed & history_support,
        )
    )
    return {
        **base,
        "loss": total,
        "p1c_base_bev_objective": base["loss"],
        "guessed_hard_pixel_objective": hard_objective,
        "guessed_hard_pixel_nll": hard_loss,
        "guessed_hard_free_nll": hard_groups[0],
        "guessed_hard_visible_surface_nll": hard_groups[1],
        "guessed_hard_hidden_occupied_nll": hard_groups[2],
        "history_objective": history_objective,
        "history_observed_gate_loss": history_gate_loss,
        "history_support_loss": history_support_loss,
        "history_guessed_nll": history_guessed_loss,
        "history_guessed_free_nll": history_guessed_groups_loss[0],
        "history_visible_surface_nll": history_guessed_groups_loss[1],
        "history_hidden_occupied_nll": history_guessed_groups_loss[2],
        "history_support_fraction": history_support.float().mean(),
        "history_observed_free_fraction": history_observed.float().mean(),
        "observed_gate_precision": gate_precision,
        "observed_gate_recall": gate_recall,
        "observed_gate_iou": gate_iou,
        "support_precision": support_precision,
        "support_recall": support_recall,
        "support_iou": support_iou,
        "guessed_pixel_precision": guessed_precision,
        "guessed_pixel_recall": guessed_recall,
        "guessed_pixel_iou": guessed_iou,
        "history_gate_precision": history_gate_precision,
        "history_gate_recall": history_gate_recall,
        "history_gate_iou": history_gate_iou,
        "history_guessed_precision": history_guess_precision,
        "history_guessed_recall": history_guess_recall,
        "history_guessed_iou": history_guess_iou,
    }
