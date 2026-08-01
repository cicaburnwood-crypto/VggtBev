from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from vggt_bev_method1.config import LabelValues
from vggt_bev_method1.data.p2b_targets import (
    PackedRayBank,
    build_packed_ray_bank,
    p2b_region_masks,
)
from vggt_bev_method1.models.p2b_probability import ProbabilityModel


@dataclass(frozen=True)
class P2BLossWeights:
    ray_sequence: float = 1.0
    first_hit: float = 1.0
    first_hit_distance: float = 0.25
    surface_continuity: float = 0.05
    guessed_pixel: float = 1.0
    wrong_evidence_kl: float = 0.01
    gate_bce: float = 1.0
    gate_monotonic: float = 0.10
    support_bce: float = 0.5
    support_dice: float = 0.5
    fusion_gate: float = 0.0


def wrong_evidence_kl_weight(
    step: int,
    total_steps: int,
    *,
    maximum: float = 0.01,
    zero_fraction: float = 0.10,
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


def _per_sample_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    cell_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    flat_values = values.flatten(1)
    flat_mask = mask.flatten(1).to(values.dtype)
    if cell_weights is not None:
        if cell_weights.shape != values.shape:
            raise ValueError("cell weights must align with values")
        flat_mask = flat_mask * cell_weights.flatten(1).to(values.dtype)
    denominator = flat_mask.sum(dim=1)
    available = denominator > 0
    per_sample = (flat_values * flat_mask).sum(dim=1) / denominator.clamp_min(1.0)
    if bool(available.any()):
        return per_sample[available].mean()
    return values.sum() * 0.0


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


def _gather_ray(values: torch.Tensor, bank: PackedRayBank) -> torch.Tensor:
    return values.flatten()[bank.indices]


def _one_sample_ray_losses(
    probability: torch.Tensor,
    occupied_loss: torch.Tensor,
    free_loss: torch.Tensor,
    observed_free: torch.Tensor,
    observed_surface: torch.Tensor,
    support: torch.Tensor,
    bank: PackedRayBank,
) -> dict[str, torch.Tensor]:
    zero = probability.sum() * 0.0
    if bank.indices.shape[0] == 0:
        return {
            "ray_sequence": zero,
            "first_hit": zero,
            "first_hit_distance": zero,
            "surface_continuity": zero,
            "ray_count": torch.zeros((), device=probability.device),
            "hit_ray_count": torch.zeros((), device=probability.device),
        }

    ray_probability = _gather_ray(probability, bank)
    ray_occupied_loss = _gather_ray(occupied_loss, bank)
    ray_free_loss = _gather_ray(free_loss, bank)
    ray_observed_free = _gather_ray(observed_free, bank) & bank.valid
    ray_surface = _gather_ray(observed_surface, bank) & bank.valid
    ray_observed = ray_observed_free | ray_surface
    positions = torch.arange(
        bank.indices.shape[1], device=probability.device
    ).unsqueeze(0)
    surface_position = torch.where(
        ray_surface,
        positions,
        torch.full_like(positions, bank.indices.shape[1]),
    ).min(dim=1).values
    has_hit = surface_position < bank.indices.shape[1]
    last_observed = torch.where(
        ray_observed,
        positions,
        torch.full_like(positions, -1),
    ).max(dim=1).values
    active_ray = last_observed >= 0
    supervised_length = torch.where(has_hit, surface_position, last_observed) + 1
    prefix = bank.valid & (positions < supervised_length[:, None])

    free_prefix = prefix & ray_observed_free & (
        ~has_hit[:, None] | (positions < surface_position[:, None])
    )
    free_term = (
        ray_free_loss[free_prefix].mean() if bool(free_prefix.any()) else zero
    )
    hit_term = (
        ray_occupied_loss[
            torch.arange(bank.indices.shape[0], device=probability.device)[has_hit],
            surface_position[has_hit],
        ].mean()
        if bool(has_hit.any())
        else zero
    )
    sequence = free_term + hit_term

    clipped_probability = ray_probability.clamp(1e-6, 1.0 - 1e-6)
    log_survival_before = torch.cumsum(
        torch.log1p(-clipped_probability) * prefix,
        dim=1,
    ) - torch.log1p(-clipped_probability) * prefix
    hit_probability = torch.exp(log_survival_before) * clipped_probability * prefix
    no_hit_probability = torch.exp(
        (torch.log1p(-clipped_probability) * prefix).sum(dim=1)
    )
    target_hit_probability = hit_probability[
        torch.arange(bank.indices.shape[0], device=probability.device)[has_hit],
        surface_position[has_hit],
    ]
    first_hit_positive = (
        -torch.log(target_hit_probability.clamp_min(1e-8)).mean()
        if bool(has_hit.any())
        else zero
    )
    no_hit_rays = active_ray & ~has_hit
    first_hit_negative = (
        -torch.log(no_hit_probability[no_hit_rays].clamp_min(1e-8)).mean()
        if bool(no_hit_rays.any())
        else zero
    )
    if bool(has_hit.any()) and bool(no_hit_rays.any()):
        first_hit = 0.5 * (first_hit_positive + first_hit_negative)
    else:
        first_hit = first_hit_positive + first_hit_negative

    distances = positions.to(probability.dtype)
    no_hit_index = supervised_length.to(probability.dtype)
    expected_distance = (hit_probability * distances).sum(dim=1) + (
        no_hit_probability * no_hit_index
    )
    target_distance = torch.where(
        has_hit,
        surface_position,
        supervised_length,
    ).to(probability.dtype)
    normalization = supervised_length.to(probability.dtype).clamp_min(1.0)
    distance_loss = (
        F.smooth_l1_loss(
            expected_distance[active_ray] / normalization[active_ray],
            target_distance[active_ray] / normalization[active_ray],
        )
        if bool(active_ray.any())
        else zero
    )

    adjacent = has_hit[:-1] & has_hit[1:]
    gt_delta = surface_position[1:] - surface_position[:-1]
    adjacent = adjacent & (gt_delta.abs() <= 2)
    continuity = (
        F.smooth_l1_loss(
            expected_distance[1:][adjacent] - expected_distance[:-1][adjacent],
            gt_delta[adjacent].to(expected_distance.dtype),
        )
        if bool(adjacent.any())
        else zero
    )
    return {
        "ray_sequence": sequence,
        "first_hit": first_hit,
        "first_hit_distance": distance_loss,
        "surface_continuity": continuity,
        "ray_count": active_ray.sum().to(probability.dtype),
        "hit_ray_count": has_hit.sum().to(probability.dtype),
    }


def _batch_ray_losses(
    expert: dict[str, torch.Tensor],
    occupied_loss: torch.Tensor,
    free_loss: torch.Tensor,
    observed_free: torch.Tensor,
    observed_surface: torch.Tensor,
    support: torch.Tensor,
    banks: list[PackedRayBank],
) -> dict[str, torch.Tensor]:
    values = [
        _one_sample_ray_losses(
            expert["occupancy_probability"][index].float(),
            occupied_loss[index],
            free_loss[index],
            observed_free[index],
            observed_surface[index],
            support[index],
            banks[index],
        )
        for index in range(support.shape[0])
    ]
    return {
        key: torch.stack([sample[key] for sample in values]).mean()
        for key in values[0]
    }


def _gate_monotonic_loss(
    gate_probability: torch.Tensor,
    support: torch.Tensor,
    banks: list[PackedRayBank],
) -> torch.Tensor:
    losses = []
    for index in range(support.shape[0]):
        bank = banks[index]
        if bank.indices.shape[0] == 0:
            continue
        values = _gather_ray(gate_probability[index], bank)
        pairs = bank.valid[:, :-1] & bank.valid[:, 1:]
        if bool(pairs.any()):
            losses.append(F.relu(values[:, 1:] - values[:, :-1])[pairs].mean())
    if losses:
        return torch.stack(losses).mean()
    return gate_probability.sum() * 0.0


def p2b_bev_loss(
    prediction: dict,
    complete_target: torch.Tensor,
    visible_target: torch.Tensor,
    support_target: torch.Tensor,
    *,
    probability_model: ProbabilityModel,
    weights: P2BLossWeights = P2BLossWeights(),
    wrong_evidence_scale: float = 1.0,
    labels: LabelValues = LabelValues(),
) -> dict[str, torch.Tensor]:
    masks = p2b_region_masks(
        complete_target,
        visible_target,
        support_target,
        labels=labels,
    )
    observed = prediction["observed"]
    guessed = prediction["guessed"]
    observed_occupied_loss, observed_free_loss = _expert_cell_losses(
        observed, probability_model
    )
    guessed_occupied_loss, guessed_free_loss = _expert_cell_losses(
        guessed, probability_model
    )
    banks = [
        build_packed_ray_bank(masks.valid[index], device=complete_target.device)
        for index in range(complete_target.shape[0])
    ]
    observed_ray = _batch_ray_losses(
        observed,
        observed_occupied_loss,
        observed_free_loss,
        masks.observed_free,
        masks.observed_surface,
        masks.valid,
        banks,
    )
    observed_loss = (
        weights.ray_sequence * observed_ray["ray_sequence"]
        + weights.first_hit * observed_ray["first_hit"]
        + weights.first_hit_distance * observed_ray["first_hit_distance"]
        + weights.surface_continuity * observed_ray["surface_continuity"]
    )

    guessed_map = torch.where(
        masks.occupied,
        guessed_occupied_loss,
        guessed_free_loss,
    )
    guessed_pixel_loss = _per_sample_masked_mean(guessed_map, masks.guessed)
    if probability_model == "evidential":
        wrong_evidence = _per_sample_masked_mean(
            _wrong_evidence_map(guessed, masks.occupied),
            masks.guessed,
        )
    else:
        wrong_evidence = guessed_pixel_loss * 0.0
    guessed_loss = weights.guessed_pixel * guessed_pixel_loss + (
        weights.wrong_evidence_kl
        * float(wrong_evidence_scale)
        * wrong_evidence
    )

    gate_logit = prediction["gate_logit"].float()
    gate_bce_map = F.binary_cross_entropy_with_logits(
        gate_logit,
        masks.gate_target.to(gate_logit.dtype),
        reduction="none",
    )
    gate_bce = _per_sample_masked_mean(gate_bce_map, masks.valid)
    gate_monotonic = _gate_monotonic_loss(
        prediction["gate_probability"].float(), masks.valid, banks
    )
    gate_loss = weights.gate_bce * gate_bce + (
        weights.gate_monotonic * gate_monotonic
    )

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

    fused_probability = prediction["fused"]["occupancy_probability"]
    fused_target = masks.occupied.to(fused_probability.dtype)
    fusion_gate_loss = _per_sample_masked_mean(
        F.binary_cross_entropy(
            (
                prediction["gate_probability"]
                * observed["occupancy_probability"].detach()
                + (1.0 - prediction["gate_probability"])
                * guessed["occupancy_probability"].detach()
            ).clamp(1e-6, 1.0 - 1e-6),
            fused_target,
            reduction="none",
        ),
        masks.valid,
    )
    routing_loss = gate_loss + support_loss + (
        weights.fusion_gate * fusion_gate_loss
    )
    total = observed_loss + guessed_loss + routing_loss
    return {
        "loss": total,
        "observed_objective": observed_loss,
        "guessed_objective": guessed_loss,
        "routing_objective": routing_loss,
        "observed_ray_sequence_loss": observed_ray["ray_sequence"],
        "observed_first_hit_loss": observed_ray["first_hit"],
        "observed_first_hit_distance_loss": observed_ray["first_hit_distance"],
        "observed_surface_continuity_loss": observed_ray["surface_continuity"],
        "observed_ray_count": observed_ray["ray_count"],
        "observed_hit_ray_count": observed_ray["hit_ray_count"],
        "guessed_pixel_loss": guessed_pixel_loss,
        "wrong_evidence_kl": wrong_evidence,
        "wrong_evidence_scale": torch.tensor(
            float(wrong_evidence_scale), device=complete_target.device
        ),
        "gate_bce_loss": gate_bce,
        "gate_monotonic_loss": gate_monotonic,
        "support_bce_loss": support_bce,
        "support_dice_loss": support_dice,
        "fusion_gate_loss": fusion_gate_loss,
        "observed_fraction": masks.observed.float().mean(),
        "guessed_fraction": masks.guessed.float().mean(),
        "support_fraction": masks.valid.float().mean(),
    }
