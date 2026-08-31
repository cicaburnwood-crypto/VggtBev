from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from vggt_bev_method1.data.p1b_targets import p1b_region_masks


def _masked_per_sample_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.shape != mask.shape:
        raise ValueError("M04 loss value and mask shapes must match")
    count = mask.flatten(1).sum(dim=1)
    mean = (
        (value * mask.to(value.dtype)).flatten(1).sum(dim=1)
        / count.clamp_min(1).to(value.dtype)
    )
    return mean, count > 0


def balanced_binary_logit_nll(
    logit: torch.Tensor,
    truth: torch.Tensor,
    domain: torch.Tensor,
) -> torch.Tensor:
    """Parameter-free macro conditional Bernoulli negative log-likelihood."""

    if not (logit.shape == truth.shape == domain.shape):
        raise ValueError("M04 Bernoulli target tensors must align")
    truth = truth.bool()
    domain = domain.bool()
    positive, positive_present = _masked_per_sample_mean(
        F.softplus(-logit.float()), domain & truth
    )
    negative, negative_present = _masked_per_sample_mean(
        F.softplus(logit.float()), domain & ~truth
    )
    present = torch.stack((positive_present, negative_present), dim=1)
    values = torch.stack((positive, negative), dim=1)
    per_sample = (
        (values * present.to(values.dtype)).sum(dim=1)
        / present.sum(dim=1).clamp_min(1).to(values.dtype)
    )
    available = present.any(dim=1)
    return (
        per_sample * available.to(per_sample.dtype)
    ).sum() / available.sum().clamp_min(1).to(per_sample.dtype)


def _beta_expected_nll(
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    strength = alpha + beta
    return (
        torch.digamma(strength) - torch.digamma(alpha),
        torch.digamma(strength) - torch.digamma(beta),
    )


def _beta_uniform_kl(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    strength = alpha + beta
    return (
        torch.lgamma(strength)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + (alpha - 1.0) * torch.digamma(alpha)
        + (beta - 1.0) * torch.digamma(beta)
        + (2.0 - strength) * torch.digamma(strength)
    )


def _balanced_beta_nll(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    occupied: torch.Tensor,
    domain: torch.Tensor,
) -> torch.Tensor:
    occupied_nll, free_nll = _beta_expected_nll(alpha.float(), beta.float())
    positive, positive_present = _masked_per_sample_mean(
        occupied_nll, domain & occupied
    )
    negative, negative_present = _masked_per_sample_mean(
        free_nll, domain & ~occupied
    )
    present = torch.stack((positive_present, negative_present), dim=1)
    values = torch.stack((positive, negative), dim=1)
    per_sample = (
        (values * present.to(values.dtype)).sum(dim=1)
        / present.sum(dim=1).clamp_min(1).to(values.dtype)
    )
    available = present.any(dim=1)
    return (
        per_sample * available.to(per_sample.dtype)
    ).sum() / available.sum().clamp_min(1).to(per_sample.dtype)


def m04_bev_loss(
    prediction: dict,
    complete_target: torch.Tensor,
    visible_target: torch.Tensor,
    support_target: torch.Tensor,
    *,
    gt_valid_mask: torch.Tensor,
    evidence_kl_weight: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """One hierarchical map likelihood over the existing M04 outputs.

    Void and cells outside the 10 m metric source are hard ignored.  The loss
    has no boundary, Dice, hard-pixel, history-region, planner, or curriculum
    term.
    """

    if evidence_kl_weight < 0.0:
        raise ValueError("M04 evidence KL weight cannot be negative")
    masks = p1b_region_masks(
        complete_target, visible_target, support_target
    )
    gt_valid = gt_valid_mask.bool()
    support_domain = gt_valid
    semantic_domain = masks.valid & gt_valid
    guessed_domain = masks.guessed & semantic_domain
    support_nll = balanced_binary_logit_nll(
        prediction["fov_support_logit"],
        masks.valid,
        support_domain,
    )
    gate_nll = balanced_binary_logit_nll(
        prediction["observed_gate_logit"],
        masks.observed_free,
        semantic_domain,
    )
    alpha = prediction["guessed"]["alpha_occupied"]
    beta = prediction["guessed"]["beta_free"]
    occupancy_nll = _balanced_beta_nll(
        alpha,
        beta,
        masks.occupied,
        guessed_domain,
    )
    evidence_kl_map = _beta_uniform_kl(alpha.float(), beta.float())
    evidence_kl_per_sample, evidence_present = _masked_per_sample_mean(
        evidence_kl_map, guessed_domain
    )
    evidence_kl = (
        evidence_kl_per_sample
        * evidence_present.to(evidence_kl_per_sample.dtype)
    ).sum() / evidence_present.sum().clamp_min(1).to(
        evidence_kl_per_sample.dtype
    )
    map_nll = support_nll + gate_nll + occupancy_nll
    loss = map_nll + float(evidence_kl_weight) * evidence_kl
    occupancy_probability = alpha.float() / (alpha.float() + beta.float())
    return {
        "loss": loss,
        "map_nll": map_nll,
        "support_nll": support_nll,
        "observed_gate_nll": gate_nll,
        "occupancy_nll": occupancy_nll,
        "evidence_kl": evidence_kl,
        "evidence_kl_weight": loss.new_tensor(float(evidence_kl_weight)),
        "gt_valid_fraction": gt_valid.float().mean(),
        "support_fraction": masks.valid.float().mean(),
        "semantic_valid_fraction": semantic_domain.float().mean(),
        "guessed_fraction": guessed_domain.float().mean(),
        "occupied_fraction": (masks.occupied & semantic_domain).float().mean(),
        "mean_occupancy_probability": (
            occupancy_probability * guessed_domain.to(occupancy_probability.dtype)
        ).sum()
        / guessed_domain.sum().clamp_min(1).to(occupancy_probability.dtype),
    }


def _student_t_nll(
    residual: torch.Tensor,
    sigma: torch.Tensor,
    *,
    degrees_of_freedom: float,
) -> torch.Tensor:
    if degrees_of_freedom <= 0.0:
        raise ValueError("Student-t degrees of freedom must be positive")
    nu = float(degrees_of_freedom)
    constant = (
        math.lgamma(nu / 2.0)
        - math.lgamma((nu + 1.0) / 2.0)
        + 0.5 * math.log(nu * math.pi)
    )
    return (
        sigma.log()
        + 0.5 * (nu + 1.0) * torch.log1p(residual.square() / (nu * sigma.square()))
        + constant
    )


def m04_scale_loss(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    degrees_of_freedom: float = 3.0,
    minimum_sigma_log: float = 0.02,
    maximum_sigma_log: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Robust dense log-depth-ratio likelihood for one Scale Token."""

    if not 0.0 < minimum_sigma_log < maximum_sigma_log:
        raise ValueError("M04 scale sigma bounds are invalid")
    predicted_log = prediction["log_lambda_m_per_vggt"].float()
    log_variance = prediction.get("log_variance")
    if log_variance is None:
        sigma = torch.ones_like(predicted_log)
    else:
        sigma = torch.exp(0.5 * log_variance.float()).clamp(
            minimum_sigma_log, maximum_sigma_log
        )
    dense_residual = (
        target["gt_depth_m"].float().clamp_min(1e-6).log()
        - target["vggt_depth"].float().clamp_min(1e-6).log()
        - predicted_log[:, None, None, None]
    )
    dense_nll = _student_t_nll(
        dense_residual,
        sigma[:, None, None, None],
        degrees_of_freedom=degrees_of_freedom,
    )
    dense_mask = target["dense_inlier_mask"].bool()
    dense_weight = torch.where(
        dense_mask,
        target["dense_weight"].to(dense_nll.dtype),
        torch.zeros_like(dense_nll),
    )
    weight_sum = dense_weight.flatten(1).sum(dim=1)
    per_sample_dense = (
        (dense_nll * dense_weight).flatten(1).sum(dim=1)
        / weight_sum.clamp_min(1e-6)
    )
    dense_available = weight_sum > 0

    # A robust scalar fallback keeps synthetic/unit-test targets and rare
    # teacher windows without dense inliers well-defined.
    scalar_residual = predicted_log - target["log_lambda_gt"].float()
    scalar_nll = _student_t_nll(
        scalar_residual,
        sigma,
        degrees_of_freedom=degrees_of_freedom,
    )
    per_sample = torch.where(dense_available, per_sample_dense, scalar_nll)
    valid = target["target_valid"].bool()
    quality = target["quality_weight"].to(per_sample.dtype) * valid.to(
        per_sample.dtype
    )
    loss = (quality * per_sample).sum() / valid.sum().clamp_min(1).to(
        per_sample.dtype
    )
    absolute_log_error = scalar_residual.abs()
    return {
        "loss": loss,
        "student_t_nll": loss,
        "valid_scale_fraction": valid.float().mean(),
        "dense_scale_fraction": (valid & dense_available).float().mean(),
        "mean_sigma_log": (
            sigma * valid.to(sigma.dtype)
        ).sum() / valid.sum().clamp_min(1).to(sigma.dtype),
        "mean_abs_log_error": (
            absolute_log_error * valid.to(absolute_log_error.dtype)
        ).sum()
        / valid.sum().clamp_min(1).to(absolute_log_error.dtype),
    }
