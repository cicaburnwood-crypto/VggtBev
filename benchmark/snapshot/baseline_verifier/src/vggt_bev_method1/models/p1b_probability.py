from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F

ProbabilityModel = Literal["evidential", "bce"]


def decode_binary_prediction(
    raw: torch.Tensor,
    probability_model: ProbabilityModel,
    *,
    include_diagnostics: bool = True,
) -> dict[str, torch.Tensor]:
    """Decode one expert without pretending BCE logits are evidence."""

    if probability_model == "evidential":
        if raw.shape[1] != 2:
            raise ValueError("evidential experts require two output channels")
        evidence = F.softplus(raw.float())
        alpha = evidence[:, 0] + 1.0
        beta = evidence[:, 1] + 1.0
        strength = alpha + beta
        probability = alpha / strength
        variance = alpha * beta / (strength.square() * (strength + 1.0))
        output = {
            "raw": raw,
            "alpha_occupied": alpha,
            "beta_free": beta,
            "occupancy_probability": probability,
        }
        if not include_diagnostics:
            return output
        output.update(
            {
            "evidence_strength": strength,
            "occupancy_distribution_variance": variance,
            "epistemic_uncertainty": 2.0 / strength,
            "evidence_confidence": (1.0 - 2.0 / strength).clamp(0.0, 1.0),
            "classification_confidence": (2.0 * probability - 1.0).abs(),
            }
        )
        return output
    if probability_model == "bce":
        if raw.shape[1] != 1:
            raise ValueError("BCE experts require one output channel")
        logit = raw[:, 0].float()
        probability = torch.sigmoid(logit)
        output = {
            "raw": raw,
            "occupancy_logit": logit,
            "occupancy_probability": probability,
        }
        if include_diagnostics:
            output["classification_confidence"] = (
                2.0 * probability - 1.0
            ).abs()
        return output
    raise ValueError(f"unsupported probability model: {probability_model}")


def fuse_pixel_routing(
    routing_probability: torch.Tensor,
    guessed: dict[str, torch.Tensor],
    support_probability: torch.Tensor,
    probability_model: ProbabilityModel,
) -> dict[str, torch.Tensor]:
    """Fuse observed-free, guessed-free and guessed-occupied probabilities."""

    if routing_probability.ndim != 4 or routing_probability.shape[1] != 3:
        raise ValueError("routing probability must be Bx3xHxW")
    observed_free = routing_probability[:, 0].float()
    guessed_free = routing_probability[:, 1].float()
    guessed_occupied = routing_probability[:, 2].float()
    guessed_region = guessed_free + guessed_occupied
    guessed_mean = guessed["occupancy_probability"].float()
    support = support_probability.float().clamp(0.0, 1.0)
    if not (
        observed_free.shape
        == guessed_free.shape
        == guessed_occupied.shape
        == guessed_region.shape
        == guessed_mean.shape
        == support.shape
    ):
        raise ValueError("routing, completion and support raster shapes must match")

    probability = guessed_occupied
    classification_confidence = (2.0 * probability - 1.0).abs()
    routing_entropy = -(
        routing_probability.float().clamp_min(1e-8).log()
        * routing_probability.float()
    ).sum(dim=1) / torch.log(
        torch.tensor(3.0, device=probability.device, dtype=probability.dtype)
    )
    routing_confidence = (1.0 - routing_entropy).clamp(0.0, 1.0)
    output = {
        "occupancy_probability": probability,
        "free_probability": 1.0 - probability,
        "unknown_probability": 1.0 - support,
        "classification_confidence": classification_confidence,
        "routing_entropy": routing_entropy,
        "routing_confidence": routing_confidence,
        "observed_free_probability": observed_free,
        "guessed_free_probability": guessed_free,
        "guessed_occupied_probability": guessed_occupied,
        "guessed_region_probability": guessed_region,
    }

    if probability_model == "evidential":
        guessed_variance = guessed["occupancy_distribution_variance"].float()
        mixture_variance = observed_free * probability.square() + guessed_region * (
            guessed_variance + (guessed_mean - probability).square()
        )
        distribution_confidence = (1.0 - 4.0 * mixture_variance).clamp(0.0, 1.0)
        output.update(
            {
                "occupancy_distribution_variance": mixture_variance,
                "distribution_confidence": distribution_confidence,
                "mixture_uncertainty": 1.0 - distribution_confidence,
                "completion_epistemic_uncertainty": guessed_region
                * guessed["epistemic_uncertainty"].float(),
                "navigation_confidence": (
                    support * classification_confidence * distribution_confidence
                ),
            }
        )
    elif probability_model == "bce":
        # This is classification certainty, not epistemic confidence.
        output["navigation_confidence"] = (
            support * classification_confidence * routing_confidence
        )
    else:
        raise ValueError(f"unsupported probability model: {probability_model}")
    return output


def compose_semantic(
    fused: dict[str, torch.Tensor],
    support_probability: torch.Tensor,
    *,
    support_threshold: float = 0.5,
    occupancy_threshold: float = 0.5,
    occupied_value: int = 0,
    unknown_value: int = 112,
    free_value: int = 255,
) -> torch.Tensor:
    probability = fused["occupancy_probability"]
    output = torch.full_like(probability, unknown_value, dtype=torch.uint8)
    inside = support_probability >= support_threshold
    output[inside & (probability >= occupancy_threshold)] = occupied_value
    output[inside & (probability < occupancy_threshold)] = free_value
    return output
