from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F

ProbabilityModel = Literal["evidential", "bce"]


def decode_binary_prediction(
    raw: torch.Tensor,
    probability_model: ProbabilityModel,
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
        return {
            "raw": raw,
            "alpha_occupied": alpha,
            "beta_free": beta,
            "evidence_strength": strength,
            "occupancy_probability": probability,
            "occupancy_distribution_variance": variance,
            "epistemic_uncertainty": 2.0 / strength,
            "evidence_confidence": (1.0 - 2.0 / strength).clamp(0.0, 1.0),
            "classification_confidence": (2.0 * probability - 1.0).abs(),
        }
    if probability_model == "bce":
        if raw.shape[1] != 1:
            raise ValueError("BCE experts require one output channel")
        logit = raw[:, 0].float()
        probability = torch.sigmoid(logit)
        return {
            "raw": raw,
            "occupancy_logit": logit,
            "occupancy_probability": probability,
            "classification_confidence": (2.0 * probability - 1.0).abs(),
        }
    raise ValueError(f"unsupported probability model: {probability_model}")


def fuse_experts(
    observed: dict[str, torch.Tensor],
    guessed: dict[str, torch.Tensor],
    gate_probability: torch.Tensor,
    support_probability: torch.Tensor,
    probability_model: ProbabilityModel,
) -> dict[str, torch.Tensor]:
    """Fuse conditional experts and retain disagreement as uncertainty."""

    observed_mean = observed["occupancy_probability"].float()
    guessed_mean = guessed["occupancy_probability"].float()
    gate = gate_probability.float().clamp(0.0, 1.0)
    support = support_probability.float().clamp(0.0, 1.0)
    if not (
        observed_mean.shape
        == guessed_mean.shape
        == gate.shape
        == support.shape
    ):
        raise ValueError("expert, gate and support raster shapes must match")

    probability = gate * observed_mean + (1.0 - gate) * guessed_mean
    disagreement = gate * (1.0 - gate) * (observed_mean - guessed_mean).square()
    classification_confidence = (2.0 * probability - 1.0).abs()
    output = {
        "occupancy_probability": probability,
        "free_probability": 1.0 - probability,
        "unknown_probability": 1.0 - support,
        "classification_confidence": classification_confidence,
        "expert_disagreement": disagreement,
    }

    if probability_model == "evidential":
        observed_variance = observed["occupancy_distribution_variance"].float()
        guessed_variance = guessed["occupancy_distribution_variance"].float()
        mixture_variance = gate * (
            observed_variance + (observed_mean - probability).square()
        ) + (1.0 - gate) * (
            guessed_variance + (guessed_mean - probability).square()
        )
        distribution_confidence = (1.0 - 4.0 * mixture_variance).clamp(0.0, 1.0)
        output.update(
            {
                "occupancy_distribution_variance": mixture_variance,
                "distribution_confidence": distribution_confidence,
                "mixture_uncertainty": 1.0 - distribution_confidence,
                "navigation_confidence": (
                    support * classification_confidence * distribution_confidence
                ),
            }
        )
    elif probability_model == "bce":
        # This is classification certainty, not epistemic confidence. Keep the
        # disagreement channel explicit so downstream code cannot confuse them.
        output["navigation_confidence"] = support * classification_confidence
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
