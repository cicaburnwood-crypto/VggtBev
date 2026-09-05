from __future__ import annotations

import torch

from vggt_bev_method1.p1b_losses import (
    P1BLossWeights,
    hidden_occupied_supervision_weight,
    p1b_bev_loss,
    wrong_evidence_kl_weight,
)


def m05_bev_loss(
    merged_prediction: dict,
    latest_prediction: dict,
    merged_target: dict[str, torch.Tensor],
    latest_target: dict[str, torch.Tensor],
    *,
    weights: P1BLossWeights,
    global_step: int,
    total_steps: int,
    wrong_evidence_zero_fraction: float = 0.20,
    wrong_evidence_ramp_fraction: float = 0.10,
    hidden_occupied_zero_fraction: float = 0.10,
    hidden_occupied_ramp_fraction: float = 0.15,
    latest_auxiliary_weight: float = 0.50,
    loss_combination: str = "convex",
) -> dict[str, torch.Tensor]:
    """Apply the proven Single objective to latest and Merged predictions.

    The historical M05 default remains a convex mix. M05+ selects
    ``merged_primary_additive`` so the Merged objective keeps unit gradient
    weight while latest-frame supervision acts only as a weak auxiliary.
    """

    if loss_combination not in {"convex", "merged_primary_additive"}:
        raise ValueError(f"unsupported M05 loss combination: {loss_combination}")
    if not 0.0 <= latest_auxiliary_weight <= 1.0:
        raise ValueError("M05 latest auxiliary weight must be in [0,1]")

    wrong_scale = wrong_evidence_kl_weight(
        global_step,
        total_steps,
        maximum=1.0,
        zero_fraction=wrong_evidence_zero_fraction,
        ramp_fraction=wrong_evidence_ramp_fraction,
    )
    hidden_scale = hidden_occupied_supervision_weight(
        global_step,
        total_steps,
        zero_fraction=hidden_occupied_zero_fraction,
        ramp_fraction=hidden_occupied_ramp_fraction,
    )

    def branch(prediction: dict, target: dict[str, torch.Tensor]) -> dict:
        return p1b_bev_loss(
            prediction,
            target["complete_target"],
            target["visible_target"],
            target["support_target"],
            gt_valid_mask=target["gt_valid_mask"],
            probability_model="evidential",
            weights=weights,
            wrong_evidence_scale=wrong_scale,
            hidden_occupied_scale=hidden_scale,
        )

    merged = branch(merged_prediction, merged_target)
    latest = branch(latest_prediction, latest_target)
    latest_weight = float(latest_auxiliary_weight)
    merged_weight = (
        1.0 if loss_combination == "merged_primary_additive" else 1.0 - latest_weight
    )
    loss = merged_weight * merged["loss"] + latest_weight * latest["loss"]
    return {
        "loss": loss,
        "merged_loss": merged["loss"],
        "latest_auxiliary_loss": latest["loss"],
        "merged_loss_weight": loss.new_tensor(merged_weight),
        "latest_auxiliary_loss_weight": loss.new_tensor(latest_weight),
        "loss_combination_is_merged_primary_additive": loss.new_tensor(
            float(loss_combination == "merged_primary_additive")
        ),
        "wrong_evidence_schedule_scale": loss.new_tensor(wrong_scale),
        "hidden_occupied_schedule_scale": loss.new_tensor(hidden_scale),
        **{f"merged_{key}": value for key, value in merged.items() if key != "loss"},
        **{
            f"latest_{key}": value
            for key, value in latest.items()
            if key != "loss"
        },
    }


def m05_loss_weights(training: dict) -> P1BLossWeights:
    """Construct the exact configurable loss family used by Single Baseline."""

    return P1BLossWeights(
        observed_gate_pixel=float(training.get("observed_gate_pixel_weight", 1.0)),
        observed_gate_region=float(training.get("observed_gate_region_weight", 0.0)),
        observed_gate_boundary_emphasis=float(
            training.get("observed_gate_boundary_emphasis", 0.0)
        ),
        guessed_pixel=float(training.get("guessed_pixel_weight", 1.0)),
        guessed_surface=float(training.get("guessed_surface_weight", 0.5)),
        guessed_free=float(training.get("guessed_free_weight", 0.35)),
        guessed_visible_surface=float(
            training.get("guessed_visible_surface_weight", 0.40)
        ),
        guessed_hidden_occupied=float(
            training.get("guessed_hidden_occupied_weight", 0.25)
        ),
        wrong_evidence_kl=float(training.get("wrong_evidence_kl_weight", 0.005)),
        support_bce=float(training.get("support_bce_weight", 0.5)),
        support_dice=float(training.get("support_dice_weight", 0.5)),
        support_boundary_emphasis=float(
            training.get("support_boundary_emphasis", 0.0)
        ),
        boundary_sigma=float(training.get("boundary_sigma", 3.0)),
    )
