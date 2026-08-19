from __future__ import annotations

import torch
from torch.nn import functional as F


def relative_se2_pose_losses(
    prediction: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    translation_weight: float = 1.0,
    yaw_weight: float = 0.5,
    refinement_gamma: float = 1.5,
    smooth_l1_beta_m: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Supervise all pose refinements against metric latest-from-frame GT."""

    stages = prediction["refinement_stages"]
    if stages.ndim != 4 or stages.shape[-1] != 4:
        raise ValueError("refinement_stages must have shape [S,B,N,4]")
    if target.shape != stages.shape[1:]:
        raise ValueError("relative pose target shape does not match prediction")
    if translation_weight < 0.0 or yaw_weight < 0.0:
        raise ValueError("pose loss weights cannot be negative")
    if refinement_gamma <= 0.0 or smooth_l1_beta_m <= 0.0:
        raise ValueError("pose refinement settings must be positive")

    frames = target.shape[1]
    non_reference = torch.ones(
        target.shape[:2],
        device=target.device,
        dtype=torch.bool,
    )
    non_reference[:, -1] = False
    denominator = non_reference.sum().clamp_min(1)
    stage_weights = refinement_gamma ** torch.arange(
        stages.shape[0],
        device=stages.device,
        dtype=stages.dtype,
    )
    stage_weights = stage_weights / stage_weights.sum()
    translation_losses = []
    yaw_losses = []
    for stage in stages:
        translation_per_axis = F.smooth_l1_loss(
            stage[..., :2],
            target[..., :2],
            beta=smooth_l1_beta_m,
            reduction="none",
        ).sum(dim=-1)
        translation_losses.append(
            (translation_per_axis * non_reference).sum() / denominator
        )
        predicted_direction = stage[..., 2:4]
        target_direction = target[..., 2:4]
        yaw_cosine = (predicted_direction * target_direction).sum(dim=-1)
        yaw_losses.append(
            ((1.0 - yaw_cosine.clamp(-1.0, 1.0)) * non_reference).sum()
            / denominator
        )
    translation = torch.stack(translation_losses)
    yaw = torch.stack(yaw_losses)
    translation_total = (stage_weights * translation).sum()
    yaw_total = (stage_weights * yaw).sum()
    total = translation_weight * translation_total + yaw_weight * yaw_total
    final_pose = prediction["relative_pose"]
    translation_error = torch.linalg.vector_norm(
        final_pose[..., :2] - target[..., :2],
        dim=-1,
    )
    yaw_dot = (final_pose[..., 2:4] * target[..., 2:4]).sum(dim=-1)
    yaw_error = torch.acos(yaw_dot.clamp(-1.0, 1.0))
    valid_count = non_reference.sum()
    return {
        "loss": total,
        "translation_loss": translation_total,
        "yaw_loss": yaw_total,
        "final_translation_mae_m": (
            (translation_error * non_reference).sum() / denominator
        ),
        "final_yaw_mae_rad": ((yaw_error * non_reference).sum() / denominator),
        "supervised_pose_count": valid_count.to(dtype=total.dtype),
        "history_frame_count": total.new_tensor(float(frames)),
    }


def relative_se2_metric_totals(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape or prediction.shape[-1] != 4:
        raise ValueError("pose prediction and target must share shape [B,N,4]")
    valid = torch.ones(target.shape[:2], device=target.device, dtype=torch.bool)
    valid[:, -1] = False
    translation_error = torch.linalg.vector_norm(
        prediction[..., :2] - target[..., :2],
        dim=-1,
    )
    yaw_dot = (prediction[..., 2:4] * target[..., 2:4]).sum(dim=-1)
    yaw_error = torch.acos(yaw_dot.clamp(-1.0, 1.0))
    return {
        "translation_error_sum_m": (translation_error * valid).sum(),
        "yaw_error_sum_rad": (yaw_error * valid).sum(),
        "pose_count": valid.sum(),
    }
