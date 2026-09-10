from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ScaleFitConfig:
    confidence_threshold: float = 0.2
    minimum_depth_m: float = 0.05
    maximum_depth_m: float = 80.0
    residual_threshold_log: float = 0.25
    huber_delta_log: float = 0.05
    minimum_valid_pixels: int = 1024
    maximum_pixels_per_frame: int = 8192
    minimum_inlier_ratio: float = 0.35
    minimum_quality_weight: float = 0.10
    quality_sigma_log: float = 0.10
    desired_log_depth_range: float = 2.0
    irls_iterations: int = 4


def vggt_confidence_probability(confidence: torch.Tensor) -> torch.Tensor:
    """Convert VGGT's positive confidence parameter to a [0,1] weight."""

    return ((confidence - 1.0) / confidence.clamp_min(1.0)).clamp(0.0, 1.0)


def _resize_to(
    value: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str,
) -> torch.Tensor:
    batch, frames = value.shape[:2]
    resized = F.interpolate(
        value.reshape(batch * frames, 1, *value.shape[-2:]).float(),
        size=size,
        mode=mode,
    )
    return resized.reshape(batch, frames, *size)


def _deterministic_subset(indices: torch.Tensor, maximum: int) -> torch.Tensor:
    if indices.numel() <= maximum:
        return indices
    positions = torch.linspace(
        0,
        indices.numel() - 1,
        maximum,
        device=indices.device,
    ).round().long()
    return indices[positions]


@torch.no_grad()
def fit_metric_scale_targets(
    gt_depth_m: torch.Tensor,
    gt_valid_mask: torch.Tensor,
    vggt_depth: torch.Tensor,
    vggt_confidence: torch.Tensor,
    config: ScaleFitConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Robustly fit one metric scale label for each VGGT window.

    All depths must represent camera-axis z-depth.  The fit uses a log-ratio
    median, rejects non-scale residuals, and refines the label with
    confidence-weighted Huber IRLS.
    """

    if config is None:
        config = ScaleFitConfig()

    if vggt_depth.ndim == 5 and vggt_depth.shape[-1] == 1:
        vggt_depth = vggt_depth[..., 0]
    if vggt_confidence.ndim == 5 and vggt_confidence.shape[-1] == 1:
        vggt_confidence = vggt_confidence[..., 0]
    if any(value.ndim != 4 for value in (gt_depth_m, gt_valid_mask, vggt_depth, vggt_confidence)):
        raise ValueError("scale fitting expects [B,N,H,W] tensors")
    if gt_depth_m.shape[:2] != vggt_depth.shape[:2]:
        raise ValueError("GT and VGGT frame dimensions must match")
    size = vggt_depth.shape[-2:]
    if gt_depth_m.shape[-2:] != size:
        gt_depth_m = _resize_to(gt_depth_m, size, mode="nearest")
        gt_valid_mask = _resize_to(
            gt_valid_mask.to(torch.float32), size, mode="nearest"
        ) > 0.5
    if vggt_confidence.shape[-2:] != size:
        vggt_confidence = _resize_to(vggt_confidence, size, mode="bilinear")

    confidence = vggt_confidence_probability(vggt_confidence.float())
    gt_depth_m = gt_depth_m.float()
    vggt_depth = vggt_depth.float()
    valid = (
        gt_valid_mask.bool()
        & torch.isfinite(gt_depth_m)
        & torch.isfinite(vggt_depth)
        & torch.isfinite(confidence)
        & (gt_depth_m >= config.minimum_depth_m)
        & (gt_depth_m <= config.maximum_depth_m)
        & (vggt_depth > 0)
        & (confidence >= config.confidence_threshold)
    )

    batch, frames, height, width = valid.shape
    log_lambda = gt_depth_m.new_zeros(batch)
    quality = gt_depth_m.new_zeros(batch)
    residual_median = gt_depth_m.new_full((batch,), float("inf"))
    inlier_ratio = gt_depth_m.new_zeros(batch)
    valid_count = torch.zeros(batch, device=gt_depth_m.device, dtype=torch.long)
    depth_range_coverage = gt_depth_m.new_zeros(batch)
    frame_agreement = gt_depth_m.new_zeros(batch)
    target_valid = torch.zeros(batch, device=gt_depth_m.device, dtype=torch.bool)
    dense_inlier = torch.zeros_like(valid)
    dense_weight = torch.zeros_like(gt_depth_m)

    flat_gt = gt_depth_m.reshape(batch, frames, -1)
    flat_vggt = vggt_depth.reshape(batch, frames, -1)
    flat_conf = confidence.reshape(batch, frames, -1)
    flat_valid = valid.reshape(batch, frames, -1)
    for batch_index in range(batch):
        selected_global: list[torch.Tensor] = []
        selected_frame: list[torch.Tensor] = []
        for frame_index in range(frames):
            indices = torch.nonzero(
                flat_valid[batch_index, frame_index],
                as_tuple=False,
            )[:, 0]
            indices = _deterministic_subset(
                indices,
                config.maximum_pixels_per_frame,
            )
            if indices.numel():
                selected_global.append(indices + frame_index * height * width)
                selected_frame.append(
                    torch.full_like(indices, frame_index)
                )
        if not selected_global:
            continue
        indices = torch.cat(selected_global)
        frame_ids = torch.cat(selected_frame)
        gt_values = flat_gt[batch_index].reshape(-1)[indices]
        vggt_values = flat_vggt[batch_index].reshape(-1)[indices]
        weights = flat_conf[batch_index].reshape(-1)[indices].clamp_min(1e-3)
        y = gt_values.log() - vggt_values.log()
        initial = y.median()
        initial_residual = (y - initial).abs()
        inlier = initial_residual < config.residual_threshold_log
        valid_count[batch_index] = indices.numel()
        inlier_ratio[batch_index] = inlier.float().mean()
        if int(inlier.sum()) < config.minimum_valid_pixels:
            continue

        y_inlier = y[inlier]
        weights_inlier = weights[inlier]
        estimate = initial
        for _ in range(config.irls_iterations):
            residual = y_inlier - estimate
            huber_weight = torch.where(
                residual.abs() <= config.huber_delta_log,
                torch.ones_like(residual),
                config.huber_delta_log / residual.abs().clamp_min(1e-6),
            )
            combined = weights_inlier * huber_weight
            estimate = (combined * y_inlier).sum() / combined.sum().clamp_min(1e-6)

        final_residual = (y_inlier - estimate).abs()
        residual_value = final_residual.median()
        log_lambda[batch_index] = estimate
        residual_median[batch_index] = residual_value
        depth_range = (
            gt_values[inlier].amax().log() - gt_values[inlier].amin().log()
        )
        coverage = (depth_range / config.desired_log_depth_range).clamp(0.0, 1.0)
        depth_range_coverage[batch_index] = coverage

        per_frame = []
        for frame_index in range(frames):
            frame_values = y[inlier & (frame_ids == frame_index)]
            if frame_values.numel() >= max(16, config.minimum_valid_pixels // frames):
                per_frame.append(frame_values.median())
        if len(per_frame) <= 1:
            agreement = torch.ones_like(estimate)
        else:
            frame_scales = torch.stack(per_frame)
            agreement = torch.exp(
                -(frame_scales - frame_scales.median()).abs().median()
                / config.quality_sigma_log
            )
        frame_agreement[batch_index] = agreement
        quality_value = (
            torch.exp(-residual_value / config.quality_sigma_log)
            * inlier_ratio[batch_index]
            * coverage
            * agreement
        ).clamp(0.0, 1.0)
        if inlier_ratio[batch_index] < config.minimum_inlier_ratio:
            quality_value.zero_()
        quality[batch_index] = quality_value
        target_valid[batch_index] = (
            quality_value > 0
        ) & (quality_value >= config.minimum_quality_weight)
        kept_indices = indices[inlier]
        dense_inlier.reshape(batch, -1)[batch_index, kept_indices] = True
        dense_weight.reshape(batch, -1)[batch_index, kept_indices] = weights_inlier

    return {
        "log_lambda_gt": log_lambda,
        "lambda_gt": log_lambda.exp(),
        "quality_weight": quality,
        "target_valid": target_valid,
        "depth_alignment_residual": residual_median,
        "inlier_ratio": inlier_ratio,
        "valid_pixel_count": valid_count,
        "depth_range_coverage": depth_range_coverage,
        "frame_scale_agreement": frame_agreement,
        "dense_inlier_mask": dense_inlier,
        "dense_weight": dense_weight,
        "gt_depth_m": gt_depth_m,
        "vggt_depth": vggt_depth,
    }


def metric_scale_losses(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    smooth_l1_beta: float = 0.1,
) -> dict[str, torch.Tensor]:
    predicted_log = prediction["log_lambda_m_per_vggt"]
    target_valid = target["target_valid"]
    valid = target_valid.to(predicted_log.dtype)
    quality = target["quality_weight"].to(predicted_log.dtype) * valid
    valid_count = valid.sum().clamp_min(1.0)
    scalar_error = F.smooth_l1_loss(
        predicted_log,
        target["log_lambda_gt"],
        beta=smooth_l1_beta,
        reduction="none",
    )
    # Do not divide by sum(quality): with per-rank batch size one that makes
    # q * loss / q == loss and silently removes label-quality weighting.
    scalar = (quality * scalar_error).sum() / valid_count

    dense_mask = target["dense_inlier_mask"]
    predicted_metric_depth = (
        predicted_log.exp()[:, None, None, None] * target["vggt_depth"]
    )
    dense_error = (
        predicted_metric_depth.clamp_min(1e-6).log()
        - target["gt_depth_m"].clamp_min(1e-6).log()
    )
    dense_huber = F.smooth_l1_loss(
        dense_error,
        torch.zeros_like(dense_error),
        beta=smooth_l1_beta,
        reduction="none",
    )
    weights = torch.where(dense_mask, target["dense_weight"], 0.0)
    flat_weights = weights.flatten(1)
    per_sample_weight = flat_weights.sum(dim=1)
    per_sample_dense = (
        (weights * dense_huber).flatten(1).sum(dim=1)
        / per_sample_weight.clamp_min(1e-6)
    )
    dense_valid = target_valid & (per_sample_weight > 0)
    dense = (
        target["quality_weight"].to(predicted_log.dtype)
        * dense_valid.to(predicted_log.dtype)
        * per_sample_dense
    ).sum() / dense_valid.to(predicted_log.dtype).sum().clamp_min(1.0)

    uncertainty = predicted_log.new_zeros(())
    if "log_variance" in prediction:
        absolute_error = (
            predicted_log - target["log_lambda_gt"]
        ).abs()
        nll = (
            (-prediction["log_variance"]).exp() * absolute_error
            + prediction["log_variance"]
        )
        uncertainty = (quality * nll).sum() / valid_count
    return {
        "scale": scalar,
        "depth_scale": dense,
        "uncertainty": uncertainty,
        "valid_scale_fraction": target["target_valid"].float().mean(),
    }


def metric_scale_metrics(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    valid = target["target_valid"]
    if not bool(valid.any()):
        zero = prediction["lambda_m_per_vggt"].new_zeros(())
        return {
            "scale_relative_error": zero,
            "scale_log_error": zero,
            "scale_valid_fraction": zero,
        }
    predicted = prediction["lambda_m_per_vggt"][valid]
    expected = target["lambda_gt"][valid]
    return {
        "scale_relative_error": ((predicted - expected).abs() / expected).median(),
        "scale_log_error": (
            prediction["log_lambda_m_per_vggt"][valid]
            - target["log_lambda_gt"][valid]
        ).abs().median(),
        "scale_valid_fraction": valid.float().mean(),
    }
