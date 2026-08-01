from __future__ import annotations

import torch

from vggt_bev_method1.models import (
    ScaleFitConfig,
    fit_metric_scale_targets,
    metric_scale_losses,
)
from vggt_bev_method1.models.reprojection import metric_points_to_vggt_units


def test_robust_scale_fit_recovers_metric_ratio_with_outliers() -> None:
    torch.manual_seed(3)
    vggt = torch.rand(2, 3, 24, 32) * 5.0 + 0.5
    gt = 2.5 * vggt
    gt[:, :, :2, :2] *= 10.0
    confidence = torch.full_like(vggt, 10.0)
    target = fit_metric_scale_targets(
        gt,
        torch.ones_like(gt, dtype=torch.bool),
        vggt,
        confidence,
        ScaleFitConfig(
            minimum_valid_pixels=100,
            maximum_pixels_per_frame=1000,
        ),
    )
    torch.testing.assert_close(target["lambda_gt"], torch.full((2,), 2.5))
    assert target["target_valid"].all()


def test_scale_loss_is_minimal_at_gt() -> None:
    vggt = torch.rand(1, 2, 20, 20) + 1.0
    target = fit_metric_scale_targets(
        3.0 * vggt,
        torch.ones_like(vggt, dtype=torch.bool),
        vggt,
        torch.full_like(vggt, 10.0),
        ScaleFitConfig(minimum_valid_pixels=20),
    )
    log_scale = torch.tensor([3.0]).log().requires_grad_()
    losses = metric_scale_losses(
        {
            "log_lambda_m_per_vggt": log_scale,
            "lambda_m_per_vggt": log_scale.exp(),
            "log_variance": torch.zeros(1, requires_grad=True),
        },
        target,
    )
    (losses["scale"] + losses["depth_scale"] + losses["uncertainty"]).backward()
    assert float(losses["scale"].detach()) < 1e-6
    assert float(losses["depth_scale"].detach()) < 1e-6


def test_scale_quality_weights_survive_batch_normalization() -> None:
    predicted_log = torch.ones(2, requires_grad=True)
    target = {
        "log_lambda_gt": torch.zeros(2),
        "quality_weight": torch.tensor([0.1, 1.0]),
        "target_valid": torch.ones(2, dtype=torch.bool),
        "dense_inlier_mask": torch.zeros(2, 1, 1, 1, dtype=torch.bool),
        "dense_weight": torch.zeros(2, 1, 1, 1),
        "vggt_depth": torch.ones(2, 1, 1, 1),
        "gt_depth_m": torch.ones(2, 1, 1, 1),
    }
    losses = metric_scale_losses(
        {
            "log_lambda_m_per_vggt": predicted_log,
            "lambda_m_per_vggt": predicted_log.exp(),
        },
        target,
        smooth_l1_beta=0.1,
    )
    # SmoothL1(1, beta=.1) = .95. The mean keeps the absolute quality
    # magnitude: (.1*.95 + 1*.95) / 2 = .5225.
    torch.testing.assert_close(losses["scale"], torch.tensor(0.5225))


def test_metric_reprojection_divides_by_lambda() -> None:
    points_m = torch.tensor([[[2.0, 0.0, 4.0]]])
    converted = metric_points_to_vggt_units(points_m, torch.tensor([2.0]))
    torch.testing.assert_close(converted, torch.tensor([[[1.0, 0.0, 2.0]]]))
