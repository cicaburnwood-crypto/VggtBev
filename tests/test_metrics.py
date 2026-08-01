from __future__ import annotations

import torch

from vggt_bev_method1.metrics import fov_complete_evidential_metrics


def test_perfect_complete_bev_reports_observed_guessed_confidence_separately() -> None:
    complete = torch.tensor([[[0, 255], [0, 255]]], dtype=torch.uint8)
    observed = torch.tensor([[[0, 255], [112, 112]]], dtype=torch.uint8)
    occupied = complete == 0
    observed_mask = observed != 112
    correct_evidence = torch.where(
        observed_mask,
        torch.full_like(complete, 10.0, dtype=torch.float32),
        torch.full_like(complete, 4.0, dtype=torch.float32),
    )
    alpha = torch.where(
        occupied,
        correct_evidence + 1.0,
        torch.ones_like(correct_evidence),
    )
    beta = torch.where(
        occupied,
        torch.ones_like(correct_evidence),
        correct_evidence + 1.0,
    )
    metrics = fov_complete_evidential_metrics(
        {
            "alpha_occupied": alpha,
            "beta_free": beta,
            "fov_support_probability": torch.ones_like(alpha),
        },
        complete,
        observed,
        complete != 112,
        target_extent_m=torch.tensor([6.5]),
        surface_tolerance_pixels=0,
    )
    assert float(metrics["occupied_iou"]) == 1.0
    assert float(metrics["free_iou"]) == 1.0
    assert float(metrics["occupied_precision"]) == 1.0
    assert float(metrics["occupied_recall"]) == 1.0
    assert metrics["observed_mean_confidence"] > metrics["guessed_mean_confidence"]
    assert metrics["confidence_gap"] > 0
    assert float(metrics["observed_confidence_bias"]) < 0
    assert float(metrics["guessed_confidence_bias"]) < 0
    assert float(metrics["observed_confidence_brier_correctness"]) >= 0
    assert float(metrics["guessed_confidence_brier_correctness"]) >= 0
    assert float(metrics["observed_confidence_ece"]) >= 0
    assert float(metrics["guessed_confidence_ece"]) >= 0
    assert float(metrics["observed_free_false_occupied_rate"]) == 0.0
    assert float(metrics["observed_surface_recall"]) == 1.0
    assert float(metrics["observed_direct_balanced_accuracy"]) == 1.0
    assert float(metrics["guessed_occupied_iou"]) == 1.0
    assert float(metrics["guessed_occupied_precision"]) == 1.0
    assert float(metrics["guessed_occupied_recall"]) == 1.0
    assert (
        metrics["guessed_occupied_iou_gain_over_all_occupied"]
        > 0
    )
    assert float(metrics["support_iou"]) == 1.0
    assert float(metrics["boundary_chamfer_m"]) == 0.0


def test_surface_metric_uses_same_tolerance_as_training() -> None:
    complete = torch.full((1, 9, 9), 255, dtype=torch.uint8)
    complete[:, 4, 4] = 0
    observed = complete.clone()
    occupied = torch.zeros_like(complete, dtype=torch.bool)
    occupied[:, 4, 6] = True
    evidence = torch.full_like(complete, 8.0, dtype=torch.float32)
    alpha = torch.where(occupied, evidence, torch.ones_like(evidence))
    beta = torch.where(occupied, torch.ones_like(evidence), evidence)
    prediction = {
        "alpha_occupied": alpha,
        "beta_free": beta,
        "fov_support_probability": torch.ones_like(evidence),
    }
    exact = fov_complete_evidential_metrics(
        prediction,
        complete,
        observed,
        complete != 112,
        target_extent_m=torch.tensor([6.5]),
        surface_tolerance_pixels=0,
    )
    tolerant = fov_complete_evidential_metrics(
        prediction,
        complete,
        observed,
        complete != 112,
        target_extent_m=torch.tensor([6.5]),
        surface_tolerance_pixels=2,
    )
    assert float(exact["observed_surface_recall"]) == 0.0
    assert float(tolerant["observed_surface_recall_with_tolerance"]) == 1.0
