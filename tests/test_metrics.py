from __future__ import annotations

import torch

from vggt_bev_method1.metrics import (
    branch_metrics,
    evidential_complete_metrics,
    observed_categorical_metrics,
)


def test_method1_metrics_cover_free_occupied_unknown_and_boundary() -> None:
    target = torch.tensor([[[0, 255], [112, 255]]], dtype=torch.uint8)
    prediction = {
        "occupancy_logit": torch.tensor([[[10.0, -10.0], [0.0, -10.0]]]),
        "observed_logit": torch.tensor([[[10.0, 10.0], [-10.0, 10.0]]]),
    }
    metrics = branch_metrics(
        prediction,
        target,
        supervision="observed",
        target_extent_m=torch.tensor([2.0]),
    )
    for key in (
        "occupied_iou",
        "free_iou",
        "known_macro_f1",
        "unknown_iou",
        "three_class_macro_f1",
        "occupancy_ece",
        "observation_ece",
        "boundary_chamfer_m",
    ):
        assert key in metrics
        assert torch.isfinite(metrics[key])
    assert float(metrics["three_class_macro_f1"]) == 1.0


def test_paired_metrics_separate_observed_and_guessed_evidence() -> None:
    masked = torch.tensor([[[0, 255], [112, 112]]], dtype=torch.uint8)
    complete = torch.tensor([[[0, 255], [0, 255]]], dtype=torch.uint8)
    class_logits = torch.full((1, 3, 2, 2), -10.0)
    class_logits[0, 2, 0, 0] = 10.0
    class_logits[0, 1, 0, 1] = 10.0
    class_logits[0, 0, 1] = 10.0
    observed_metrics = observed_categorical_metrics(
        {"class_logits": class_logits},
        masked,
        target_extent_m=torch.tensor([2.0]),
    )
    assert float(observed_metrics["three_class_macro_f1"]) == 1.0

    alpha = torch.tensor([[[20.0, 2.0], [3.0, 2.0]]])
    beta = torch.tensor([[[2.0, 20.0], [2.0, 3.0]]])
    complete_metrics = evidential_complete_metrics(
        {
            "alpha_occupied": alpha,
            "beta_free": beta,
            "runtime_fov_support": torch.ones_like(alpha, dtype=torch.bool),
        },
        complete,
        masked,
        complete != 112,
        target_extent_m=torch.tensor([2.0]),
        surface_tolerance_pixels=0,
    )
    for key in (
        "known_macro_f1",
        "confidence_brier",
        "observed_accuracy",
        "guessed_accuracy",
        "observed_mean_evidence_confidence",
        "guessed_mean_evidence_confidence",
    ):
        assert key in complete_metrics
        assert torch.isfinite(complete_metrics[key])
    assert (
        complete_metrics["observed_mean_evidence_confidence"]
        > complete_metrics["guessed_mean_evidence_confidence"]
    )
