from __future__ import annotations

import torch

from vggt_bev_method1.losses import (
    EvidentialModelLossWeights,
    LossWeights,
    ObservedModelLossWeights,
    branch_loss,
    complete_evidential_model_loss,
    dual_method1_loss,
    masked_target_classes,
    observed_model_loss,
)


def test_complete_loss_ignores_unknown_cells() -> None:
    target = torch.tensor([[[0, 255], [112, 255]]], dtype=torch.uint8)
    first = {
        "occupancy_logit": torch.zeros(1, 2, 2),
        "observed_logit": torch.zeros(1, 2, 2),
    }
    second = {
        "occupancy_logit": first["occupancy_logit"].clone(),
        "observed_logit": first["observed_logit"].clone(),
    }
    second["occupancy_logit"][0, 1, 0] = 100.0
    weights = LossWeights()
    loss_a = branch_loss(first, target, supervision="complete", weights=weights)
    loss_b = branch_loss(second, target, supervision="complete", weights=weights)
    assert torch.allclose(loss_a["loss"], loss_b["loss"])


def test_observed_dual_loss_supervises_both_products() -> None:
    target = torch.tensor([[[0, 255], [112, 255]]], dtype=torch.uint8)
    branch = {
        "occupancy_logit": torch.zeros(1, 2, 2, requires_grad=True),
        "observed_logit": torch.zeros(1, 2, 2, requires_grad=True),
    }
    prediction = {"single": branch, "merged": branch}
    batch = {"single_target": target, "merged_target": target}
    result = dual_method1_loss(
        prediction,
        batch,
        supervision="observed",
        weights=LossWeights(),
    )
    result["loss"].backward()
    assert branch["occupancy_logit"].grad is not None
    assert branch["observed_logit"].grad is not None


def test_joint_loss_supervises_observed_and_complete_outputs() -> None:
    observed_target = torch.tensor([[[0, 255], [112, 255]]], dtype=torch.uint8)
    complete_target = torch.tensor([[[0, 255], [255, 255]]], dtype=torch.uint8)

    def output_branch() -> dict[str, torch.Tensor]:
        return {
            "occupancy_logit": torch.zeros(1, 2, 2, requires_grad=True),
            "observed_logit": torch.zeros(1, 2, 2, requires_grad=True),
        }

    branches = {
        extent: {task: output_branch() for task in ("observed", "complete")}
        for extent in ("single", "merged")
    }
    batch = {
        "single_observed_target": observed_target,
        "merged_observed_target": observed_target,
        "single_complete_target": complete_target,
        "merged_complete_target": complete_target,
    }
    result = dual_method1_loss(
        branches,
        batch,
        supervision="joint",
        weights=LossWeights(),
    )
    result["loss"].backward()
    assert "observed_loss" in result
    assert "complete_loss" in result
    for extent in ("single", "merged"):
        for task in ("observed", "complete"):
            assert branches[extent][task]["occupancy_logit"].grad is not None


def test_direct_observed_loss_uses_unknown_free_occupied_classes() -> None:
    target = torch.tensor([[[112, 255], [0, 255]]], dtype=torch.uint8)
    assert torch.equal(
        masked_target_classes(target),
        torch.tensor([[[0, 1], [2, 1]]]),
    )
    logits = torch.zeros(1, 3, 2, 2, requires_grad=True)
    result = observed_model_loss(
        {
            "single": {"class_logits": logits},
            "merged": {"class_logits": logits},
        },
        {
            "single_fov_visible_target": target,
            "merged_fov_visible_target": target,
        },
        weights=ObservedModelLossWeights(),
        surface_tolerance_single_pixels=0,
        surface_tolerance_merged_pixels=0,
    )
    result["loss"].backward()
    assert logits.grad is not None
    assert torch.isfinite(result["loss"])


def test_complete_evidential_loss_learns_full_map_and_mask_relationship() -> None:
    observed = torch.tensor([[[0, 255], [112, 112]]], dtype=torch.uint8)
    complete = torch.tensor([[[0, 255], [0, 255]]], dtype=torch.uint8)
    alpha = torch.full((1, 2, 2), 2.0, requires_grad=True)
    beta = torch.full((1, 2, 2), 2.0, requires_grad=True)

    def prediction() -> dict[str, torch.Tensor]:
        strength = alpha + beta
        return {
            "alpha_occupied": alpha,
            "beta_free": beta,
            "occupancy_probability": alpha / strength,
            "epistemic_uncertainty": 2.0 / strength,
        }

    result = complete_evidential_model_loss(
        {"single": prediction(), "merged": prediction()},
        {
            "single_fov_complete_target": complete,
            "merged_fov_complete_target": complete,
            "single_fov_visible_target": observed,
            "merged_fov_visible_target": observed,
            "single_fov_support_target": complete != 112,
            "merged_fov_support_target": complete != 112,
        },
        guessed_class_weights=torch.ones(2),
        weights=EvidentialModelLossWeights(),
        regularizer_scale=1.0,
        guessed_supervision_scale=1.0,
        surface_tolerance_single_pixels=0,
        surface_tolerance_merged_pixels=0,
    )
    result["loss"].backward()
    assert alpha.grad is not None
    assert beta.grad is not None
    assert torch.isfinite(result["loss"])
    assert "single_observed_mean_strength" in result
    assert "single_guessed_mean_strength" in result


def test_evidential_nll_pushes_evidence_toward_complete_gt_class() -> None:
    observed = torch.tensor([[[112]]], dtype=torch.uint8)
    complete = torch.tensor([[[0]]], dtype=torch.uint8)
    alpha = torch.tensor([[[2.0]]], requires_grad=True)
    beta = torch.tensor([[[2.0]]], requires_grad=True)
    branch = {
        "alpha_occupied": alpha,
        "beta_free": beta,
    }
    result = complete_evidential_model_loss(
        {"single": branch, "merged": branch},
        {
            "single_fov_complete_target": complete,
            "merged_fov_complete_target": complete,
            "single_fov_visible_target": observed,
            "merged_fov_visible_target": observed,
            "single_fov_support_target": complete != 112,
            "merged_fov_support_target": complete != 112,
        },
        guessed_class_weights=torch.ones(2),
        weights=EvidentialModelLossWeights(
            guessed_overlap=0.0,
            incorrect_evidence=0.0,
            observation_relation=0.0,
            calibration=0.0,
        ),
        regularizer_scale=0.0,
        guessed_supervision_scale=1.0,
        surface_tolerance_single_pixels=0,
        surface_tolerance_merged_pixels=0,
    )
    result["loss"].backward()
    assert alpha.grad is not None and float(alpha.grad) < 0.0
    assert beta.grad is not None and float(beta.grad) > 0.0


def test_complete_occupancy_has_zero_gradient_outside_fov() -> None:
    complete = torch.tensor([[[0, 112]]], dtype=torch.uint8)
    visible = torch.tensor([[[112, 112]]], dtype=torch.uint8)
    alpha = torch.full((1, 1, 2), 2.0, requires_grad=True)
    beta = torch.full((1, 1, 2), 2.0, requires_grad=True)
    branch = {"alpha_occupied": alpha, "beta_free": beta}
    result = complete_evidential_model_loss(
        {"single": branch, "merged": branch},
        {
            "single_fov_complete_target": complete,
            "merged_fov_complete_target": complete,
            "single_fov_visible_target": visible,
            "merged_fov_visible_target": visible,
            "single_fov_support_target": complete != 112,
            "merged_fov_support_target": complete != 112,
        },
        guessed_class_weights=torch.ones(2),
        weights=EvidentialModelLossWeights(
            guessed_overlap=0.0,
            incorrect_evidence=0.0,
            observation_relation=0.0,
            calibration=0.0,
        ),
        regularizer_scale=0.0,
        guessed_supervision_scale=1.0,
        surface_tolerance_single_pixels=0,
        surface_tolerance_merged_pixels=0,
    )
    result["loss"].backward()
    assert float(alpha.grad[0, 0, 1]) == 0.0
    assert float(beta.grad[0, 0, 1]) == 0.0
