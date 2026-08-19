import math

import torch
import pytest

from vggt_bev_method1.cli_train_p1b import (
    _scheduled_ramp,
    _scheduled_role_weights,
)

from vggt_bev_method1.models import (
    P1CHead,
    RelativeSE2PoseHead,
    compose_se2_residual,
)
from vggt_bev_method1.p1c_losses import (
    relative_se2_metric_totals,
    relative_se2_pose_losses,
)
from vggt_bev_method1.p1b_config import (
    load_p1b_config,
    validate_p1b_config,
)


def test_se2_residual_uses_transform_composition() -> None:
    pose = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]])
    residual = torch.tensor([[[0.0, 0.0, math.pi / 2.0]]])

    composed = compose_se2_residual(pose, residual)

    assert torch.allclose(
        composed,
        torch.tensor([[[0.0, 1.0, 1.0, 0.0]]]),
        atol=1e-6,
    )


def test_pose_head_keeps_latest_frame_identity() -> None:
    head = RelativeSE2PoseHead(
        input_dim=16,
        hidden_dim=16,
        heads=4,
        layers=2,
        refinements=3,
        maximum_history=5,
    )

    output = head(torch.randn(2, 4, 17, 16))

    assert output["relative_pose"].shape == (2, 4, 4)
    assert output["refinement_stages"].shape == (3, 2, 4, 4)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(2, -1)
    assert torch.equal(output["relative_pose"][:, -1], expected)


def test_p1c_merged_output_is_conditioned_by_predicted_pose() -> None:
    head = P1CHead(
        probability_model="evidential",
        cached_layers=(1,),
        spatial_scales=(1.0,),
        vggt_token_dim=16,
        hidden_dim=8,
        heads=2,
        decoder_layers=1,
        scale_decoder_layers=1,
        self_attention_mode="linear",
        cross_attention_mode="deformable",
        deformable_samples=1,
        cross_query_chunk_size=64,
        single_latent_bev_size=8,
        merged_latent_bev_size=8,
        single_output_size=8,
        merged_output_size=8,
        single_bev_extent_m=6.5,
        merged_bev_extent_m=6.5,
        pose_hidden_dim=16,
        pose_attention_heads=4,
        pose_layers=1,
        pose_refinements=2,
        maximum_history=4,
    )
    extraction = {
        "tokens": {1: torch.randn(2, 3, 4, 16)},
        "camera_register_tokens": torch.randn(2, 3, 17, 16),
        "patch_grid": (2, 2),
    }

    output = head(
        extraction,
        enabled_bev_branches=("merged",),
        include_scale=False,
        bev_objective="fov_support_and_observed_gate",
    )

    assert output["merged_bev"]["fov_support_logit"].shape == (2, 8, 8)
    assert output["merged_bev"]["observed_gate_logit"].shape == (2, 8, 8)
    assert output["pose"]["relative_pose"].shape == (2, 3, 4)


def test_pose_loss_excludes_latest_identity_and_supervises_all_refinements() -> None:
    target = torch.tensor(
        [[[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]]
    )
    stages = torch.stack(
        (
            torch.tensor(
                [[[0.0, 0.0, 0.0, 1.0], [9.0, 9.0, 1.0, 0.0]]]
            ),
            target.clone(),
        )
    )
    prediction = {
        "relative_pose": target.clone(),
        "refinement_stages": stages,
    }

    losses = relative_se2_pose_losses(prediction, target)
    totals = relative_se2_metric_totals(prediction["relative_pose"], target)

    assert losses["loss"] > 0
    assert losses["final_translation_mae_m"] == 0
    assert losses["supervised_pose_count"] == 1
    assert totals["pose_count"] == 1
    assert totals["translation_error_sum_m"] == 0


def test_p1c_pose_only_config_is_fail_closed() -> None:
    config = load_p1b_config("configs/p1c_pose_only_v1.toml")

    assert config["training"]["pipeline"] == "P1C-NLL"
    assert config["training"]["bev_objective"] == "pose_only"
    config["teacher_cache"]["mode"] = "read"
    with pytest.raises(ValueError, match="requires live VGGT"):
        validate_p1b_config(config)


def test_p1c_stage2_requires_pose_parent_and_ramps_boundaries() -> None:
    config = load_p1b_config("configs/p1c_pose_fov_gate_v1.toml")
    assert config["training"]["pose_warmstart_checkpoint"]

    assert _scheduled_ramp(
        10, 100, start_fraction=0.10, ramp_fraction=0.30
    ) == 0.0
    assert _scheduled_ramp(
        25, 100, start_fraction=0.10, ramp_fraction=0.30
    ) == pytest.approx(0.5)
    assert _scheduled_role_weights(
        0.25, 0.45, 0.20, 0.10, boundary_scale=0.0
    ) == pytest.approx((0.90, 0.0, 0.0, 0.10))
    assert _scheduled_role_weights(
        0.25, 0.45, 0.20, 0.10, boundary_scale=1.0
    ) == pytest.approx((0.25, 0.45, 0.20, 0.10))

    config["training"]["pose_warmstart_checkpoint"] = ""
    with pytest.raises(ValueError, match="requires a verified pose_only"):
        validate_p1b_config(config)
