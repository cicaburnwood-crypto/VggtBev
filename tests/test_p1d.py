from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from vggt_bev_method1.data.vggt_unit_targets import (
    regrid_merged_metric_targets_to_vggt_units,
)
from vggt_bev_method1.models import P1DSystem
from vggt_bev_method1.models.attention import MultiheadAttention
from vggt_bev_method1.p1d_config import load_p1d_config
from vggt_bev_method1.p1d_losses import (
    P1DAdditionalLossWeights,
    p1d_bev_loss,
)
from vggt_bev_method1.p1b_losses import P1BLossWeights


class _Adapter(nn.Module):
    def aggregate(self, images: torch.Tensor) -> dict:
        raise NotImplementedError


def _model() -> P1DSystem:
    return P1DSystem(
        _Adapter(),
        probability_model="evidential",
        cached_layers=(1,),
        spatial_scales=(1.0,),
        vggt_token_dim=16,
        hidden_dim=8,
        heads=2,
        decoder_layers=1,
        scale_decoder_layers=1,
        self_attention_mode="linear",
        cross_attention_mode="linear",
        deformable_samples=2,
        cross_query_chunk_size=16,
        merged_latent_bev_size=4,
        merged_output_size=8,
        merged_extent_vggt=6.5,
        predict_scale_uncertainty=True,
        implicit_geometry_hidden_dim=8,
        implicit_geometry_heads=2,
        implicit_geometry_layers=1,
        maximum_history=4,
        maximum_prefix_tokens=3,
        frame_reliability_hidden_dim=8,
        training_frame_dropout_probability=0.5,
    )


def test_p1d_config_template_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_p1d_config(root / "configs/p1d_direct_merged_scale_template.toml")
    assert config["training"]["stage"] == "joint"
    assert config["model"]["training_frame_dropout_probability"] == 0.15


def test_weighted_linear_attention_matches_uniform_attention() -> None:
    torch.manual_seed(4)
    attention = MultiheadAttention(8, 2, mode="linear")
    query = torch.randn(2, 5, 8)
    context = torch.randn(2, 7, 8)
    expected = attention(query, context)
    actual = attention(query, context, torch.ones(2, 7))
    torch.testing.assert_close(actual, expected)


def test_p1d_is_one_pass_rgb_only_and_reliability_starts_uniform() -> None:
    torch.manual_seed(5)
    model = _model().eval()
    extraction = {
        "tokens": {1: torch.randn(2, 3, 4, 16)},
        "camera_register_tokens": torch.randn(2, 3, 3, 16),
        "patch_grid": (2, 2),
    }
    output = model.forward_head(extraction)
    torch.testing.assert_close(
        output["frame_reliability"],
        torch.ones(2, 3),
    )
    assert output["merged_bev"]["occupancy_probability"].shape == (2, 8, 8)
    assert output["runtime_inputs"] == ("rgb_window",)
    assert output["runtime_passes"] == 1
    assert not output["extrinsic_input_present"]
    assert not output["single_bev_present"]
    assert not output["runtime_postprocessing_present"]


def test_training_dropout_keeps_latest_frame() -> None:
    torch.manual_seed(7)
    model = _model().train()
    extraction = {
        "tokens": {1: torch.randn(2, 3, 4, 16)},
        "camera_register_tokens": torch.randn(2, 3, 3, 16),
        "patch_grid": (2, 2),
    }
    output = model.forward_head(extraction, assemble_runtime_outputs=False)
    assert bool(output["frame_keep_mask"][:, -1].all())
    torch.testing.assert_close(
        output["effective_frame_reliability"].sum(dim=1),
        output["frame_keep_mask"].sum(dim=1).float(),
    )


def test_temporal_targets_use_same_vggt_unit_lookup() -> None:
    complete = torch.full((1, 8, 8), 255, dtype=torch.uint8)
    visible = torch.full((1, 8, 8), 112, dtype=torch.uint8)
    visible[:, 2:6, 2:6] = 255
    support = torch.ones(1, 8, 8, dtype=torch.bool)
    valid = torch.ones_like(support)
    latest_observed = torch.zeros_like(support)
    latest_observed[:, 4:6, 3:5] = True
    latest_support = torch.zeros_like(support)
    latest_support[:, 3:7, 2:6] = True
    target = regrid_merged_metric_targets_to_vggt_units(
        complete,
        visible,
        support,
        valid,
        torch.ones(1),
        torch.ones(1, dtype=torch.bool),
        source_extent_m=8.0,
        target_extent_vggt=8.0,
        target_size=8,
        latest_observed_free_metric=latest_observed,
        latest_support_metric=latest_support,
    )
    assert torch.equal(target["latest_observed_free_target"], latest_observed)
    assert torch.equal(target["latest_support_target"], latest_support)
    assert bool(target["history_support_region"].any())


def test_p1d_loss_backpropagates_history_and_hard_nll() -> None:
    complete = torch.full((1, 16, 16), 112, dtype=torch.uint8)
    complete[:, 2:14, 2:14] = 255
    complete[:, 4:6, 8:12] = 0
    visible = torch.full_like(complete, 112)
    visible[:, 8:14, 4:12] = complete[:, 8:14, 4:12]
    support = complete != 112
    latest_support = torch.zeros_like(support)
    latest_support[:, 9:14, 6:10] = True
    latest_observed = latest_support & (visible == 255)
    raw_evidence = torch.randn(1, 2, 16, 16, requires_grad=True)
    alpha = torch.nn.functional.softplus(raw_evidence[:, 0]) + 1.0
    beta = torch.nn.functional.softplus(raw_evidence[:, 1]) + 1.0
    gate_logit = torch.randn(1, 16, 16, requires_grad=True)
    support_logit = torch.randn(1, 16, 16, requires_grad=True)
    prediction = {
        "guessed": {
            "alpha_occupied": alpha,
            "beta_free": beta,
            "occupancy_probability": alpha / (alpha + beta),
        },
        "observed_gate_logit": gate_logit,
        "fov_support_logit": support_logit,
        "fov_support_probability": torch.sigmoid(support_logit),
    }
    result = p1d_bev_loss(
        prediction,
        complete,
        visible,
        support,
        latest_observed_free_target=latest_observed,
        latest_support_target=latest_support,
        gt_valid_mask=torch.ones_like(support),
        probability_model="evidential",
        base_weights=P1BLossWeights(),
        additional_weights=P1DAdditionalLossWeights(
            guessed_hard_minimum=4,
            guessed_hard_maximum_per_group=16,
        ),
        wrong_evidence_scale=1.0,
        hidden_occupied_scale=1.0,
    )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert raw_evidence.grad is not None
    assert gate_logit.grad is not None
    assert support_logit.grad is not None
    assert result["history_support_fraction"] > 0
