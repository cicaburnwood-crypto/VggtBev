from __future__ import annotations

import copy

import pytest
import torch

from vggt_bev_method1.config import LabelValues
from vggt_bev_method1.data.p2b_targets import (
    build_packed_ray_bank,
    p2b_region_masks,
)
from vggt_bev_method1.models.p2b import P2BHead
from vggt_bev_method1.models.p2b_probability import (
    decode_binary_prediction,
    fuse_experts,
)
from vggt_bev_method1.p2b_losses import P2BLossWeights, p2b_bev_loss
from vggt_bev_method1.p2b_metrics import finalize_p2b_metrics
from vggt_bev_method1.p2b_config import load_p2b_config, validate_p2b_config


def _targets(size: int = 16) -> tuple[torch.Tensor, ...]:
    labels = LabelValues()
    support = torch.zeros((1, size, size), dtype=torch.bool)
    support[:, : size // 2 + 1, size // 4 : 3 * size // 4] = True
    complete = torch.full((1, size, size), labels.unknown, dtype=torch.uint8)
    complete[support] = labels.free
    complete[:, size // 4, size // 2] = labels.occupied
    visible = torch.full_like(complete, labels.unknown)
    visible[:, size // 2, size // 2] = labels.free
    visible[:, size // 2 - 1, size // 2] = labels.free
    visible[:, size // 2 - 2, size // 2] = labels.free
    visible[:, size // 4, size // 2] = labels.occupied
    return complete, visible, support


def _prediction(size: int, probability_model: str) -> tuple[dict, list[torch.Tensor]]:
    channels = 2 if probability_model == "evidential" else 1
    observed_raw = torch.zeros((1, channels, size, size), requires_grad=True)
    guessed_raw = torch.zeros((1, channels, size, size), requires_grad=True)
    gate_logit = torch.zeros((1, size, size), requires_grad=True)
    support_logit = torch.zeros((1, size, size), requires_grad=True)
    observed = decode_binary_prediction(observed_raw, probability_model)
    guessed = decode_binary_prediction(guessed_raw, probability_model)
    gate_probability = torch.sigmoid(gate_logit)
    support_probability = torch.sigmoid(support_logit)
    fused = fuse_experts(
        observed,
        guessed,
        gate_probability,
        support_probability,
        probability_model,
    )
    return (
        {
            "observed": observed,
            "guessed": guessed,
            "gate_logit": gate_logit,
            "gate_probability": gate_probability,
            "fov_support_logit": support_logit,
            "fov_support_probability": support_probability,
            "fused": fused,
        },
        [observed_raw, guessed_raw, gate_logit, support_logit],
    )


def test_gt_routing_is_mutually_exclusive() -> None:
    complete, visible, support = _targets()
    masks = p2b_region_masks(complete, visible, support)
    assert not bool((masks.observed & masks.guessed).any())
    assert torch.equal(masks.observed | masks.guessed, masks.valid)
    assert masks.observed_surface.sum() == 1


def test_ray_bank_is_deterministic_and_ordered() -> None:
    _, _, support = _targets()
    first = build_packed_ray_bank(support[0])
    second = build_packed_ray_bank(support[0])
    assert torch.equal(first.indices, second.indices)
    assert torch.equal(first.valid, second.valid)
    assert first.indices.shape[0] > 0
    assert bool((first.valid.sum(dim=1) > 1).all())


def test_nll_and_bce_losses_backward_through_their_own_outputs() -> None:
    complete, visible, support = _targets()
    for probability_model in ("evidential", "bce"):
        prediction, leaves = _prediction(complete.shape[-1], probability_model)
        result = p2b_bev_loss(
            prediction,
            complete,
            visible,
            support,
            probability_model=probability_model,
            weights=P2BLossWeights(surface_continuity=0.0),
        )
        assert torch.isfinite(result["loss"])
        result["loss"].backward()
        assert all(value.grad is not None for value in leaves)
        if probability_model == "bce":
            assert result["wrong_evidence_kl"].item() == 0.0


def test_fusion_tracks_expert_disagreement() -> None:
    observed = decode_binary_prediction(
        torch.tensor([[[[8.0]], [[-8.0]]]]), "evidential"
    )
    guessed = decode_binary_prediction(
        torch.tensor([[[[-8.0]], [[8.0]]]]), "evidential"
    )
    fused = fuse_experts(
        observed,
        guessed,
        torch.full((1, 1, 1), 0.5),
        torch.ones((1, 1, 1)),
        "evidential",
    )
    assert abs(float(fused["occupancy_probability"]) - 0.5) < 1e-5
    assert float(fused["expert_disagreement"]) > 0.1
    assert float(fused["navigation_confidence"]) < 0.05


def test_expert_decoders_have_no_trainable_parameter_sharing() -> None:
    head = P2BHead(
        probability_model="bce",
        cached_layers=(0,),
        spatial_scales=(1.0,),
        vggt_token_dim=8,
        hidden_dim=8,
        heads=2,
        decoder_layers=1,
        scale_decoder_layers=1,
        self_attention_mode="linear",
        cross_attention_mode="linear",
        single_latent_bev_size=4,
        merged_latent_bev_size=4,
        single_output_size=8,
        merged_output_size=8,
        single_bev_extent_m=6.5,
        merged_bev_extent_m=10.0,
    )
    extraction = {
        "tokens": {0: torch.randn(1, 2, 4, 8)},
        "patch_grid": (2, 2),
    }
    output = head(
        extraction,
        enabled_bev_branches=("single",),
        include_scale=False,
    )
    output["single_bev"]["observed"]["occupancy_probability"].mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in head.single_bev_decoder.observed.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in head.single_bev_decoder.guessed.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in head.single_bev_decoder.routing.parameters()
    )


def test_metric_finalization_uses_raw_counts() -> None:
    metrics = finalize_p2b_metrics(
        {
            "surface_tp": 8,
            "surface_fp": 2,
            "surface_fn": 2,
            "surface_hit_distance_error_cells": 4,
            "surface_hit_distance_count": 8,
            "guessed_tp": 6,
            "guessed_fp": 2,
            "guessed_fn": 4,
            "guessed_tn": 10,
            "observed_free_fp": 5,
            "observed_free_count": 100,
            "gate_tp": 9,
            "gate_fp": 1,
            "gate_fn": 1,
            "support_tp": 10,
            "support_fp": 0,
            "support_fn": 0,
            "confidence_brier_sum": 2,
            "confidence_count": 10,
            "high_confidence_wrong": 1,
        }
    )
    assert metrics["surface_precision"] == 0.8
    assert metrics["surface_recall"] == 0.8
    assert metrics["observed_free_false_occupied_rate"] == 0.05


def test_nll_and_bce_configs_have_incompatible_contracts() -> None:
    nll = load_p2b_config("configs/p2b_nll_local_smoke.toml")
    bce = load_p2b_config("configs/p2b_bce_local_smoke.toml")
    assert nll["model"]["probability_model"] == "evidential"
    assert bce["model"]["probability_model"] == "bce"
    assert nll["training"]["pipeline"] != bce["training"]["pipeline"]


def test_config_rejects_model_and_target_grid_mismatch() -> None:
    config = load_p2b_config("configs/p2b_nll_local_smoke.toml")
    mismatch = copy.deepcopy(config)
    mismatch["model"]["single_bev_output_size"] = 256
    with pytest.raises(ValueError, match="single_bev_output_size"):
        validate_p2b_config(mismatch)
