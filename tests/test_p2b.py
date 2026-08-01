from __future__ import annotations

import copy

import pytest
import torch

from vggt_bev_method1.cli_freeze_split import load_split_config
from vggt_bev_method1.config import LabelValues
from vggt_bev_method1.data.p2b_targets import (
    ROUTING_GUESSED,
    ROUTING_OBSERVED_FREE,
    ROUTING_OBSERVED_SURFACE,
    p2b_region_masks,
)
from vggt_bev_method1.models.p2b import P2BHead
from vggt_bev_method1.models.p2b_probability import (
    decode_binary_prediction,
    fuse_pixel_routing,
)
from vggt_bev_method1.p2b_config import load_p2b_config, validate_p2b_config
from vggt_bev_method1.p2b_losses import P2BLossWeights, p2b_bev_loss
from vggt_bev_method1.p2b_metrics import finalize_p2b_metrics


def _targets(size: int = 16) -> tuple[torch.Tensor, ...]:
    labels = LabelValues()
    support = torch.zeros((1, size, size), dtype=torch.bool)
    support[:, : size // 2 + 1, size // 4 : 3 * size // 4] = True
    complete = torch.full((1, size, size), labels.unknown, dtype=torch.uint8)
    complete[support] = labels.free
    complete[:, size // 4, size // 2] = labels.occupied
    complete[:, 2, size // 2 + 2] = labels.occupied
    visible = torch.full_like(complete, labels.unknown)
    visible[:, size // 2, size // 2] = labels.free
    visible[:, size // 2 - 1, size // 2] = labels.free
    visible[:, size // 2 - 2, size // 2] = labels.free
    visible[:, size // 4, size // 2] = labels.occupied
    return complete, visible, support


def _prediction(size: int, probability_model: str) -> tuple[dict, list[torch.Tensor]]:
    channels = 2 if probability_model == "evidential" else 1
    guessed_raw = torch.zeros((1, channels, size, size), requires_grad=True)
    observed_gate_logit = torch.zeros((1, size, size), requires_grad=True)
    surface_gate_logit = torch.zeros((1, size, size), requires_grad=True)
    support_logit = torch.zeros((1, size, size), requires_grad=True)
    guessed = decode_binary_prediction(guessed_raw, probability_model)
    observed_gate_probability = torch.sigmoid(observed_gate_logit)
    surface_gate_probability = torch.sigmoid(surface_gate_logit)
    routing_probability = torch.stack(
        (
            observed_gate_probability * (1.0 - surface_gate_probability),
            observed_gate_probability * surface_gate_probability,
            1.0 - observed_gate_probability,
        ),
        dim=1,
    )
    support_probability = torch.sigmoid(support_logit)
    fused = fuse_pixel_routing(
        routing_probability,
        guessed,
        support_probability,
        probability_model,
    )
    return (
        {
            "guessed": guessed,
            "routing_probability": routing_probability,
            "observed_gate_logit": observed_gate_logit,
            "observed_gate_probability": observed_gate_probability,
            "surface_gate_logit": surface_gate_logit,
            "surface_gate_probability": surface_gate_probability,
            "fov_support_logit": support_logit,
            "fov_support_probability": support_probability,
            "fused": fused,
        },
        [guessed_raw, observed_gate_logit, surface_gate_logit, support_logit],
    )


def test_pixel_routing_targets_are_mutually_exclusive() -> None:
    complete, visible, support = _targets()
    masks = p2b_region_masks(complete, visible, support)
    assert not bool((masks.observed & masks.guessed).any())
    assert torch.equal(masks.observed | masks.guessed, masks.valid)
    assert masks.observed_surface.sum() == 1
    assert torch.equal(masks.observed_gate_target.bool(), masks.observed)
    assert torch.equal(masks.surface_gate_target.bool(), masks.observed_surface)
    assert torch.equal(
        masks.routing_target[masks.observed_free],
        torch.full_like(
            masks.routing_target[masks.observed_free], ROUTING_OBSERVED_FREE
        ),
    )
    assert torch.equal(
        masks.routing_target[masks.observed_surface],
        torch.full_like(
            masks.routing_target[masks.observed_surface], ROUTING_OBSERVED_SURFACE
        ),
    )
    assert torch.equal(
        masks.routing_target[masks.guessed],
        torch.full_like(masks.routing_target[masks.guessed], ROUTING_GUESSED),
    )


def test_nll_and_bce_pixel_losses_backward() -> None:
    complete, visible, support = _targets()
    for probability_model in ("evidential", "bce"):
        prediction, leaves = _prediction(complete.shape[-1], probability_model)
        result = p2b_bev_loss(
            prediction,
            complete,
            visible,
            support,
            probability_model=probability_model,
        )
        assert torch.isfinite(result["loss"])
        result["loss"].backward()
        assert all(value.grad is not None for value in leaves)
        if probability_model == "bce":
            assert result["wrong_evidence_kl"].item() == 0.0


def test_pixel_loss_is_amp_safe() -> None:
    complete, visible, support = _targets()
    prediction, leaves = _prediction(complete.shape[-1], "evidential")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = p2b_bev_loss(
            prediction,
            complete,
            visible,
            support,
            probability_model="evidential",
        )
    result["loss"].backward()
    assert torch.isfinite(result["surface_gate_pixel_bce"])
    assert all(value.grad is not None for value in leaves)


def test_fusion_obeys_deterministic_routes() -> None:
    guessed = decode_binary_prediction(
        torch.tensor([[[[8.0]], [[-8.0]]]]), "evidential"
    )
    support = torch.ones((1, 1, 1))
    expected = (
        ((1.0, 0.0, 0.0), 0.0),
        ((0.0, 1.0, 0.0), 1.0),
    )
    for route, occupancy in expected:
        routing = torch.tensor(route).reshape(1, 3, 1, 1)
        fused = fuse_pixel_routing(routing, guessed, support, "evidential")
        assert abs(float(fused["occupancy_probability"]) - occupancy) < 1e-5


def test_uncertain_routing_lowers_navigation_confidence() -> None:
    guessed = decode_binary_prediction(
        torch.tensor([[[[8.0]], [[-8.0]]]]), "evidential"
    )
    routing = torch.full((1, 3, 1, 1), 1.0 / 3.0)
    fused = fuse_pixel_routing(
        routing,
        guessed,
        torch.ones((1, 1, 1)),
        "evidential",
    )
    assert float(fused["routing_entropy"]) > 0.99
    assert float(fused["navigation_confidence"]) < 0.05


def test_guessed_and_routing_decoders_are_independent() -> None:
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
    assert not hasattr(head, "observed_token_projector")
    assert not hasattr(head.single_bev_decoder, "observed")
    extraction = {
        "tokens": {0: torch.randn(1, 2, 4, 8)},
        "patch_grid": (2, 2),
    }
    output = head(extraction, enabled_bev_branches=("single",), include_scale=False)
    output["single_bev"]["guessed"]["occupancy_probability"].mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in head.single_bev_decoder.guessed.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in head.single_bev_decoder.routing.parameters()
    )


def test_metric_finalization_uses_pixel_counts() -> None:
    metrics = finalize_p2b_metrics(
        {
            "surface_tp": 8,
            "surface_fp": 2,
            "surface_fn": 2,
            "routing_free_tp": 90,
            "routing_free_fp": 5,
            "routing_free_fn": 10,
            "routing_guessed_tp": 80,
            "routing_guessed_fp": 10,
            "routing_guessed_fn": 20,
            "guessed_tp": 6,
            "guessed_fp": 2,
            "guessed_fn": 4,
            "guessed_tn": 10,
            "observed_free_fp": 5,
            "observed_free_count": 100,
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
    assert metrics["guessed_pixelwise_precision"] == 0.75


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


def test_config_rejects_removed_ray_losses() -> None:
    config = load_p2b_config("configs/p2b_nll_local_smoke.toml")
    mismatch = copy.deepcopy(config)
    mismatch["training"]["ray_sequence_weight"] = 1.0
    with pytest.raises(ValueError, match="forbids legacy loss fields"):
        validate_p2b_config(mismatch)


def test_split_freezer_accepts_p2b_contract() -> None:
    config = load_split_config("configs/p2b_nll_local_smoke.toml")
    assert config["data"]["coordinate_mode"] == "p2b_fixed_metric"
    assert config["training"]["pipeline"] == "P2B-NLL"
