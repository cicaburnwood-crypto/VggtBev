from __future__ import annotations

import copy

import pytest
import torch

from vggt_bev_method1.cli_freeze_split import load_split_config
from vggt_bev_method1.config import LabelValues
from vggt_bev_method1.data.p2b_targets import (
    ROUTING_GUESSED_FREE,
    ROUTING_GUESSED_OCCUPIED,
    ROUTING_OBSERVED_FREE,
    p2b_region_masks,
)
from vggt_bev_method1.models.p2b import P2BHead
from vggt_bev_method1.models.p2b_probability import (
    decode_binary_prediction,
    fuse_pixel_routing,
)
from vggt_bev_method1.p2b_config import load_p2b_config, validate_p2b_config
from vggt_bev_method1.p2b_losses import (
    P2BLossWeights,
    hidden_occupied_supervision_weight,
    p2b_bev_loss,
    wrong_evidence_kl_weight,
)
from vggt_bev_method1.p2b_metrics import (
    finalize_p2b_metrics,
    p2b_metric_totals,
)


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
    support_logit = torch.zeros((1, size, size), requires_grad=True)
    guessed = decode_binary_prediction(guessed_raw, probability_model)
    observed_gate_probability = torch.sigmoid(observed_gate_logit)
    guessed_region_probability = 1.0 - observed_gate_probability
    guessed_occupied_probability = (
        guessed_region_probability * guessed["occupancy_probability"]
    )
    guessed_free_probability = guessed_region_probability * (
        1.0 - guessed["occupancy_probability"]
    )
    routing_probability = torch.stack(
        (
            observed_gate_probability,
            guessed_free_probability,
            guessed_occupied_probability,
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
            "fov_support_logit": support_logit,
            "fov_support_probability": support_probability,
            "fused": fused,
        },
        [guessed_raw, observed_gate_logit, support_logit],
    )


def test_pixel_routing_targets_are_mutually_exclusive() -> None:
    complete, visible, support = _targets()
    masks = p2b_region_masks(complete, visible, support)
    assert not bool((masks.observed_free & masks.guessed).any())
    assert torch.equal(masks.observed_free | masks.guessed, masks.valid)
    assert torch.equal(
        masks.guessed_free | masks.guessed_occupied,
        masks.guessed,
    )
    assert masks.guessed_occupied.sum() == 2
    assert masks.visible_surface.sum() == 1
    assert masks.hidden_guessed_occupied.sum() == 1
    assert torch.equal(
        masks.guessed_occupied,
        masks.visible_surface | masks.hidden_guessed_occupied,
    )
    four_groups = torch.stack(
        (
            masks.observed_free,
            masks.visible_surface,
            masks.guessed_free,
            masks.hidden_guessed_occupied,
        ),
        dim=0,
    )
    assert not bool((four_groups.sum(dim=0) > 1).any())
    assert torch.equal(four_groups.any(dim=0), masks.valid)
    assert torch.equal(masks.observed_gate_target.bool(), masks.observed_free)
    assert torch.equal(
        masks.routing_target[masks.observed_free],
        torch.full_like(
            masks.routing_target[masks.observed_free], ROUTING_OBSERVED_FREE
        ),
    )
    assert torch.equal(
        masks.routing_target[masks.guessed_free],
        torch.full_like(
            masks.routing_target[masks.guessed_free], ROUTING_GUESSED_FREE
        ),
    )
    assert torch.equal(
        masks.routing_target[masks.guessed_occupied],
        torch.full_like(
            masks.routing_target[masks.guessed_occupied],
            ROUTING_GUESSED_OCCUPIED,
        ),
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
    assert torch.isfinite(result["observed_gate_pixel_bce"])
    assert all(value.grad is not None for value in leaves)


def test_guessed_loss_uses_explicit_per_sample_group_weights() -> None:
    complete, visible, support = _targets()
    prediction, leaves = _prediction(complete.shape[-1], "bce")
    # Each exact subset is reduced to one task mean before applying 35/40/25.
    leaves[0].data.fill_(4.0)
    result = p2b_bev_loss(
        prediction,
        complete,
        visible,
        support,
        probability_model="bce",
    )
    expected_free = torch.nn.functional.softplus(torch.tensor(4.0))
    expected_occupied = torch.nn.functional.softplus(torch.tensor(-4.0))
    assert torch.allclose(
        result["guessed_pixel_loss"],
        0.35 * expected_free + 0.40 * expected_occupied + 0.25 * expected_occupied,
    )
    assert torch.allclose(result["guessed_free_pixel_loss"], expected_free)
    assert torch.allclose(
        result["visible_surface_occupied_pixel_loss"], expected_occupied
    )
    assert torch.allclose(
        result["hidden_guessed_occupied_pixel_loss"], expected_occupied
    )


def test_missing_surface_and_occupied_groups_are_skipped() -> None:
    complete, visible, support = _targets()
    labels = LabelValues()
    complete = complete.clone()
    visible = visible.clone()
    complete[complete == labels.occupied] = labels.free
    visible[visible == labels.occupied] = labels.free
    prediction, leaves = _prediction(complete.shape[-1], "bce")
    leaves[0].data.fill_(2.0)
    result = p2b_bev_loss(
        prediction,
        complete,
        visible,
        support,
        probability_model="bce",
    )
    expected = torch.nn.functional.softplus(torch.tensor(2.0))
    assert torch.allclose(result["guessed_pixel_loss"], expected)
    assert result["visible_surface_occupied_pixel_loss"].item() == 0.0
    assert result["hidden_guessed_occupied_pixel_loss"].item() == 0.0
    assert result["guessed_surface_loss"].item() == 0.0


def test_surface_loss_reaches_only_guessed_expert() -> None:
    complete, visible, support = _targets()
    prediction, leaves = _prediction(complete.shape[-1], "bce")
    weights = P2BLossWeights(
        observed_gate_pixel=0.0,
        guessed_pixel=0.0,
        guessed_surface=1.0,
        wrong_evidence_kl=0.0,
        support_bce=0.0,
        support_dice=0.0,
    )
    result = p2b_bev_loss(
        prediction,
        complete,
        visible,
        support,
        probability_model="bce",
        weights=weights,
    )
    assert torch.allclose(
        result["guessed_surface_loss"], torch.log(torch.tensor(2.0))
    )
    result["loss"].backward()
    surface = p2b_region_masks(complete, visible, support).visible_surface
    assert bool((leaves[0].grad[0, 0][surface[0]].abs() > 0).all())
    assert leaves[1].grad is None or not bool((leaves[1].grad != 0).any())


def test_surface_partition_does_not_change_legacy_gate_objective() -> None:
    complete, visible, support = _targets()
    prediction, leaves = _prediction(complete.shape[-1], "bce")
    weights = P2BLossWeights(
        observed_gate_pixel=1.0,
        guessed_pixel=0.0,
        guessed_surface=0.0,
        wrong_evidence_kl=0.0,
        support_bce=0.0,
        support_dice=0.0,
    )
    result = p2b_bev_loss(
        prediction,
        complete,
        visible,
        support,
        probability_model="bce",
        weights=weights,
    )
    assert torch.allclose(
        result["observed_gate_pixel_bce"], torch.log(torch.tensor(2.0))
    )
    result["loss"].backward()
    assert leaves[0].grad is None or not bool((leaves[0].grad != 0).any())
    assert bool((leaves[1].grad.abs() > 0).any())


def test_hidden_curriculum_blocks_only_hidden_occupied_expert_gradient() -> None:
    complete, visible, support = _targets()
    masks = p2b_region_masks(complete, visible, support)
    weights = P2BLossWeights(
        observed_gate_pixel=0.0,
        guessed_pixel=1.0,
        guessed_surface=0.0,
        wrong_evidence_kl=0.0,
        support_bce=0.0,
        support_dice=0.0,
    )
    prediction, leaves = _prediction(complete.shape[-1], "bce")
    result = p2b_bev_loss(
        prediction,
        complete,
        visible,
        support,
        probability_model="bce",
        weights=weights,
        hidden_occupied_scale=0.0,
    )
    result["loss"].backward()
    gradient = leaves[0].grad[0, 0]
    assert bool((gradient[masks.guessed_free[0]].abs() > 0).all())
    assert bool((gradient[masks.visible_surface[0]].abs() > 0).all())
    assert bool((gradient[masks.hidden_guessed_occupied[0]] == 0).all())

    prediction, leaves = _prediction(complete.shape[-1], "bce")
    result = p2b_bev_loss(
        prediction,
        complete,
        visible,
        support,
        probability_model="bce",
        weights=weights,
        hidden_occupied_scale=1.0,
    )
    result["loss"].backward()
    assert bool(
        (leaves[0].grad[0, 0][masks.hidden_guessed_occupied[0]].abs() > 0).all()
    )


def test_curriculum_boundaries() -> None:
    assert hidden_occupied_supervision_weight(0, 100) == 0.0
    assert hidden_occupied_supervision_weight(10, 100) == 0.0
    assert hidden_occupied_supervision_weight(17, 100) == pytest.approx(7.0 / 15.0)
    assert hidden_occupied_supervision_weight(25, 100) == 1.0
    assert hidden_occupied_supervision_weight(100, 100) == 1.0
    assert wrong_evidence_kl_weight(20, 100, maximum=1.0) == 0.0
    assert wrong_evidence_kl_weight(25, 100, maximum=1.0) == pytest.approx(0.5)
    assert wrong_evidence_kl_weight(30, 100, maximum=1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        hidden_occupied_supervision_weight(-1, 100)


def test_fusion_obeys_deterministic_routes() -> None:
    guessed = decode_binary_prediction(
        torch.tensor([[[[8.0]], [[-8.0]]]]), "evidential"
    )
    support = torch.ones((1, 1, 1))
    expected = (
        ((1.0, 0.0, 0.0), 0.0),
        ((0.0, 1.0, 0.0), 0.0),
        ((0.0, 0.0, 1.0), 1.0),
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
        single_latent_bev_size=8,
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
    assert any(
        parameter.grad is not None
        for parameter in head.guessed_token_projector.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in head.single_bev_decoder.routing.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in head.routing_token_projector.parameters()
    )


def test_metric_finalization_uses_pixel_counts() -> None:
    metrics = finalize_p2b_metrics(
        {
            "routing_free_tp": 90,
            "routing_free_fp": 5,
            "routing_free_fn": 10,
            "routing_guessed_free_tp": 40,
            "routing_guessed_free_fp": 5,
            "routing_guessed_free_fn": 10,
            "routing_guessed_occupied_tp": 8,
            "routing_guessed_occupied_fp": 2,
            "routing_guessed_occupied_fn": 2,
            "routing_guessed_tp": 48,
            "routing_guessed_fp": 5,
            "routing_guessed_fn": 12,
            "observed_gate_tp": 90,
            "observed_gate_fp": 5,
            "observed_gate_fn": 10,
            "surface_gate_tp": 8,
            "surface_gate_fp": 2,
            "surface_gate_fn": 2,
            "fused_surface_tp": 7,
            "fused_surface_fp": 3,
            "fused_surface_fn": 3,
            "hidden_guessed_tp": 5,
            "hidden_guessed_fp": 5,
            "hidden_guessed_fn": 5,
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
    assert metrics["guessed_occupied_precision"] == 0.8
    assert metrics["guessed_occupied_recall"] == 0.8
    assert metrics["observed_gate_precision"] == 90 / 95
    assert metrics["observed_gate_recall"] == 0.9
    assert metrics["observed_free_false_occupied_rate"] == 0.05
    assert metrics["guessed_pixelwise_precision"] == 0.75
    assert metrics["visible_surface_precision"] == 0.7
    assert metrics["visible_surface_recall"] == 0.7
    assert metrics["hidden_occupied_precision"] == 0.5
    assert metrics["hidden_occupied_recall"] == 0.5


def test_observed_monitor_reads_gate_not_three_way_argmax() -> None:
    complete, visible, support = _targets()
    prediction, _ = _prediction(complete.shape[-1], "evidential")
    gate = torch.full_like(prediction["observed_gate_probability"], 0.4)
    prediction["observed_gate_probability"] = gate
    prediction["routing_probability"] = torch.stack(
        (gate, 0.3 * torch.ones_like(gate), 0.3 * torch.ones_like(gate)),
        dim=1,
    )
    totals = p2b_metric_totals(prediction, complete, visible, support)
    assert int(totals["observed_gate_tp"]) == 0
    assert int(totals["observed_gate_fn"]) == 3
    assert int(totals["routing_free_tp"]) == 3


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


def test_config_rejects_removed_surface_gate() -> None:
    config = load_p2b_config("configs/p2b_nll_local_smoke.toml")
    mismatch = copy.deepcopy(config)
    mismatch["training"]["surface_gate_pixel_weight"] = 1.0
    with pytest.raises(ValueError, match="forbids legacy loss fields"):
        validate_p2b_config(mismatch)


def test_split_freezer_accepts_p2b_contract() -> None:
    config = load_split_config("configs/p2b_nll_local_smoke.toml")
    assert config["data"]["coordinate_mode"] == "p2b_fixed_metric"
    assert config["training"]["pipeline"] == "P2B-NLL"
