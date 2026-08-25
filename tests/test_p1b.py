from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from vggt_bev_method1.cli_freeze_split import load_split_config
from vggt_bev_method1.cli_train_p1b import (
    _resize_square_query_content,
    _set_stage,
    checkpoint_contract,
    initialize_frozen_single_baseline,
)
from vggt_bev_method1.config import LabelValues
from vggt_bev_method1.data.p1b_targets import (
    ROUTING_GUESSED_FREE,
    ROUTING_GUESSED_OCCUPIED,
    ROUTING_OBSERVED_FREE,
    p1b_region_masks,
)
from vggt_bev_method1.data.void_coverage import FINAL_GT_VOID_FILTER
from vggt_bev_method1.models.p1b import P1BHead, P1BSystem
from vggt_bev_method1.models.p1b_probability import (
    decode_binary_prediction,
    fuse_pixel_routing,
)
from vggt_bev_method1.p1b_config import load_p1b_config, validate_p1b_config
from vggt_bev_method1.p1b_losses import (
    P1BLossWeights,
    _gaussian_boundary_weight,
    hidden_occupied_supervision_weight,
    p1b_bev_loss,
    p1b_fov_support_loss,
    p1b_routing_geometry_loss,
    wrong_evidence_kl_weight,
)
from vggt_bev_method1.p1b_metrics import (
    finalize_p1b_metrics,
    p1b_metric_totals,
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


def test_checkpoint_contract_accepts_strict_v4_void_index(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "algorithm": "strict-solid-voxel-or-navmesh-coverage-v4",
                "content_sha256": "strict-v4-index",
            }
        ),
        encoding="utf-8",
    )
    contract = checkpoint_contract(
        {
            "data": {"void_coverage_index": str(index_path)},
            "model": {"probability_model": "evidential"},
            "training": {"pipeline": "P1B-NLL"},
        },
        "manifest",
    )
    assert contract["loss_contract"] == "scene-geometry-only-gt-validity-v7"
    assert contract["gt_void_filter"] == FINAL_GT_VOID_FILTER
    assert contract["scene_coverage_algorithm"].endswith("-v4")
    assert contract["void_coverage_index_sha256"] == "strict-v4-index"


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
    masks = p1b_region_masks(complete, visible, support)
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
        result = p1b_bev_loss(
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
        result = p1b_bev_loss(
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
    result = p1b_bev_loss(
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
    result = p1b_bev_loss(
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
    weights = P1BLossWeights(
        observed_gate_pixel=0.0,
        guessed_pixel=0.0,
        guessed_surface=1.0,
        wrong_evidence_kl=0.0,
        support_bce=0.0,
        support_dice=0.0,
    )
    result = p1b_bev_loss(
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
    surface = p1b_region_masks(complete, visible, support).visible_surface
    assert bool((leaves[0].grad[0, 0][surface[0]].abs() > 0).all())
    assert leaves[1].grad is None or not bool((leaves[1].grad != 0).any())


def test_surface_partition_does_not_change_legacy_gate_objective() -> None:
    complete, visible, support = _targets()
    prediction, leaves = _prediction(complete.shape[-1], "bce")
    weights = P1BLossWeights(
        observed_gate_pixel=1.0,
        guessed_pixel=0.0,
        guessed_surface=0.0,
        wrong_evidence_kl=0.0,
        support_bce=0.0,
        support_dice=0.0,
    )
    result = p1b_bev_loss(
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


def test_smooth_boundary_weighting_reaches_only_gate_and_support() -> None:
    complete, visible, support = _targets()
    prediction, leaves = _prediction(complete.shape[-1], "bce")
    weights = P1BLossWeights(
        observed_gate_pixel=0.8,
        observed_gate_region=0.2,
        observed_gate_boundary_emphasis=3.0,
        guessed_pixel=0.0,
        guessed_surface=0.0,
        wrong_evidence_kl=0.0,
        support_bce=0.75,
        support_dice=0.25,
        support_boundary_emphasis=3.0,
        boundary_sigma=2.0,
    )
    result = p1b_bev_loss(
        prediction,
        complete,
        visible,
        support,
        probability_model="bce",
        weights=weights,
    )
    assert result["observed_gate_boundary_weight_max"] > 1.0
    assert result["observed_gate_boundary_weight_mean"] > 1.0
    assert result["support_boundary_weight_max"] > 1.0
    assert result["support_boundary_weight_mean"] > 1.0
    result["loss"].backward()
    assert leaves[0].grad is None or not bool((leaves[0].grad != 0).any())
    assert bool((leaves[1].grad != 0).any())
    assert bool((leaves[2].grad != 0).any())


def test_gaussian_boundary_weight_is_nonlinear_and_continuous() -> None:
    truth = torch.zeros((1, 41, 41), dtype=torch.bool)
    truth[:, :, :21] = True
    domain = torch.ones_like(truth)
    weight = _gaussian_boundary_weight(
        truth,
        domain,
        emphasis=5.0,
        sigma=3.0,
    )[0, 20]

    # The GT transition lies between columns 20 and 21. Weight is maximal
    # beside it and decays smoothly/nonlinearly toward the region interior.
    left = weight[:21].flip(0)
    assert weight.max() > 5.9
    assert bool((left[:-1] >= left[1:]).all())
    assert left[1] > 1.0
    assert left[-1] == pytest.approx(1.0, abs=1e-5)
    first_drop = float(left[0] - left[1])
    distant_drop = float(left[5] - left[6])
    assert first_drop != pytest.approx(distant_drop)


def test_scene_gt_validity_hard_ignores_every_bev_loss() -> None:
    complete, visible, support = _targets()
    masks = p1b_region_masks(complete, visible, support)
    valid = torch.ones_like(support)
    void_cells = torch.zeros_like(support)
    regions = (
        masks.observed_free,
        masks.visible_surface,
        masks.guessed_free,
        masks.hidden_guessed_occupied,
        ~support,
    )
    for region in regions:
        coordinate = region[0].nonzero()[0]
        void_cells[0, coordinate[0], coordinate[1]] = True
    valid[void_cells] = False
    prediction, leaves = _prediction(complete.shape[-1], "bce")
    result = p1b_bev_loss(
        prediction,
        complete,
        visible,
        support,
        gt_valid_mask=valid,
        probability_model="bce",
    )
    result["loss"].backward()
    guessed_gradient = leaves[0].grad[0, 0]
    gate_gradient = leaves[1].grad[0]
    support_gradient = leaves[2].grad[0]
    assert bool((guessed_gradient[void_cells[0]] == 0).all())
    assert bool((gate_gradient[void_cells[0]] == 0).all())
    assert bool((support_gradient[void_cells[0]] == 0).all())
    assert result["guessed_surface_loss"].item() == 0.0
    assert bool((guessed_gradient[valid[0] & support[0]].abs() > 0).any())
    assert bool((gate_gradient[valid[0] & support[0]].abs() > 0).any())
    assert bool((support_gradient[valid[0]].abs() > 0).any())
    assert result["gt_void_inside_support_fraction"] > 0


def test_hidden_curriculum_blocks_only_hidden_occupied_expert_gradient() -> None:
    complete, visible, support = _targets()
    masks = p1b_region_masks(complete, visible, support)
    weights = P1BLossWeights(
        observed_gate_pixel=0.0,
        guessed_pixel=1.0,
        guessed_surface=0.0,
        wrong_evidence_kl=0.0,
        support_bce=0.0,
        support_dice=0.0,
    )
    prediction, leaves = _prediction(complete.shape[-1], "bce")
    result = p1b_bev_loss(
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
    result = p1b_bev_loss(
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
    head = P1BHead(
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
        for parameter in head.single_guessed_token_projector.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in head.single_bev_decoder.routing.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in head.single_routing_token_projector.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in head.merged_guessed_token_projector.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in head.merged_routing_token_projector.parameters()
    )


def _tiny_system(probability_model: str = "bce") -> P1BSystem:
    return P1BSystem(
        torch.nn.Identity(),
        probability_model=probability_model,
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


def _legacy_shared_projector_state(head: P1BHead) -> dict[str, torch.Tensor]:
    current = head.state_dict()
    legacy: dict[str, torch.Tensor] = {}
    for key, value in current.items():
        if key.startswith("merged_guessed_token_projector.") or key.startswith(
            "merged_routing_token_projector."
        ):
            continue
        if key.startswith("single_guessed_token_projector."):
            key = "guessed_token_projector." + key.removeprefix(
                "single_guessed_token_projector."
            )
        elif key.startswith("single_routing_token_projector."):
            key = "routing_token_projector." + key.removeprefix(
                "single_routing_token_projector."
            )
        legacy[key] = value.clone()
    return legacy


def test_legacy_shared_projectors_load_into_both_branches() -> None:
    source = _tiny_system().unwrapped_head()
    target = _tiny_system().unwrapped_head()
    legacy = _legacy_shared_projector_state(source)
    target.load_state_dict(legacy, strict=True)
    loaded = target.state_dict()
    for legacy_prefix, branch_prefixes in (
        (
            "guessed_token_projector.",
            (
                "single_guessed_token_projector.",
                "merged_guessed_token_projector.",
            ),
        ),
        (
            "routing_token_projector.",
            (
                "single_routing_token_projector.",
                "merged_routing_token_projector.",
            ),
        ),
    ):
        for key, value in legacy.items():
            if not key.startswith(legacy_prefix):
                continue
            suffix = key.removeprefix(legacy_prefix)
            for branch_prefix in branch_prefixes:
                assert torch.equal(loaded[branch_prefix + suffix], value)


def test_merged_step_cannot_change_frozen_single_or_scale() -> None:
    system = _tiny_system()
    _set_stage(system, "bev_only", ("merged",))
    head = system.unwrapped_head()
    frozen_before = {
        key: value.detach().clone()
        for key, value in head.state_dict().items()
        if key.startswith(("single_", "scale_"))
    }
    trainable = [parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.1)
    extraction = {
        "tokens": {0: torch.randn(1, 2, 4, 8)},
        "patch_grid": (2, 2),
    }
    prediction = head(
        extraction,
        enabled_bev_branches=("merged",),
        include_scale=False,
    )["merged_bev"]
    loss = (
        prediction["guessed"]["occupancy_probability"].mean()
        + prediction["observed_gate_logit"].mean()
    )
    loss.backward()
    optimizer.step()
    assert any(
        parameter.grad is not None
        for parameter in head.merged_guessed_token_projector.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in head.merged_routing_token_projector.parameters()
    )
    for key, value in frozen_before.items():
        assert torch.equal(head.state_dict()[key], value), key


@pytest.mark.parametrize(
    ("variant", "arguments"),
    (
        ("balanced_bce_dice", {"bce_weight": 0.65, "dice_weight": 0.35}),
        (
            "balanced_bce_dice_boundary",
            {
                "bce_weight": 0.55,
                "dice_weight": 0.30,
                "boundary_weight": 0.15,
            },
        ),
        (
            "boundary_tversky",
            {
                "bce_weight": 0.55,
                "dice_weight": 0.0,
                "tversky_weight": 0.25,
                "boundary_weight": 0.20,
            },
        ),
    ),
)
def test_fov_support_only_losses_are_finite_and_differentiable(
    variant: str,
    arguments: dict[str, float],
) -> None:
    logit = torch.zeros((2, 16, 16), requires_grad=True)
    target = torch.zeros_like(logit, dtype=torch.bool)
    target[:, :10, 4:12] = True
    result = p1b_fov_support_loss(
        {
            "fov_support_logit": logit,
            "fov_support_probability": torch.sigmoid(logit),
        },
        target,
        variant=variant,
        **arguments,
    )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert logit.grad is not None
    assert bool((logit.grad.abs() > 0).any())


def test_fov_support_only_stage_trains_only_merged_routing_path() -> None:
    system = _tiny_system()
    _set_stage(system, "bev_only", ("merged",), "fov_support_only")
    head = system.unwrapped_head()
    trainable = [
        name for name, parameter in head.named_parameters() if parameter.requires_grad
    ]
    assert trainable
    assert all(
        name.startswith(
            ("merged_routing_token_projector.", "merged_bev_decoder.routing.")
        )
        for name in trainable
    )
    extraction = {
        "tokens": {0: torch.randn(1, 2, 4, 8)},
        "patch_grid": (2, 2),
    }
    output = head(
        extraction,
        enabled_bev_branches=("merged",),
        include_scale=False,
        bev_objective="fov_support_only",
    )["merged_bev"]
    assert set(output) == {
        "observed_gate_logit",
        "observed_gate_probability",
        "fov_support_logit",
        "fov_support_probability",
    }


def test_native_query_warmstart_resizes_content_without_changing_channels() -> None:
    source = torch.arange(4 * 4 * 3, dtype=torch.float32).reshape(16, 3)
    target = torch.empty(8 * 8, 3)
    resized = _resize_square_query_content(source, target)
    assert resized.shape == target.shape
    assert torch.isfinite(resized).all()


def test_combined_routing_geometry_trains_gate_and_support_only() -> None:
    complete, visible, support = _targets()
    gate_logit = torch.zeros_like(support, dtype=torch.float32, requires_grad=True)
    support_logit = torch.zeros_like(gate_logit, requires_grad=True)
    result = p1b_routing_geometry_loss(
        {
            "observed_gate_logit": gate_logit,
            "fov_support_logit": support_logit,
            "fov_support_probability": torch.sigmoid(support_logit),
        },
        complete,
        visible,
        support,
        variant="balanced_bce_dice",
        bce_weight=0.65,
        dice_weight=0.35,
    )
    result["loss"].backward()
    assert gate_logit.grad is not None
    assert support_logit.grad is not None
    assert bool((gate_logit.grad.abs() > 0).any())
    assert bool((support_logit.grad.abs() > 0).any())


def test_baseline_initialization_preserves_single_and_seeds_merged_projectors(
    tmp_path: Path,
) -> None:
    source = _tiny_system()
    target = _tiny_system()
    baseline_path = tmp_path / "baseline.pt"
    torch.save(
        {
            "probability_model": "bce",
            "epoch": 10,
            "global_step": 14320,
            "head": _legacy_shared_projector_state(source.unwrapped_head()),
        },
        baseline_path,
    )
    merged_decoder_before = {
        key: value.clone()
        for key, value in target.unwrapped_head().merged_bev_decoder.state_dict().items()
    }
    metadata = initialize_frozen_single_baseline(target, baseline_path)
    source_state = source.unwrapped_head().state_dict()
    target_state = target.unwrapped_head().state_dict()
    for prefix in (
        "single_guessed_token_projector.",
        "single_routing_token_projector.",
        "single_bev_decoder.",
        "scale_token_projector.",
        "scale_decoder.",
    ):
        for key, value in source_state.items():
            if key.startswith(prefix):
                assert torch.equal(target_state[key], value), key
    for single_prefix, merged_prefix in (
        ("single_guessed_token_projector.", "merged_guessed_token_projector."),
        ("single_routing_token_projector.", "merged_routing_token_projector."),
    ):
        for key, value in source_state.items():
            if key.startswith(single_prefix):
                merged_key = merged_prefix + key.removeprefix(single_prefix)
                assert torch.equal(target_state[merged_key], value), merged_key
    for key, value in merged_decoder_before.items():
        assert torch.equal(target.unwrapped_head().merged_bev_decoder.state_dict()[key], value)
    assert metadata["source_global_step"] == 14320


def test_fresh_routing_initialization_does_not_copy_single_projector(
    tmp_path: Path,
) -> None:
    source = _tiny_system()
    target = _tiny_system()
    baseline_path = tmp_path / "baseline.pt"
    torch.save(
        {
            "probability_model": "bce",
            "epoch": 10,
            "global_step": 14320,
            "head": _legacy_shared_projector_state(source.unwrapped_head()),
        },
        baseline_path,
    )
    before = {
        key: value.clone()
        for key, value in target.unwrapped_head().state_dict().items()
        if key.startswith("merged_routing_token_projector.")
    }
    metadata = initialize_frozen_single_baseline(
        target,
        baseline_path,
        seed_merged_routing_from_single=False,
    )
    after = target.unwrapped_head().state_dict()
    assert before
    for key, value in before.items():
        assert torch.equal(after[key], value), key
    assert metadata["seeded_merged_routing_from_single"] is False



def test_metric_finalization_uses_pixel_counts() -> None:
    metrics = finalize_p1b_metrics(
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
    totals = p1b_metric_totals(prediction, complete, visible, support)
    assert int(totals["observed_gate_tp"]) == 0
    assert int(totals["observed_gate_fn"]) == 3
    assert int(totals["routing_free_tp"]) == 3


def test_nll_and_bce_configs_have_incompatible_contracts() -> None:
    nll = load_p1b_config("configs/p1b_nll_local_smoke.toml")
    bce = load_p1b_config("configs/p1b_bce_local_smoke.toml")
    assert nll["model"]["probability_model"] == "evidential"
    assert bce["model"]["probability_model"] == "bce"
    assert nll["training"]["pipeline"] != bce["training"]["pipeline"]


def test_config_rejects_model_and_target_grid_mismatch() -> None:
    config = load_p1b_config("configs/p1b_nll_local_smoke.toml")
    mismatch = copy.deepcopy(config)
    mismatch["model"]["single_bev_output_size"] = 256
    with pytest.raises(ValueError, match="single_bev_output_size"):
        validate_p1b_config(mismatch)


def test_config_rejects_removed_ray_losses() -> None:
    config = load_p1b_config("configs/p1b_nll_local_smoke.toml")
    mismatch = copy.deepcopy(config)
    mismatch["training"]["ray_sequence_weight"] = 1.0
    with pytest.raises(ValueError, match="forbids legacy loss fields"):
        validate_p1b_config(mismatch)


def test_config_rejects_removed_surface_gate() -> None:
    config = load_p1b_config("configs/p1b_nll_local_smoke.toml")
    mismatch = copy.deepcopy(config)
    mismatch["training"]["surface_gate_pixel_weight"] = 1.0
    with pytest.raises(ValueError, match="forbids legacy loss fields"):
        validate_p1b_config(mismatch)


def test_config_requires_strict_void_index_when_enabled() -> None:
    config = load_p1b_config("configs/p1b_nll_local_smoke.toml")
    mismatch = copy.deepcopy(config)
    mismatch["data"]["require_gt_void_mask"] = True
    with pytest.raises(ValueError, match="void_coverage_index is required"):
        validate_p1b_config(mismatch)


def test_frozen_single_config_requires_merged_only_bev_training() -> None:
    config = load_p1b_config("configs/p1b_nll_local_smoke.toml")
    config["training"].update(
        {
            "freeze_single_to_baseline": True,
            "single_baseline_checkpoint": "/tmp/baseline.pt",
            "stage": "bev_only",
            "enabled_bev_branches": ["merged"],
            "single_task_weight": 0.0,
            "merged_task_weight": 1.0,
        }
    )
    validate_p1b_config(config)
    invalid = copy.deepcopy(config)
    invalid["training"]["enabled_bev_branches"] = ["single", "merged"]
    with pytest.raises(ValueError, match="enable only the merged"):
        validate_p1b_config(invalid)


def test_routing_geometry_configs_are_fail_closed() -> None:
    candidates = (Path("configs/p1b_merged6p5_role_contour_2epoch.toml"),)
    available = [path for path in candidates if path.is_file()]
    assert available
    for path in available:
        config = load_p1b_config(path)
        assert (
            config["training"]["bev_objective"]
            == "fov_support_and_observed_gate"
        )
        assert config["training"]["enabled_bev_branches"] == ["merged"]
        invalid = copy.deepcopy(config)
        invalid["training"]["guessed_pixel_weight"] = 1.0
        with pytest.raises(ValueError, match="non-support loss weights"):
            validate_p1b_config(invalid)


def test_config_rejects_shared_projector_layout() -> None:
    config = load_p1b_config("configs/p1b_nll_local_smoke.toml")
    config["model"]["projector_layout"] = "shared"
    with pytest.raises(ValueError, match="projector_layout"):
        validate_p1b_config(config)


def test_split_freezer_accepts_p1b_contract() -> None:
    config = load_split_config("configs/p1b_nll_local_smoke.toml")
    assert config["data"]["coordinate_mode"] == "p1b_fixed_metric"
    assert config["training"]["pipeline"] == "P1B-NLL"
