from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from vggt_bev_method1.cli_train_m05_plus import build_checkpoint_contract
from vggt_bev_method1.m05_plus_config import (
    CHECKPOINT_SCHEMA,
    PIPELINE_ID,
    load_m05_plus_config,
    validate_m05_plus_config,
)
from vggt_bev_method1.models import (
    DPTLiteLocalGlobalPyramid,
    M05PlusSystem,
    PerCellPrefixReader,
    PerQueryTemporalAttention,
    RoleSeparatedPrefixTrunk,
    TemporalFrameProposal,
)


class _Adapter(nn.Module):
    def aggregate(self, images: torch.Tensor) -> dict:
        raise NotImplementedError


class _RecordingAdapter(nn.Module):
    supports_native_token_dtype = False

    def __init__(self) -> None:
        super().__init__()
        self.received: torch.Tensor | None = None

    def aggregate(self, images: torch.Tensor) -> dict:
        self.received = images.clone()
        frames = images.shape[1]
        return {
            "tokens": {1: torch.randn(1, frames, 4, 16)},
            "camera_register_tokens": torch.randn(1, frames, 3, 16),
            "patch_grid": (2, 2),
            "_images": images,
        }

    def decode_scale_teacher(self, extraction: dict) -> dict:
        frames = extraction["_images"].shape[1]
        order = torch.arange(frames, dtype=torch.float32)[None, :, None, None]
        return {
            "estimated_depth_vggt": order,
            "estimated_depth_confidence": order + 10.0,
        }


def _model() -> M05PlusSystem:
    return M05PlusSystem(
        _Adapter(),
        cached_layers=(1,),
        spatial_scales=(1.0,),
        vggt_token_dim=16,
        hidden_dim=8,
        query_content_dim=4,
        heads=2,
        latest_decoder_layers=1,
        temporal_proposal_layers=1,
        scale_hidden_dim=8,
        scale_decoder_layers=1,
        self_attention_mode="linear",
        deformable_samples=2,
        cross_query_chunk_size=16,
        merged_bev_size=8,
        merged_extent_m=10.0,
        query_fourier_bands=2,
        shared_refinement_layers=1,
        routing_refinement_layers=1,
        evidence_refinement_layers=1,
        predict_scale_uncertainty=True,
        patch_stream_dim=4,
        prefix_context_hidden_dim=8,
        prefix_context_heads=2,
        prefix_context_layers=1,
        maximum_history=4,
        maximum_prefix_tokens=3,
        frame_reliability_hidden_dim=8,
        frame_reliability_minimum=0.25,
        frame_reliability_maximum=1.75,
        temporal_null_initial_probability=0.90,
    )


def _extraction(frames: int) -> dict:
    return {
        "tokens": {1: torch.randn(2, frames, 4, 16)},
        "camera_register_tokens": torch.randn(2, frames, 3, 16),
        "patch_grid": (2, 2),
    }


def test_m05_plus_has_an_independent_checkpoint_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_m05_plus_config(
        root / "configs/m05_plus_temporal_attention_10m_template.toml"
    )
    assert config["model"]["pipeline_variant"] == PIPELINE_ID
    assert CHECKPOINT_SCHEMA.endswith("-v3")
    assert config["model"]["hidden_dim"] == 96
    assert config["model"]["query_content_dim"] == 64
    assert config["model"]["latest_decoder_layers"] == 3
    assert config["model"]["temporal_proposal_layers"] == 2
    assert config["model"]["patch_fusion"] == "dpt_lite_top_down"
    assert config["model"]["patch_stream_dim"] == 48
    assert config["model"]["prefix_context_hidden_dim"] == 1024
    assert config["model"]["prefix_context_heads"] == 16
    assert config["model"]["prefix_context_layers"] == 8
    assert not config["model"]["explicit_geometry_module"]
    assert not config["model"]["geometry_auxiliary_loss"]
    assert config["model"]["merged_bev_output_size"] == 512
    assert config["model"]["merged_bev_extent_m"] == 10.0
    assert config["model"]["dataset_input_order"] == "oldest_to_latest"
    assert config["model"]["vggt_input_order"] == "latest_to_oldest"
    assert config["training"]["latest_auxiliary_loss_mode"] == (
        "merged_primary_additive"
    )
    assert config["training"]["latest_auxiliary_multiplier"] == 0.10
    assert _model().head.pipeline_id == PIPELINE_ID


def test_m05_plus_uses_per_query_frame_attention_and_null() -> None:
    torch.manual_seed(23)
    model = _model().eval()
    one = model.forward_head(_extraction(1))
    assert one["temporal_frame_attention_mean"].shape == (2, 0)
    torch.testing.assert_close(
        one["temporal_null_attention_mean"], torch.ones(2)
    )
    four = model.forward_head(_extraction(4))
    assert four["history_frame_indices_newest_to_oldest"] == (1, 2, 3)
    assert four["history_original_indices_newest_to_oldest"] == (2, 1, 0)
    assert four["temporal_frame_attention_mean"].shape == (2, 3)
    assert four["temporal_null_attention_mean"].shape == (2,)
    assert four["merged_bev"]["occupancy_probability"].shape == (2, 8, 8)
    assert four["geometry_conditioning"] == "none"
    assert four["patch_fusion"] == "dpt_lite_top_down"
    assert four["patch_token_streams"] == (
        "local_global_separate_until_fpn_output"
    )
    assert four["prefix_conditioning"] == (
        "per_cell_role_separated_token_attention"
    )
    assert not four["explicit_geometry_module_present"]
    assert not four["geometry_auxiliary_loss_present"]
    assert four["runtime_inputs"] == ("rgb_window",)
    assert four["merged_extent_m"] == 10.0
    assert four["temporal_execution"] == (
        "one_window_latest_anchor_parallel_history_proposals_per_query_softmax"
    )
    assert four["strict_zero_history_residual"]


def test_m05_plus_reverses_only_at_vggt_boundary() -> None:
    adapter = _RecordingAdapter()
    model = _model()
    model.adapter = adapter
    chronological = torch.arange(4, dtype=torch.float32)[None, :, None, None, None]
    extraction = model.extract(chronological)
    assert adapter.received is not None
    torch.testing.assert_close(
        adapter.received[:, :, 0, 0, 0],
        torch.tensor([[3.0, 2.0, 1.0, 0.0]]),
    )
    teacher = model.decode_scale_teacher(extraction)
    torch.testing.assert_close(
        teacher["estimated_depth_vggt"][:, :, 0, 0],
        torch.tensor([[3.0, 2.0, 1.0, 0.0]]),
    )
    torch.testing.assert_close(
        teacher["estimated_depth_confidence"][:, :, 0, 0],
        torch.tensor([[13.0, 12.0, 11.0, 10.0]]),
    )


def test_temporal_softmax_is_normalized_and_spatially_independent() -> None:
    module = PerQueryTemporalAttention(
        hidden_dim=8,
        maximum_history=4,
        null_initial_probability=0.90,
    ).eval()
    anchor = torch.randn(2, 7, 8)
    proposals = [torch.randn_like(anchor) for _ in range(3)]
    contexts = [torch.randn(2, 8) for _ in range(3)]
    reliabilities = [torch.ones(2) for _ in range(3)]
    merged, history_mean, null_mean = module(
        anchor, proposals, contexts, reliabilities, [1, 2, 3]
    )
    assert merged.shape == anchor.shape
    torch.testing.assert_close(
        history_mean.sum(dim=1) + null_mean,
        torch.ones(2),
        atol=1e-6,
        rtol=1e-6,
    )


def test_temporal_proposal_zero_update_is_exact_zero() -> None:
    module = TemporalFrameProposal(
        hidden_dim=8,
        heads=2,
        feature_levels=1,
        layers=1,
        deformable_samples=2,
        cross_query_chunk_size=16,
    ).eval()
    anchor = torch.randn(2, 4, 8)
    pyramid = [torch.randn(2, 1, 8, 2, 2)]
    reference_grid = torch.zeros(4, 2)
    context = torch.randn(2, 8)
    initial_residual = module(anchor, pyramid, reference_grid, context)
    assert torch.count_nonzero(initial_residual).item() == 0

    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    residual = module(
        anchor,
        [torch.zeros(2, 1, 8, 2, 2)],
        torch.zeros(4, 2),
        torch.zeros(2, 8),
    )
    assert torch.count_nonzero(residual).item() == 0


def test_initial_null_probability_is_history_count_invariant() -> None:
    module = PerQueryTemporalAttention(
        hidden_dim=8,
        maximum_history=10,
        null_initial_probability=0.90,
    ).eval()
    assert torch.count_nonzero(module.null_score[-1].weight).item() == 0
    anchor = torch.randn(2, 7, 8)
    null_probabilities = []
    for history_count in (1, 3, 9):
        zero_proposals = [torch.zeros_like(anchor) for _ in range(history_count)]
        merged, history_mean, null_mean = module(
            anchor,
            zero_proposals,
            [torch.randn(2, 8) for _ in range(history_count)],
            [torch.ones(2) for _ in range(history_count)],
            list(range(1, history_count + 1)),
        )
        assert torch.equal(merged, anchor)
        torch.testing.assert_close(
            null_mean,
            torch.full_like(null_mean, 0.90),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            history_mean.sum(dim=1) + null_mean,
            torch.ones_like(null_mean),
            atol=1e-6,
            rtol=1e-6,
        )
        null_probabilities.append(null_mean)
    torch.testing.assert_close(
        torch.stack(null_probabilities).amax(dim=0),
        torch.stack(null_probabilities).amin(dim=0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_m05_plus_rejects_prefix_and_patch_contract_mismatches() -> None:
    model = _model().eval()
    for prefix_count in (2, 4):
        wrong_prefix = _extraction(3)
        wrong_prefix["camera_register_tokens"] = torch.randn(
            2, 3, prefix_count, 16
        )
        try:
            model.forward_head(wrong_prefix)
        except ValueError as error:
            assert "requires exactly" in str(error)
        else:
            raise AssertionError(
                f"{prefix_count} prefix tokens must fail the exact contract"
            )

    wrong_frames = _extraction(3)
    wrong_frames["tokens"][1] = torch.randn(2, 2, 4, 16)
    try:
        model.forward_head(wrong_frames)
    except ValueError as error:
        assert "must have shape" in str(error)
    else:
        raise AssertionError("patch/prefix frame mismatch must fail")


def test_m05_plus_checkpoint_contract_matches_runtime_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_m05_plus_config(
        root / "configs/m05_plus_temporal_attention_10m_template.toml"
    )
    contract = build_checkpoint_contract(
        config,
        "manifest",
        "vggt",
        pipeline_id=PIPELINE_ID,
        checkpoint_schema=CHECKPOINT_SCHEMA,
    )
    runtime = _model().eval().forward_head(_extraction(3))
    for key in (
        "pipeline_id",
        "geometry_conditioning",
        "temporal_execution",
        "history_order",
        "dataset_input_order",
        "vggt_input_order",
        "strict_zero_history_residual",
        "temporal_null_history_count_invariant",
        "temporal_null_initial_probability",
        "patch_fusion",
        "patch_token_streams",
        "prefix_conditioning",
        "prefix_context_training",
        "prefix_token_pooling_present",
        "explicit_geometry_module_present",
        "geometry_auxiliary_loss_present",
    ):
        assert contract[key] == runtime[key]
    assert contract["expected_prefix_tokens"] == 17
    assert runtime["expected_prefix_tokens"] == 3
    assert contract["checkpoint_schema"] == CHECKPOINT_SCHEMA
    assert contract["patch_stream_dim"] == 48
    assert contract["patch_local_input_dim"] == 1024
    assert contract["patch_global_input_dim"] == 1024
    assert contract["prefix_context_hidden_dim"] == 1024
    assert contract["prefix_context_heads"] == 16
    assert contract["prefix_context_layers"] == 8
    assert runtime["patch_local_input_dim"] == 8
    assert runtime["patch_global_input_dim"] == 8
    assert runtime["patch_stream_dim"] == 4
    assert runtime["prefix_context_hidden_dim"] == 8


def test_scale_remains_opt_in_and_cannot_condition_bev() -> None:
    torch.manual_seed(29)
    model = _model().eval()
    extraction = _extraction(3)
    default = model.forward_head(extraction)
    requested = model.forward_head(extraction, include_scale=True)
    assert "scale" not in default
    assert "scale" in requested
    assert not default["scale_output_present"]
    assert requested["scale_output_present"]
    torch.testing.assert_close(
        default["merged_bev"]["occupancy_probability"],
        requested["merged_bev"]["occupancy_probability"],
    )


def test_m05_plus_has_separate_routing_and_evidence_refinement() -> None:
    model = _model()
    assert model.head.routing_refinement is not model.head.evidence_refinement
    assert model.head.routing_head is not model.head.evidence_head


def test_dpt_lite_fuses_depths_after_separate_local_global_projection() -> None:
    torch.manual_seed(31)
    projector = DPTLiteLocalGlobalPyramid(
        layers=(1, 2, 3),
        input_dim=16,
        hidden_dim=8,
        stream_dim=4,
        spatial_scales=(2.0, 1.0, 0.5),
    )
    tokens = {
        layer: torch.randn(2, 3, 16, 16, requires_grad=True)
        for layer in (1, 2, 3)
    }
    pyramid = projector(tokens, (4, 4))
    assert [feature.shape for feature in pyramid] == [
        (2, 3, 8, 8, 8),
        (2, 3, 8, 4, 4),
        (2, 3, 8, 2, 2),
    ]
    assert projector.local_projections["1"][1].in_features == 8
    assert projector.global_projections["1"][1].in_features == 8
    assert (
        projector.local_projections["1"][1]
        is not projector.global_projections["1"][1]
    )

    # The shallow/high-resolution output must receive gradients from the
    # deepest cached layer through both preserved streams.
    pyramid[0].square().mean().backward()
    deepest_gradient = tokens[3].grad
    assert deepest_gradient is not None
    assert deepest_gradient[..., :8].abs().sum().item() > 0.0
    assert deepest_gradient[..., 8:].abs().sum().item() > 0.0


def test_prefix_tokens_stay_role_separated_until_per_cell_attention() -> None:
    torch.manual_seed(37)
    trunk = RoleSeparatedPrefixTrunk(
        input_dim=16,
        hidden_dim=8,
        heads=2,
        layers=1,
        maximum_history=4,
        maximum_prefix_tokens=3,
    )
    prefix = torch.randn(2, 4, 3, 16, requires_grad=True)
    camera, registers = trunk(prefix, frame_reliability=torch.ones(2, 4))
    assert camera.shape == (2, 4, 8)
    assert registers.shape == (2, 4, 2, 8)
    assert trunk.camera_projection is not trunk.register_projection

    reader = PerCellPrefixReader(
        prefix_dim=8,
        hidden_dim=8,
        heads=2,
        query_chunk_size=3,
    )
    query = torch.randn(2, 7, 8)
    per_cell = reader(query, camera[:, 1], registers[:, 1])
    assert per_cell.shape == query.shape
    assert not torch.equal(per_cell[:, :1], per_cell[:, 1:2])
    per_cell.square().mean().backward()
    assert prefix.grad is not None
    assert prefix.grad[:, :, 0].abs().sum().item() > 0.0
    assert prefix.grad[:, :, 1:].abs().sum().item() > 0.0


def test_m05_plus_has_no_explicit_or_separately_supervised_geometry_head() -> None:
    model = _model()
    assert not hasattr(model.head, "implicit_geometry_trunk")
    assert all(
        "geometry" not in name for name, _ in model.head.named_modules()
    )
    runtime = model.eval().forward_head(_extraction(3))
    assert runtime["geometry_conditioning"] == "none"
    assert runtime["prefix_context_training"] == "joint_bev_only"
    assert not runtime["relative_pose_head_present"]
    assert not runtime["explicit_geometry_module_present"]
    assert not runtime["geometry_auxiliary_loss_present"]


def test_m05_plus_config_rejects_explicit_or_auxiliary_geometry() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_m05_plus_config(
        root / "configs/m05_plus_temporal_attention_10m_template.toml"
    )
    for key in ("explicit_geometry_module", "geometry_auxiliary_loss"):
        invalid = deepcopy(config)
        invalid["model"][key] = True
        try:
            validate_m05_plus_config(invalid)
        except ValueError as error:
            assert "geometry" in str(error)
        else:
            raise AssertionError(f"M05+ must reject {key}=true")

    retired = deepcopy(config)
    retired["model"]["implicit_geometry_hidden_dim"] = 1024
    try:
        validate_m05_plus_config(retired)
    except ValueError as error:
        assert "replaces the implicit geometry trunk" in str(error)
    else:
        raise AssertionError("M05+ must reject the retired geometry trunk")
