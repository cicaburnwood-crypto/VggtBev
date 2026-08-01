from __future__ import annotations

import torch
from torch import nn

from vggt_bev_method1.models.attention import (
    DeformableCrossAttention,
    MultiheadAttention,
)
from vggt_bev_method1.models.method1 import Method1System, PairedMethod1System
from vggt_bev_method1.models.vggt_adapter import (
    _scene_radius,
    predicted_geometry_summary,
    stabilize_intrinsics,
    world_camera_centers,
)


class FakeLiveAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_count = 0

    def forward(
        self,
        images: torch.Tensor,
        camera_height_m: torch.Tensor,
    ) -> dict:
        self.forward_count += 1
        batch, frames = images.shape[:2]
        tokens = {
            layer: torch.randn(batch, frames, 6, 16, device=images.device)
            for layer in (4, 11, 17, 23)
        }
        return {
            "tokens": tokens,
            "patch_grid": (2, 3),
            "geometry_cue": torch.randn(batch, 24, device=images.device),
            "scene_radius_vggt": torch.ones(batch, device=images.device),
            "p1a_geometry": {
                "single_extent_normalized_scale": torch.full(
                    (batch,), 6.5, device=images.device
                ),
                "merged_extent_normalized_scale": torch.full(
                    (batch,), 10.0, device=images.device
                ),
                "camera_height_m": camera_height_m,
                "geometry_quality": torch.ones(
                    batch,
                    device=images.device,
                ),
                "geometry_valid": torch.ones(
                    batch,
                    dtype=torch.bool,
                    device=images.device,
                ),
                "single_fov_support": torch.zeros(
                    batch,
                    4,
                    4,
                    dtype=torch.bool,
                    device=images.device,
                ),
                "merged_fov_support": torch.zeros(
                    batch,
                    4,
                    4,
                    dtype=torch.bool,
                    device=images.device,
                ),
            },
            "geometry_source": (
                "live VGGT-Omega depth/intrinsics/extrinsics in native scale"
            ),
        }


def test_linear_attention_connects_different_sequence_lengths() -> None:
    attention = MultiheadAttention(16, 4, mode="linear")
    output = attention(torch.randn(2, 9, 16), torch.randn(2, 13, 16))
    assert output.shape == (2, 9, 16)
    assert torch.isfinite(output).all()


def test_deformable_cross_attention_samples_every_spatial_level() -> None:
    attention = DeformableCrossAttention(
        16,
        4,
        levels=2,
        samples=2,
        query_chunk_size=3,
    )
    query = torch.randn(1, 5, 16, requires_grad=True)
    pyramid = [
        torch.randn(1, 2, 16, 4, 6, requires_grad=True),
        torch.randn(1, 2, 16, 2, 3, requires_grad=True),
    ]
    reference = torch.rand(5, 2) * 2 - 1
    output = attention(query, pyramid, reference)
    assert output.shape == (1, 5, 16)
    output.sum().backward()
    assert query.grad is not None
    assert all(level.grad is not None for level in pyramid)


def test_method1_system_produces_direct_single_and_merged_cells() -> None:
    system = Method1System(
        FakeLiveAdapter(),
        supervision="observed",
        spatial_scales=(1.0, 1.0, 1.0, 1.0),
        vggt_token_dim=16,
        hidden_dim=16,
        geometry_cue_dim=24,
        heads=4,
        decoder_layers=1,
        self_attention_mode="exact",
        cross_attention_mode="exact",
        deformable_samples=2,
        cross_query_chunk_size=16,
        single_output_size=8,
        merged_output_size=6,
        gradient_checkpointing=False,
    )
    output = system(
        torch.randn(1, 3, 3, 32, 48),
        torch.tensor([1.5]),
    )
    assert output["single"]["occupancy_logit"].shape == (1, 8, 8)
    assert output["single"]["observed_logit"].shape == (1, 8, 8)
    assert output["merged"]["occupancy_logit"].shape == (1, 6, 6)
    assert (
        output["coordinate_mode"]
        == "camera_height_anchored_fixed_normalized_scale"
    )
    assert output["single_output_size"] == 8
    assert output["merged_output_size"] == 6
    assert (
        output["single_extent_normalized_scale"] == 6.5
    ).all()
    assert (
        output["merged_extent_normalized_scale"] == 10.0
    ).all()
    assert output["camera_height_used"] is True
    assert output["runtime_aligned"] is True


def test_method1_system_backpropagates_through_deformable_pyramid() -> None:
    system = Method1System(
        FakeLiveAdapter(),
        supervision="complete",
        spatial_scales=(2.0, 1.0, 1.0, 0.5),
        vggt_token_dim=16,
        hidden_dim=16,
        geometry_cue_dim=24,
        heads=4,
        decoder_layers=1,
        self_attention_mode="linear",
        cross_attention_mode="deformable",
        deformable_samples=2,
        cross_query_chunk_size=8,
        single_output_size=4,
        merged_output_size=4,
        gradient_checkpointing=True,
    )
    output = system(
        torch.randn(1, 2, 3, 32, 48),
        torch.tensor([1.5]),
    )
    loss = (
        output["single"]["occupancy_logit"].mean()
        + output["merged"]["observed_logit"].mean()
    )
    loss.backward()
    assert system.token_projector.projections["4"][1].weight.grad is not None


def test_joint_system_outputs_both_tasks_from_shared_bev_queries() -> None:
    system = Method1System(
        FakeLiveAdapter(),
        supervision="joint",
        spatial_scales=(1.0, 1.0, 1.0, 1.0),
        vggt_token_dim=16,
        hidden_dim=16,
        geometry_cue_dim=24,
        heads=4,
        decoder_layers=1,
        self_attention_mode="exact",
        cross_attention_mode="exact",
        deformable_samples=2,
        cross_query_chunk_size=16,
        single_output_size=4,
        merged_output_size=4,
        gradient_checkpointing=False,
    )
    output = system(
        torch.randn(1, 2, 3, 32, 48),
        torch.tensor([1.5]),
    )
    for extent in ("single", "merged"):
        assert {"observed", "complete"} <= set(output[extent])
        assert output[extent]["extent_normalized_scale"].shape == (1,)
        for task in ("observed", "complete"):
            assert output[extent][task]["occupancy_logit"].shape == (1, 4, 4)
    loss = sum(
        output[extent][task][logit].mean()
        for extent in ("single", "merged")
        for task in ("observed", "complete")
        for logit in ("occupancy_logit", "observed_logit")
    )
    loss.backward()
    assert all(
        embedding.grad is not None
        for embedding in system.single_decoder.cell_embedding_chunks
    )
    assert all(
        embedding.grad is not None
        for embedding in system.merged_decoder.cell_embedding_chunks
    )
    for decoder in (system.single_decoder, system.merged_decoder):
        for task in ("observed", "complete"):
            assert decoder.occupancy_heads[task].weight.grad is not None
            assert decoder.observation_heads[task].weight.grad is not None


def test_paired_system_runs_vggt_once_and_has_independent_models() -> None:
    adapter = FakeLiveAdapter()
    system = PairedMethod1System(
        adapter,
        spatial_scales=(1.0, 1.0, 1.0, 1.0),
        vggt_token_dim=16,
        hidden_dim=16,
        geometry_cue_dim=24,
        heads=4,
        decoder_layers=1,
        self_attention_mode="exact",
        cross_attention_mode="exact",
        deformable_samples=2,
        cross_query_chunk_size=16,
        single_output_size=4,
        merged_output_size=4,
        gradient_checkpointing=False,
    )
    output = system(
        torch.randn(1, 2, 3, 32, 48),
        torch.tensor([1.5]),
    )
    assert "masked_observed_semantic" in output["observed"]["single"]
    assert "fov_complete_semantic" in output["complete"]["single"]
    assert torch.equal(
        output["complete"]["single"]["fov_complete_semantic"]
        == 112,
        ~output["complete"]["single"]["runtime_fov_support"],
    )
    assert adapter.forward_count == 1
    for extent in ("single", "merged"):
        assert output["observed"][extent]["class_logits"].shape == (1, 3, 4, 4)
        complete = output["complete"][extent]
        assert complete["alpha_occupied"].shape == (1, 4, 4)
        assert complete["beta_free"].shape == (1, 4, 4)
        assert (complete["alpha_occupied"] >= 1.0).all()
        assert (complete["beta_free"] >= 1.0).all()
        assert torch.allclose(
            complete["occupancy_probability"],
            complete["alpha_occupied"]
            / (complete["alpha_occupied"] + complete["beta_free"]),
        )
    observed_ids = {
        id(parameter)
        for parameter in system.unwrapped_observed_model().parameters()
    }
    complete_ids = {
        id(parameter)
        for parameter in system.unwrapped_complete_model().parameters()
    }
    assert observed_ids.isdisjoint(complete_ids)


def test_paired_model_losses_cannot_cross_between_parameter_sets() -> None:
    system = PairedMethod1System(
        FakeLiveAdapter(),
        spatial_scales=(1.0, 1.0, 1.0, 1.0),
        vggt_token_dim=16,
        hidden_dim=16,
        geometry_cue_dim=24,
        heads=4,
        decoder_layers=1,
        self_attention_mode="exact",
        cross_attention_mode="exact",
        deformable_samples=2,
        cross_query_chunk_size=16,
        single_output_size=4,
        merged_output_size=4,
        gradient_checkpointing=False,
    )
    extraction = system.extract(
        torch.randn(1, 2, 3, 32, 48),
        torch.tensor([1.5]),
    )
    observed = system.forward_observed(extraction)
    observed_loss = sum(
        observed[extent]["class_logits"].mean()
        for extent in ("single", "merged")
    )
    observed_loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in system.unwrapped_observed_model().parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in system.unwrapped_complete_model().parameters()
    )


def test_world_camera_centers_obey_pdf_equation_7() -> None:
    angle = torch.tensor(torch.pi / 2)
    rotation = torch.tensor(
        [
            [torch.cos(angle), torch.tensor(0.0), torch.sin(angle)],
            [torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0)],
            [-torch.sin(angle), torch.tensor(0.0), torch.cos(angle)],
        ]
    )
    center = torch.tensor([2.0, 0.5, -1.0])
    extrinsics = torch.zeros(1, 2, 3, 4)
    extrinsics[0, 0, :3, :3] = torch.eye(3)
    extrinsics[0, 1, :3, :3] = rotation
    extrinsics[0, :, :3, 3] = -extrinsics[0, :, :3, :3] @ center
    recovered = world_camera_centers(extrinsics)
    assert torch.allclose(recovered[0, 0], center, atol=1e-6)
    assert torch.allclose(recovered[0, 1], center, atol=1e-6)


def test_intrinsics_are_locked_to_one_robust_sequence_value() -> None:
    intrinsics = torch.eye(3).view(1, 1, 3, 3).expand(1, 3, -1, -1).clone()
    intrinsics[0, :, 0, 0] = torch.tensor([100.0, 300.0, 200.0])
    stable = stabilize_intrinsics(intrinsics)
    assert torch.equal(stable[0, :, 0, 0], torch.full((3,), 200.0))


def test_geometry_summary_uses_live_vggt_tensor_shapes() -> None:
    depth = torch.full((2, 3, 48, 64, 1), 2.0)
    confidence = torch.full((2, 3, 48, 64), 2.0)
    intrinsics = torch.eye(3).view(1, 1, 3, 3).expand(2, 3, -1, -1).clone()
    intrinsics[..., 0, 0] = 40.0
    intrinsics[..., 1, 1] = 36.0
    intrinsics[..., 0, 2] = 32.0
    intrinsics[..., 1, 2] = 24.0
    camera = torch.zeros(2, 3, 3, 4)
    camera[..., :3, :3] = torch.eye(3)
    camera[:, 1, 0, 3] = 0.5
    camera[:, 2, 0, 3] = 1.0
    summary = predicted_geometry_summary(
        depth,
        confidence,
        intrinsics,
        camera,
        image_height=48,
        image_width=64,
    )
    assert summary["cue"].shape == (2, 14)
    assert torch.isfinite(summary["cue"]).all()
    assert summary["scene_radius_vggt"].shape == (2,)

    scaled_camera = camera.clone()
    scaled_camera[..., :3, 3] *= 3.0
    scaled = predicted_geometry_summary(
        depth * 3.0,
        confidence,
        intrinsics,
        scaled_camera,
        image_height=48,
        image_width=64,
    )
    assert torch.allclose(
        scaled["scene_radius_vggt"],
        summary["scene_radius_vggt"] * 3.0,
        atol=1e-4,
    )
    assert torch.allclose(
        scaled["cue"],
        summary["cue"],
        atol=1e-5,
    )


def test_scene_radius_uses_one_reference_camera_frame() -> None:
    depth = torch.ones(1, 2, 1, 1)
    intrinsics = torch.eye(3).view(1, 1, 3, 3).expand(1, 2, -1, -1).clone()
    camera = torch.zeros(1, 2, 3, 4)
    camera[..., :3, :3] = torch.eye(3)
    # The second camera centre is x=10, so its optical-axis point is
    # (10, 0, 1) in the first-camera frame.
    camera[:, 1, 0, 3] = -10.0
    radius = _scene_radius(depth, intrinsics, camera)
    expected = (torch.tensor(1.0) + torch.sqrt(torch.tensor(101.0))) / 2
    assert torch.allclose(radius[0], expected, atol=1e-6)
