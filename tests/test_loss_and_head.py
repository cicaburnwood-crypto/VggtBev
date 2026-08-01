import torch

from vggt_bev.config import BEVGridSpec
from vggt_bev.losses import method2_loss
from vggt_bev.models.method2 import Method2BEVHead, Method2System, render_observed_bev


def test_unknown_pixels_never_change_occupancy_loss() -> None:
    target = torch.tensor([[[0, 255], [112, 112]]], dtype=torch.uint8)
    base = {
        "occupancy_logit": torch.zeros((1, 2, 2)),
        "observed_logit": torch.zeros((1, 2, 2)),
    }
    changed = {name: value.clone() for name, value in base.items()}
    changed["occupancy_logit"][0, 1] = torch.tensor([100.0, -100.0])
    first = method2_loss(base, target)
    second = method2_loss(changed, target)
    assert torch.allclose(first["occupancy_loss"], second["occupancy_loss"])
    assert torch.allclose(first["dice_loss"], second["dice_loss"])


def test_method2_head_shapes_and_gradient() -> None:
    torch.manual_seed(3)
    head = Method2BEVHead(feature_dim=8, hidden_dim=16, output_size=32, ray_steps=4)
    features = torch.randn((1, 2, 6, 8))
    points = torch.randn((1, 2, 6, 3))
    points[..., :2] *= 0.5
    points[..., 2] = 0.8
    confidence = torch.full((1, 2, 6), 0.75)
    valid = torch.ones((1, 2, 6), dtype=torch.bool)
    origins = torch.zeros((1, 2, 2))
    output = head(
        features,
        points,
        confidence,
        valid,
        origins,
        BEVGridSpec(extent_m=4.0, height=16, width=16),
    )
    assert output["occupancy_logit"].shape == (1, 32, 32)
    assert output["observed_logit"].shape == (1, 32, 32)
    (output["occupancy_logit"].mean() + output["observed_logit"].mean()).backward()
    assert head.point_encoder[1].weight.grad is not None


def test_render_preserves_unknown_semantics() -> None:
    occupancy = torch.tensor([[[10.0, -10.0, 10.0]]])
    observed = torch.tensor([[[10.0, 10.0, -10.0]]])
    rendered = render_observed_bev(occupancy, observed)
    assert rendered.tolist() == [[[0, 255, 112]]]


class _FakeAdapter(torch.nn.Module):
    def forward(
        self,
        images: torch.Tensor,
        *,
        include_geometry: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch, frames, _, height, width = images.shape
        patch_count = (height // 16) * (width // 16)
        output = {
            "patch_features": torch.randn(batch, frames, patch_count, 8),
            "depth": torch.ones(batch, frames, patch_count),
            "confidence": torch.full((batch, frames, patch_count), 2.0),
            "patch_centers": torch.tensor(
                [
                    [x + 7.5, y + 7.5]
                    for y in range(0, height, 16)
                    for x in range(0, width, 16)
                ]
            ),
        }
        if include_geometry:
            camera_from_world = torch.zeros(batch, frames, 3, 4)
            camera_from_world[..., :3] = torch.eye(3)
            output.update(
                {
                    "dense_depth": torch.ones(
                        batch,
                        frames,
                        height,
                        width,
                        1,
                    ),
                    "dense_confidence": torch.full(
                        (batch, frames, height, width),
                        2.0,
                    ),
                    "estimated_intrinsics": (
                        torch.eye(3).expand(batch, frames, 3, 3).clone()
                    ),
                    "estimated_camera_from_world": camera_from_world,
                }
            )
        return output


def test_dual_system_emits_single_and_merged_products() -> None:
    system = Method2System(
        _FakeAdapter(),
        Method2BEVHead(feature_dim=8, hidden_dim=8, output_size=16, ray_steps=2),
        merged_head=Method2BEVHead(
            feature_dim=8, hidden_dim=8, output_size=16, ray_steps=2
        ),
        bev_feature_size=8,
    )
    batch = {
        "images": torch.randn(1, 2, 3, 32, 32),
        "intrinsics": torch.eye(3).expand(1, 2, 3, 3).clone(),
        "camera_to_world": torch.eye(4).expand(1, 2, 4, 4).clone(),
        "reference_world_from_bev": torch.eye(3).expand(1, 3, 3).clone(),
        "floor_y": torch.zeros(1),
        "image_valid": torch.ones(1, 2, 32, 32, dtype=torch.bool),
        "frame_valid": torch.ones(1, 2, dtype=torch.bool),
        "single_target_extent_m": torch.tensor([5.0]),
        "merged_target_extent_m": torch.tensor([8.0]),
    }
    output = system(batch)
    assert set(output) == {"single", "merged", "depth_scale"}
    assert output["single"]["occupancy_logit"].shape == (1, 16, 16)
    assert output["merged"]["occupancy_logit"].shape == (1, 16, 16)


def test_runtime_system_uses_predicted_geometry_without_gt_inputs() -> None:
    system = Method2System(
        _FakeAdapter(),
        Method2BEVHead(feature_dim=8, hidden_dim=8, output_size=16, ray_steps=2),
        merged_head=Method2BEVHead(
            feature_dim=8,
            hidden_dim=8,
            output_size=16,
            ray_steps=2,
        ),
        bev_feature_size=8,
    )
    batch = {
        "images": torch.randn(1, 2, 3, 32, 32),
        "image_valid": torch.ones(1, 2, 32, 32, dtype=torch.bool),
        "frame_valid": torch.ones(1, 2, dtype=torch.bool),
        "camera_height_m": torch.tensor([0.35]),
        "single_target_extent_m": torch.tensor([5.0]),
        "merged_target_extent_m": torch.tensor([8.0]),
        "use_predicted_geometry": True,
    }

    output = system(batch)

    assert set(output) == {"single", "merged", "depth_scale"}
    assert output["single"]["occupancy_logit"].shape == (1, 16, 16)


def test_raw_vggt_mode_requires_no_height_and_applies_no_scale() -> None:
    system = Method2System(
        _FakeAdapter(),
        Method2BEVHead(
            feature_dim=8,
            hidden_dim=8,
            output_size=16,
            filter_by_height=False,
            ray_steps=2,
        ),
        merged_head=Method2BEVHead(
            feature_dim=8,
            hidden_dim=8,
            output_size=16,
            filter_by_height=False,
            ray_steps=2,
        ),
        bev_feature_size=8,
        learn_depth_scale=False,
        metric_scale_mode="vggt_raw",
        stabilize_intrinsics=True,
        ground_minimum_points=4,
        minimum_depth_m=1e-4,
        maximum_depth_m=1e4,
    )
    batch = {
        "images": torch.randn(1, 2, 3, 32, 32),
        "image_valid": torch.ones(1, 2, 32, 32, dtype=torch.bool),
        "frame_valid": torch.ones(1, 2, dtype=torch.bool),
        "single_target_extent_m": torch.tensor([5.0]),
        "merged_target_extent_m": torch.tensor([8.0]),
        "use_predicted_geometry": True,
    }

    output = system(batch)

    assert "camera_height_m" not in batch
    assert torch.equal(output["depth_scale"], torch.ones(1))
    assert output["single"]["occupancy_logit"].shape == (1, 16, 16)


def test_normalized_vggt_mode_learns_range_without_fixed_extents() -> None:
    system = Method2System(
        _FakeAdapter(),
        Method2BEVHead(
            feature_dim=8,
            hidden_dim=8,
            output_size=16,
            filter_by_height=False,
            ray_steps=2,
        ),
        merged_head=Method2BEVHead(
            feature_dim=8,
            hidden_dim=8,
            output_size=24,
            filter_by_height=False,
            ray_steps=2,
        ),
        bev_feature_size=8,
        merged_bev_feature_size=12,
        learn_depth_scale=False,
        metric_scale_mode="vggt_normalized",
        normalizer_hidden_dim=8,
        stabilize_intrinsics=True,
        ground_minimum_points=4,
        minimum_depth_m=1e-4,
        maximum_depth_m=1e4,
    )
    batch = {
        "images": torch.randn(1, 2, 3, 32, 32),
        "image_valid": torch.ones(1, 2, 32, 32, dtype=torch.bool),
        "frame_valid": torch.ones(1, 2, dtype=torch.bool),
        "use_predicted_geometry": True,
    }

    output = system(batch)

    assert "camera_height_m" not in batch
    assert "single_target_extent_m" not in batch
    assert output["single"]["occupancy_logit"].shape == (1, 16, 16)
    assert output["merged"]["occupancy_logit"].shape == (1, 24, 24)
    normalization = output["normalization"]
    assert normalization["reference_scale_vggt"].item() > 0
    assert normalization["normalized_units_per_output_pixel"].item() > 0
    assert torch.allclose(
        normalization["merged_span_normalized"]
        / normalization["single_span_normalized"],
        torch.tensor([1.5]),
    )


def test_strict_method2_system_reports_per_window_ground_scale() -> None:
    system = Method2System(
        _FakeAdapter(),
        Method2BEVHead(feature_dim=8, hidden_dim=8, output_size=16, ray_steps=8),
        merged_head=Method2BEVHead(
            feature_dim=8,
            hidden_dim=8,
            output_size=16,
            ray_steps=8,
        ),
        bev_feature_size=16,
        learn_depth_scale=False,
        metric_scale_mode="camera_height",
        stabilize_intrinsics=True,
        ground_minimum_points=4,
    )
    batch = {
        "images": torch.randn(1, 2, 3, 32, 32),
        "image_valid": torch.ones(1, 2, 32, 32, dtype=torch.bool),
        "frame_valid": torch.ones(1, 2, dtype=torch.bool),
        "camera_height_m": torch.tensor([0.35]),
        "single_target_extent_m": torch.tensor([5.0]),
        "merged_target_extent_m": torch.tensor([8.0]),
        "use_predicted_geometry": True,
        "return_geometry": True,
    }

    output = system(batch)
    loss = (
        output["single"]["occupancy_logit"].mean()
        + output["merged"]["observed_logit"].mean()
    )
    loss.backward()

    assert set(output) == {
        "single",
        "merged",
        "depth_scale",
        "ground",
        "geometry",
    }
    assert output["depth_scale"].shape == (1,)
    assert torch.isfinite(output["depth_scale"]).all()
    assert output["ground"]["normal"].shape == (1, 3)
    assert output["geometry"]["single_labels"].shape == (1, 16, 16)
    assert output["geometry"]["merged_labels"].shape == (1, 16, 16)
    assert system.head.point_encoder[1].weight.grad is not None
    assert system.log_depth_scale.grad is None
