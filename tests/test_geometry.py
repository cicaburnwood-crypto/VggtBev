import torch

from vggt_bev.config import BatchBEVGridSpec, BEVGridSpec
from vggt_bev.geometry.direct_projection import (
    camera_origins_in_latest_camera,
    project_vggt_geometry_bevs,
    scale_camera_translations,
)
from vggt_bev.geometry.frames import opencv_points_to_reference_bev
from vggt_bev.geometry.ground import (
    align_geometry_to_ground,
    align_geometry_to_ground_raw,
    stabilize_sequence_intrinsics,
)
from vggt_bev.geometry.lift import backproject_pixels, patch_centers, sample_image_at_pixels
from vggt_bev.geometry.splat import bilinear_splat, raycast_free_evidence


def test_grid_metric_pixel_round_trip() -> None:
    grid = BEVGridSpec(extent_m=8.0, height=512, width=512)
    points = torch.tensor([[0.0, 0.0], [1.25, -2.0], [-3.0, 3.5]])
    assert torch.allclose(grid.pixel_to_metric(grid.metric_to_pixel(points)), points, atol=1e-6)


def test_opencv_forward_maps_to_positive_bev_forward() -> None:
    point = torch.tensor([[[[0.0, 0.0, 1.0]]]])
    # OpenCV +z becomes Habitat -z; this camera rotation maps Habitat -z to world +z.
    camera_to_world = torch.eye(4).reshape(1, 1, 4, 4)
    camera_to_world[..., 2, 2] = -1.0
    reference_world_from_bev = torch.eye(3)[None]
    transformed = opencv_points_to_reference_bev(
        point, camera_to_world, reference_world_from_bev, torch.tensor([0.0])
    )
    assert torch.allclose(transformed[0, 0, 0], torch.tensor([0.0, 1.0, 0.0]))


def test_patch_sampling_and_backprojection() -> None:
    centers = patch_centers(32, 48, 16)
    assert centers.tolist() == [
        [7.5, 7.5],
        [23.5, 7.5],
        [39.5, 7.5],
        [7.5, 23.5],
        [23.5, 23.5],
        [39.5, 23.5],
    ]
    image = torch.arange(32 * 48, dtype=torch.float32).reshape(1, 1, 1, 32, 48)
    sampled = sample_image_at_pixels(image, centers)
    assert sampled.shape == (1, 1, 6, 1)
    depth = torch.ones((1, 1, 1))
    pixel = torch.tensor([[1.0, 2.0]])
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3)
    point = backproject_pixels(depth, pixel, intrinsics)
    assert torch.allclose(point, torch.tensor([[[[1.0, 2.0, 1.0]]]]))


def test_bilinear_splat_at_grid_center() -> None:
    grid = BEVGridSpec(extent_m=5.0, height=5, width=5)
    points = torch.tensor([[[[0.0, 0.0]]]])
    features = torch.tensor([[[[2.0, 3.0]]]])
    weights = torch.ones((1, 1, 1))
    valid = torch.ones((1, 1, 1), dtype=torch.bool)
    raster, evidence = bilinear_splat(points, features, weights, valid, grid)
    assert torch.allclose(raster[0, :, 2, 2], torch.tensor([2.0, 3.0]))
    assert evidence[0, 0, 2, 2] == 1.0
    assert evidence.sum() == 1.0


def test_batch_grid_supports_different_learned_ranges() -> None:
    grid = BatchBEVGridSpec(
        extent=torch.tensor([2.0, 4.0]),
        height=5,
        width=5,
    )
    points = torch.tensor(
        [
            [[[0.5, 0.0]]],
            [[[0.5, 0.0]]],
        ]
    )
    pixels = grid.spatial_to_pixel(points)

    assert torch.allclose(pixels[0, 0, 0], torch.tensor([3.25, 2.0]))
    assert torch.allclose(pixels[1, 0, 0], torch.tensor([2.625, 2.0]))


def test_raycast_marks_cells_before_endpoint() -> None:
    grid = BEVGridSpec(extent_m=8.0, height=33, width=33)
    origins = torch.tensor([[[0.0, 0.0]]])
    endpoints = torch.tensor([[[[0.0, 3.0]]]])
    weights = torch.ones((1, 1, 1))
    valid = torch.ones((1, 1, 1), dtype=torch.bool)
    evidence = raycast_free_evidence(origins, endpoints, weights, valid, grid, steps=8)
    assert evidence.sum() > 0
    assert evidence[0, 0, 16, 16] > 0


def test_direct_vggt_geometry_projection_marks_floor_and_obstacle() -> None:
    depth = torch.tensor(
        [[[[[1.0], [1.0]], [[2.0], [2.0]]]]]
    )
    confidence = torch.full((1, 1, 2, 2), 2.0)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3)
    intrinsics[..., 1, 1] = 4.0
    camera_from_world = torch.cat(
        (
            torch.eye(3),
            torch.zeros(3, 1),
        ),
        dim=1,
    ).reshape(1, 1, 3, 4)

    output = project_vggt_geometry_bevs(
        dense_depth=depth,
        dense_confidence=confidence,
        estimated_intrinsics=intrinsics,
        estimated_camera_from_world=camera_from_world,
        image_valid=torch.ones((1, 1, 2, 2), dtype=torch.bool),
        frame_valid=torch.ones((1, 1), dtype=torch.bool),
        camera_height_m=torch.tensor([0.5]),
        single_extent_m=4.0,
        merged_extent_m=8.0,
        output_size=9,
        depth_scale=torch.tensor(1.0),
        pixel_stride=1,
    )

    assert set(output["single_labels"].unique().tolist()) == {0, 112, 255}
    assert set(output["merged_labels"].unique().tolist()) == {0, 112, 255}


def test_predicted_pose_translation_uses_same_metric_scale_as_depth() -> None:
    camera_from_world = torch.zeros((1, 2, 3, 4))
    camera_from_world[..., :3] = torch.eye(3)
    camera_from_world[0, 1, 0, 3] = -1.0

    scaled = scale_camera_translations(
        camera_from_world,
        torch.tensor(2.0),
    )
    origins = camera_origins_in_latest_camera(
        scaled,
        torch.ones((1, 2), dtype=torch.bool),
    )

    assert torch.allclose(origins[0, 0], torch.tensor([-2.0, 0.0]))
    assert torch.allclose(origins[0, 1], torch.tensor([0.0, 0.0]))


def test_predicted_pose_scale_supports_backward() -> None:
    camera_from_world = torch.zeros((1, 2, 3, 4))
    camera_from_world[..., :3] = torch.eye(3)
    camera_from_world[0, 1, 0, 3] = -1.0
    scale = torch.tensor(2.0, requires_grad=True)

    scaled = scale_camera_translations(camera_from_world, scale)
    scaled[..., 3].sum().backward()

    assert torch.allclose(scale.grad, torch.tensor(-1.0))


def _tilted_floor_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(-2.0, 2.0, 25)
    z = torch.linspace(0.5, 5.0, 25)
    x_grid, z_grid = torch.meshgrid(x, z, indexing="ij")
    y_grid = 2.0 + 0.08 * x_grid - 0.04 * z_grid
    floor = torch.stack((x_grid, y_grid, z_grid), dim=-1).reshape(-1, 3)
    outliers = torch.stack(
        (
            torch.linspace(-2.0, 2.0, 200),
            torch.linspace(0.05, 1.4, 200),
            torch.full((200,), 2.5),
        ),
        dim=-1,
    )
    points = torch.cat((floor, outliers)).reshape(1, 1, -1, 3)
    confidence = torch.cat(
        (
            torch.ones(floor.shape[0]),
            torch.full((outliers.shape[0],), 0.2),
        )
    ).reshape(1, 1, -1)
    valid = torch.ones_like(confidence, dtype=torch.bool)
    return points, confidence, valid


def test_ground_alignment_recovers_plane_and_metric_camera_height() -> None:
    points, confidence, valid = _tilted_floor_geometry()
    floor_point_count = 25 * 25
    alignment = align_geometry_to_ground(
        points,
        torch.zeros((1, 1, 3)),
        confidence,
        valid,
        torch.tensor([0.5]),
    )

    floor_heights = alignment.points[0, 0, :floor_point_count, 2]
    assert floor_heights.abs().max() < 1e-4
    assert torch.allclose(
        alignment.camera_origins[0, 0, 2],
        torch.tensor(0.5),
        atol=1e-5,
    )
    assert torch.allclose(
        alignment.metric_scale,
        torch.tensor([0.2510]),
        atol=2e-3,
    )
    assert alignment.inlier_fraction.item() > 0.9
    assert not alignment.fallback_used.any()


def test_camera_height_scale_is_invariant_to_vggt_scene_scale() -> None:
    points, confidence, valid = _tilted_floor_geometry()
    camera_origins = torch.zeros((1, 1, 3))
    base = align_geometry_to_ground(
        points,
        camera_origins,
        confidence,
        valid,
        torch.tensor([0.5]),
    )
    scaled = align_geometry_to_ground(
        points * 3.0,
        camera_origins * 3.0,
        confidence,
        valid,
        torch.tensor([0.5]),
    )

    assert torch.allclose(base.points, scaled.points, atol=2e-4)
    assert torch.allclose(
        scaled.metric_scale,
        base.metric_scale / 3.0,
        atol=1e-5,
    )


def test_raw_ground_alignment_preserves_vggt_scale() -> None:
    points, confidence, valid = _tilted_floor_geometry()
    alignment = align_geometry_to_ground_raw(
        points,
        torch.zeros((1, 1, 3)),
        confidence,
        valid,
    )

    assert torch.equal(alignment.metric_scale, torch.ones(1))
    assert torch.allclose(
        alignment.camera_origins[0, 0, 2],
        alignment.predicted_camera_height[0],
        atol=1e-5,
    )


def test_sparse_ground_geometry_uses_level_fallback() -> None:
    points = torch.tensor(
        [[[[0.0, 2.0, 1.0], [0.5, 2.0, 2.0], [-0.5, 2.0, 3.0]]]]
    )
    confidence = torch.ones((1, 1, 3))
    valid = torch.ones_like(confidence, dtype=torch.bool)

    alignment = align_geometry_to_ground(
        points,
        torch.zeros((1, 1, 3)),
        confidence,
        valid,
        torch.tensor([0.5]),
        minimum_points=48,
    )

    assert alignment.fallback_used.all()
    assert torch.allclose(
        alignment.normal,
        torch.tensor([[0.0, -1.0, 0.0]]),
    )
    assert torch.allclose(alignment.metric_scale, torch.tensor([0.25]))
    assert torch.isfinite(alignment.points).all()


def test_intrinsics_are_constant_after_sequence_stabilization() -> None:
    intrinsics = torch.eye(3).expand(1, 3, 3, 3).clone()
    intrinsics[0, :, 0, 0] = torch.tensor([300.0, 320.0, 900.0])
    intrinsics[0, :, 1, 1] = torch.tensor([301.0, 321.0, 901.0])
    intrinsics[0, :, 0, 2] = torch.tensor([255.5, 255.0, 999.0])
    intrinsics[0, :, 1, 2] = torch.tensor([191.5, 191.0, 999.0])
    stabilized = stabilize_sequence_intrinsics(
        intrinsics,
        torch.tensor([[True, True, False]]),
    )

    assert torch.equal(stabilized[0, 0], stabilized[0, 1])
    assert torch.equal(stabilized[0, 1], stabilized[0, 2])
    assert stabilized[0, 0, 0, 0] == 300.0
    assert stabilized[0, 0, 1, 1] == 301.0
