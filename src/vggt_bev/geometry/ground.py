from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GroundPlaneEstimate:
    """A ground plane expressed in the latest OpenCV camera frame.

    ``normal`` points upward, toward the reference camera. The plane equation is
    ``normal · point + offset = 0`` and ``offset`` is therefore the predicted
    camera height in VGGT's internal spatial unit.
    """

    normal: torch.Tensor
    offset: torch.Tensor
    camera_height: torch.Tensor
    inlier_fraction: torch.Tensor
    fallback_used: torch.Tensor


@dataclass(frozen=True)
class GroundAlignment:
    """Ground-aligned metric geometry and the transform used to produce it."""

    points: torch.Tensor
    camera_origins: torch.Tensor
    metric_scale: torch.Tensor
    normal: torch.Tensor
    origin: torch.Tensor
    right: torch.Tensor
    forward: torch.Tensor
    predicted_camera_height: torch.Tensor
    inlier_fraction: torch.Tensor
    fallback_used: torch.Tensor


def stabilize_sequence_intrinsics(
    intrinsics: torch.Tensor,
    frame_valid: torch.Tensor,
) -> torch.Tensor:
    """Use one robust VGGT-predicted intrinsic matrix for every valid frame."""

    if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [B, N, 3, 3]")
    if frame_valid.shape != intrinsics.shape[:2]:
        raise ValueError("frame_valid must have shape [B, N]")
    stabilized = torch.zeros_like(intrinsics)
    for batch_index in range(intrinsics.shape[0]):
        valid = frame_valid[batch_index]
        if not valid.any():
            raise ValueError("every sample must contain at least one valid frame")
        selected = intrinsics[batch_index, valid]
        fx = selected[:, 0, 0].median()
        fy = selected[:, 1, 1].median()
        cx = selected[:, 0, 2].median()
        cy = selected[:, 1, 2].median()
        values = torch.stack((fx, fy, cx, cy))
        if not torch.isfinite(values).all() or fx <= 0 or fy <= 0:
            raise ValueError("VGGT predicted invalid sequence intrinsics")
        matrix = intrinsics.new_zeros((3, 3))
        matrix[0, 0] = fx
        matrix[1, 1] = fy
        matrix[0, 2] = cx
        matrix[1, 2] = cy
        matrix[2, 2] = 1.0
        stabilized[batch_index] = matrix
    return stabilized


def _fit_one_ground_plane(
    points: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    *,
    minimum_points: int,
    candidate_quantile: float,
    maximum_candidate_quantile: float,
    irls_iterations: int,
    huber_delta: float,
    maximum_tilt_degrees: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    mask = (
        valid
        & torch.isfinite(points).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence > 0)
        & (points[:, 1] > 0)
    )
    usable = points[mask]
    usable_confidence = confidence[mask].clamp(1e-4, 1.0)
    camera_up = points.new_tensor((0.0, -1.0, 0.0))
    if usable.shape[0] < minimum_points:
        # Sparse/low-confidence views must not terminate a long distributed
        # run. Fall back to a level ground plane and recover its internal-unit
        # camera height from whatever finite below-camera geometry is present.
        fallback_mask = (
            torch.isfinite(points).all(dim=-1)
            & (points[:, 1] > 0)
        )
        fallback_y = points[fallback_mask, 1]
        if usable.shape[0] > 0:
            offset = torch.quantile(usable[:, 1], candidate_quantile)
        elif fallback_y.shape[0] > 0:
            offset = torch.quantile(fallback_y, candidate_quantile)
        else:
            offset = points.new_tensor(1.0)
        offset = offset.clamp_min(1e-6)
        if usable.shape[0] == 0:
            inlier_fraction = points.new_zeros(())
        else:
            signed_distance = usable @ camera_up + offset
            residual_scale = (
                1.4826
                * (
                    signed_distance - signed_distance.median()
                ).abs().median()
                + 1e-6
            )
            inlier_threshold = torch.maximum(
                2.5 * residual_scale,
                0.01 * offset.abs(),
            )
            inlier_fraction = (
                usable_confidence
                * (
                    signed_distance.abs() <= inlier_threshold
                ).to(usable_confidence.dtype)
            ).sum() / usable_confidence.sum().clamp_min(1e-6)
        return camera_up, offset, inlier_fraction, True

    y = usable[:, 1]
    lower = torch.quantile(y, candidate_quantile)
    upper = torch.quantile(y, maximum_candidate_quantile)
    candidate_mask = (y >= lower) & (y <= upper)
    candidates = usable[candidate_mask]
    candidate_confidence = usable_confidence[candidate_mask]
    if candidates.shape[0] < minimum_points:
        candidates = usable
        candidate_confidence = usable_confidence

    design = torch.stack(
        (
            candidates[:, 0],
            candidates[:, 2],
            torch.ones_like(candidates[:, 0]),
        ),
        dim=-1,
    )
    target = candidates[:, 1]
    theta = torch.stack(
        (
            target.new_zeros(()),
            target.new_zeros(()),
            target.median(),
        )
    )
    robust_weights = torch.ones_like(target)
    ridge = torch.diag(target.new_tensor((1e-4, 1e-4, 1e-6)))
    for _ in range(irls_iterations):
        residual = target - design @ theta
        centered = residual - residual.median()
        robust_scale = 1.4826 * centered.abs().median() + 1e-6
        normalized = centered.abs() / (huber_delta * robust_scale)
        robust_weights = torch.where(
            normalized <= 1.0,
            torch.ones_like(normalized),
            normalized.reciprocal(),
        )
        weights = candidate_confidence * robust_weights
        weighted_design = design * weights[:, None]
        lhs = design.transpose(0, 1) @ weighted_design + ridge
        rhs = design.transpose(0, 1) @ (weights * target)
        theta = torch.linalg.solve(lhs, rhs)

    raw_normal = torch.stack((theta[0], theta.new_tensor(-1.0), theta[1]))
    normal_length = torch.linalg.vector_norm(raw_normal)
    normal = raw_normal / normal_length.clamp_min(1e-8)
    offset = theta[2] / normal_length.clamp_min(1e-8)
    cosine = torch.dot(normal, camera_up)
    minimum_cosine = math.cos(math.radians(maximum_tilt_degrees))
    fallback = bool(
        not torch.isfinite(normal).all()
        or not torch.isfinite(offset)
        or offset <= 1e-6
        or cosine < minimum_cosine
    )
    if fallback:
        normal = camera_up
        offset = torch.quantile(y, candidate_quantile)

    signed_distance = usable @ normal + offset
    residual_scale = (
        1.4826
        * (signed_distance - signed_distance.median()).abs().median()
        + 1e-6
    )
    inlier_threshold = torch.maximum(
        2.5 * residual_scale,
        0.01 * offset.abs(),
    )
    inlier_fraction = (
        usable_confidence
        * (signed_distance.abs() <= inlier_threshold).to(usable_confidence.dtype)
    ).sum() / usable_confidence.sum().clamp_min(1e-6)
    return normal, offset, inlier_fraction, fallback


def estimate_ground_plane(
    points_reference: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    *,
    minimum_points: int = 48,
    candidate_quantile: float = 0.55,
    maximum_candidate_quantile: float = 0.995,
    irls_iterations: int = 5,
    huber_delta: float = 2.5,
    maximum_tilt_degrees: float = 35.0,
) -> GroundPlaneEstimate:
    """Robustly fit the floor after all points enter the latest-camera frame."""

    if points_reference.ndim != 4 or points_reference.shape[-1] != 3:
        raise ValueError("points_reference must have shape [B, N, P, 3]")
    if confidence.shape != points_reference.shape[:-1] or valid.shape != confidence.shape:
        raise ValueError("confidence and valid must have shape [B, N, P]")
    if minimum_points < 3:
        raise ValueError("minimum_points must be at least three")
    if not 0.0 < candidate_quantile < maximum_candidate_quantile < 1.0:
        raise ValueError("ground candidate quantiles must satisfy 0 < low < high < 1")
    if irls_iterations < 1 or huber_delta <= 0:
        raise ValueError("IRLS settings must be positive")
    if not 0.0 < maximum_tilt_degrees < 90.0:
        raise ValueError("maximum_tilt_degrees must be between zero and 90")

    normals: list[torch.Tensor] = []
    offsets: list[torch.Tensor] = []
    inlier_fractions: list[torch.Tensor] = []
    fallbacks: list[bool] = []
    for batch_index in range(points_reference.shape[0]):
        normal, offset, inlier_fraction, fallback = _fit_one_ground_plane(
            points_reference[batch_index].reshape(-1, 3),
            confidence[batch_index].reshape(-1),
            valid[batch_index].reshape(-1),
            minimum_points=minimum_points,
            candidate_quantile=candidate_quantile,
            maximum_candidate_quantile=maximum_candidate_quantile,
            irls_iterations=irls_iterations,
            huber_delta=huber_delta,
            maximum_tilt_degrees=maximum_tilt_degrees,
        )
        normals.append(normal)
        offsets.append(offset)
        inlier_fractions.append(inlier_fraction)
        fallbacks.append(fallback)

    offset_tensor = torch.stack(offsets)
    return GroundPlaneEstimate(
        normal=torch.stack(normals),
        offset=offset_tensor,
        camera_height=offset_tensor,
        inlier_fraction=torch.stack(inlier_fractions),
        fallback_used=torch.tensor(
            fallbacks,
            dtype=torch.bool,
            device=points_reference.device,
        ),
    )


def project_to_ground_frame(
    points_reference: torch.Tensor,
    *,
    normal: torch.Tensor,
    origin: torch.Tensor,
    right: torch.Tensor,
    forward: torch.Tensor,
    metric_scale: torch.Tensor,
) -> torch.Tensor:
    """Convert latest-camera coordinates into metric right/forward/height."""

    if points_reference.shape[0] != normal.shape[0]:
        raise ValueError("ground transform batch size does not match points")
    expand = (slice(None),) + (None,) * (points_reference.ndim - 2)
    delta = points_reference - origin[expand]
    right_coordinate = (delta * right[expand]).sum(dim=-1)
    forward_coordinate = (delta * forward[expand]).sum(dim=-1)
    height_coordinate = (delta * normal[expand]).sum(dim=-1)
    coordinates = torch.stack(
        (right_coordinate, forward_coordinate, height_coordinate),
        dim=-1,
    )
    scale_expand = (slice(None),) + (None,) * (coordinates.ndim - 1)
    return coordinates * metric_scale[scale_expand]


def align_geometry_to_ground(
    points_reference: torch.Tensor,
    camera_origins_reference: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    physical_camera_height_m: torch.Tensor,
    *,
    minimum_points: int = 48,
    candidate_quantile: float = 0.55,
    maximum_candidate_quantile: float = 0.995,
    irls_iterations: int = 5,
    huber_delta: float = 2.5,
    maximum_tilt_degrees: float = 35.0,
    minimum_metric_scale: float = 1e-3,
    maximum_metric_scale: float = 1e3,
) -> GroundAlignment:
    """Estimate ground, recover per-window metric scale, and align geometry."""

    if camera_origins_reference.shape != points_reference.shape[:2] + (3,):
        raise ValueError("camera_origins_reference must have shape [B, N, 3]")
    if physical_camera_height_m.shape != (points_reference.shape[0],):
        raise ValueError("physical_camera_height_m must have shape [B]")
    if (
        not torch.isfinite(physical_camera_height_m).all()
        or (physical_camera_height_m <= 0).any()
    ):
        raise ValueError("physical camera height must be finite and positive")

    estimate = estimate_ground_plane(
        points_reference,
        confidence,
        valid,
        minimum_points=minimum_points,
        candidate_quantile=candidate_quantile,
        maximum_candidate_quantile=maximum_candidate_quantile,
        irls_iterations=irls_iterations,
        huber_delta=huber_delta,
        maximum_tilt_degrees=maximum_tilt_degrees,
    )
    metric_scale = (
        physical_camera_height_m / estimate.camera_height.clamp_min(1e-8)
    ).clamp(minimum_metric_scale, maximum_metric_scale)
    return _apply_ground_alignment(
        points_reference,
        camera_origins_reference,
        estimate,
        metric_scale,
    )


def align_geometry_to_ground_raw(
    points_reference: torch.Tensor,
    camera_origins_reference: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    *,
    minimum_points: int = 48,
    candidate_quantile: float = 0.55,
    maximum_candidate_quantile: float = 0.995,
    irls_iterations: int = 5,
    huber_delta: float = 2.5,
    maximum_tilt_degrees: float = 35.0,
) -> GroundAlignment:
    """Ground-align without changing VGGT's native reconstruction scale."""

    if camera_origins_reference.shape != points_reference.shape[:2] + (3,):
        raise ValueError("camera_origins_reference must have shape [B, N, 3]")
    estimate = estimate_ground_plane(
        points_reference,
        confidence,
        valid,
        minimum_points=minimum_points,
        candidate_quantile=candidate_quantile,
        maximum_candidate_quantile=maximum_candidate_quantile,
        irls_iterations=irls_iterations,
        huber_delta=huber_delta,
        maximum_tilt_degrees=maximum_tilt_degrees,
    )
    identity_scale = torch.ones_like(estimate.camera_height)
    return _apply_ground_alignment(
        points_reference,
        camera_origins_reference,
        estimate,
        identity_scale,
    )


def _apply_ground_alignment(
    points_reference: torch.Tensor,
    camera_origins_reference: torch.Tensor,
    estimate: GroundPlaneEstimate,
    spatial_scale: torch.Tensor,
) -> GroundAlignment:
    origin = -estimate.offset[:, None] * estimate.normal

    camera_right = points_reference.new_tensor((1.0, 0.0, 0.0)).expand_as(
        estimate.normal
    )
    right = camera_right - (
        camera_right * estimate.normal
    ).sum(dim=-1, keepdim=True) * estimate.normal
    right = right / torch.linalg.vector_norm(
        right,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)
    forward = torch.linalg.cross(estimate.normal, right, dim=-1)
    forward = forward / torch.linalg.vector_norm(
        forward,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)

    aligned_points = project_to_ground_frame(
        points_reference,
        normal=estimate.normal,
        origin=origin,
        right=right,
        forward=forward,
        metric_scale=spatial_scale,
    )
    aligned_origins = project_to_ground_frame(
        camera_origins_reference,
        normal=estimate.normal,
        origin=origin,
        right=right,
        forward=forward,
        metric_scale=spatial_scale,
    )
    return GroundAlignment(
        points=aligned_points,
        camera_origins=aligned_origins,
        metric_scale=spatial_scale,
        normal=estimate.normal,
        origin=origin,
        right=right,
        forward=forward,
        predicted_camera_height=estimate.camera_height,
        inlier_fraction=estimate.inlier_fraction,
        fallback_used=estimate.fallback_used,
    )
