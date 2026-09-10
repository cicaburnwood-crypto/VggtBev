"""Offline metric-scale evaluation helpers for VGGT camera trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryScaleMetrics:
    frame_count: int
    height_scale_m_per_vggt: float
    oracle_scale_m_per_vggt: float
    scale_absolute_error_m_per_vggt: float
    scale_relative_error: float
    scale_log_absolute_error: float
    true_path_length_m: float
    height_scaled_path_length_m: float
    path_length_absolute_error_m: float
    path_length_relative_error: float
    height_scaled_ate_rmse_m: float
    oracle_scaled_ate_rmse_m: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleChainMetrics:
    """Separate the learned token error from the camera-height scale error."""

    model_scale_token_m_per_vggt: float
    model_scale_token_absolute_error_m_per_vggt: float
    model_scale_token_relative_error: float
    height_to_token_ratio_m_per_bev: float
    height_to_token_ratio_absolute_error: float
    chained_single_extent_m: float
    chained_single_extent_absolute_error_m: float
    chained_single_extent_relative_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def camera_centers_from_world_to_camera(
    camera_from_world: np.ndarray,
) -> np.ndarray:
    """Recover camera centers from a sequence of world-to-camera matrices."""

    extrinsics = np.asarray(camera_from_world, dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError("camera_from_world must have shape [T, 3, 4]")
    if not np.isfinite(extrinsics).all():
        raise ValueError("camera extrinsics must be finite")
    centers = []
    for extrinsic in extrinsics:
        rotation = extrinsic[:, :3]
        translation = extrinsic[:, 3]
        centers.append(np.linalg.solve(rotation, -translation))
    return np.stack(centers)


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(points.shape[0], k=1)
    return np.linalg.norm(points[indices[0]] - points[indices[1]], axis=1)


def _fixed_scale_ate_rmse(
    predicted_centers: np.ndarray,
    true_centers: np.ndarray,
    scale: float,
) -> float:
    """Rigidly align a fixed-scale trajectory, then return translation RMSE."""

    scaled = predicted_centers * scale
    predicted_centered = scaled - scaled.mean(axis=0, keepdims=True)
    true_centered = true_centers - true_centers.mean(axis=0, keepdims=True)
    covariance = predicted_centered.T @ true_centered
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    aligned = (rotation @ predicted_centered.T).T
    residual = aligned - true_centered
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def evaluate_height_scale_trajectory(
    *,
    predicted_camera_from_world_vggt: np.ndarray,
    true_camera_centers_world_m: np.ndarray,
    height_scale_m_per_vggt: float,
    minimum_true_baseline_m: float = 0.05,
) -> TrajectoryScaleMetrics:
    """Compare camera-height scale with an oracle fitted from GT motion.

    The oracle is the least-squares scale between all pairwise predicted and
    GT camera-center distances. Pairwise distances make the comparison
    independent of the unrelated VGGT and Habitat world coordinate frames.
    """

    if not np.isfinite(height_scale_m_per_vggt) or height_scale_m_per_vggt <= 0:
        raise ValueError("height_scale_m_per_vggt must be positive and finite")
    predicted_centers = camera_centers_from_world_to_camera(
        predicted_camera_from_world_vggt
    )
    true_centers = np.asarray(true_camera_centers_world_m, dtype=np.float64)
    if true_centers.shape != predicted_centers.shape:
        raise ValueError("true and predicted camera-center sequences must align")
    if predicted_centers.shape[0] < 3 or not np.isfinite(true_centers).all():
        raise ValueError("at least three finite aligned camera poses are required")

    predicted_distances = _pairwise_distances(predicted_centers)
    true_distances = _pairwise_distances(true_centers)
    valid = true_distances >= float(minimum_true_baseline_m)
    valid &= predicted_distances > 1e-8
    if np.count_nonzero(valid) < 3:
        raise ValueError("trajectory has insufficient non-zero metric baseline")
    predicted_valid = predicted_distances[valid]
    true_valid = true_distances[valid]
    denominator = float(np.dot(predicted_valid, predicted_valid))
    if denominator <= 1e-12:
        raise ValueError("predicted trajectory has degenerate scale")
    oracle_scale = float(
        np.dot(predicted_valid, true_valid) / denominator
    )
    if not np.isfinite(oracle_scale) or oracle_scale <= 0:
        raise ValueError("oracle trajectory scale is invalid")

    predicted_steps = np.linalg.norm(np.diff(predicted_centers, axis=0), axis=1)
    true_steps = np.linalg.norm(np.diff(true_centers, axis=0), axis=1)
    true_path_length = float(true_steps.sum())
    if true_path_length < minimum_true_baseline_m:
        raise ValueError("true trajectory path length is too small")
    height_path_length = float(predicted_steps.sum() * height_scale_m_per_vggt)
    scale_absolute_error = abs(height_scale_m_per_vggt - oracle_scale)
    path_absolute_error = abs(height_path_length - true_path_length)

    return TrajectoryScaleMetrics(
        frame_count=int(predicted_centers.shape[0]),
        height_scale_m_per_vggt=float(height_scale_m_per_vggt),
        oracle_scale_m_per_vggt=oracle_scale,
        scale_absolute_error_m_per_vggt=scale_absolute_error,
        scale_relative_error=scale_absolute_error / oracle_scale,
        scale_log_absolute_error=abs(
            float(np.log(height_scale_m_per_vggt / oracle_scale))
        ),
        true_path_length_m=true_path_length,
        height_scaled_path_length_m=height_path_length,
        path_length_absolute_error_m=path_absolute_error,
        path_length_relative_error=path_absolute_error / true_path_length,
        height_scaled_ate_rmse_m=_fixed_scale_ate_rmse(
            predicted_centers, true_centers, height_scale_m_per_vggt
        ),
        oracle_scaled_ate_rmse_m=_fixed_scale_ate_rmse(
            predicted_centers, true_centers, oracle_scale
        ),
    )


def evaluate_scale_chain(
    *,
    height_scale_m_per_vggt: float,
    oracle_scale_m_per_vggt: float,
    model_scale_token_m_per_vggt: float,
    nominal_single_extent_m: float = 6.5,
) -> ScaleChainMetrics:
    """Evaluate the learned scale token and the verifier's scale ratio.

    The checkpoint was trained with ``lambda_m_per_vggt`` as a metric-scale
    target.  The interactive verifier currently divides the independently
    estimated camera-height scale by that token, treating the quotient as
    metres per nominal BEV unit.  Since the nominal BEV coordinates are
    defined in metres, the ideal quotient is one.
    """

    values = (
        height_scale_m_per_vggt,
        oracle_scale_m_per_vggt,
        model_scale_token_m_per_vggt,
        nominal_single_extent_m,
    )
    if not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError("all scale-chain values must be positive and finite")

    token_absolute_error = abs(
        model_scale_token_m_per_vggt - oracle_scale_m_per_vggt
    )
    ratio = height_scale_m_per_vggt / model_scale_token_m_per_vggt
    chained_extent = nominal_single_extent_m * ratio
    chained_extent_absolute_error = abs(chained_extent - nominal_single_extent_m)
    return ScaleChainMetrics(
        model_scale_token_m_per_vggt=float(model_scale_token_m_per_vggt),
        model_scale_token_absolute_error_m_per_vggt=float(token_absolute_error),
        model_scale_token_relative_error=float(
            token_absolute_error / oracle_scale_m_per_vggt
        ),
        height_to_token_ratio_m_per_bev=float(ratio),
        height_to_token_ratio_absolute_error=float(abs(ratio - 1.0)),
        chained_single_extent_m=float(chained_extent),
        chained_single_extent_absolute_error_m=float(
            chained_extent_absolute_error
        ),
        chained_single_extent_relative_error=float(
            chained_extent_absolute_error / nominal_single_extent_m
        ),
    )
