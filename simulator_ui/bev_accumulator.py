"""Ground-truth BEV fusion matching the VGGNAV training-data collector."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
from scipy import ndimage

UNKNOWN_VALUE = np.uint8(112)


def crop_corners(
    position: Sequence[float],
    forward: Sequence[float],
    right: Sequence[float],
    extent: float,
) -> np.ndarray:
    half = extent / 2.0
    return np.asarray(
        [
            [
                position[0] + right[0] * local_right + forward[0] * local_forward,
                position[2] + right[1] * local_right + forward[1] * local_forward,
            ]
            for local_right in (-half, half)
            for local_forward in (-half, half)
        ],
        dtype=np.float64,
    )


class BEVAccumulator:
    """Accumulate visibility and render static truth in the latest ego frame."""

    def __init__(self, extent: float, size: int) -> None:
        if extent <= 0 or size <= 1:
            raise ValueError("extent and size must be positive")
        self.extent = float(extent)
        self.size = int(size)
        self.meters_per_pixel = self.extent / self.size
        self.minimum_x: Optional[float] = None
        self.maximum_z: Optional[float] = None
        self.complete = np.full((1, 1), UNKNOWN_VALUE, dtype=np.uint8)
        self.complete_known = np.zeros((1, 1), dtype=bool)
        self.observed = np.zeros((1, 1), dtype=bool)
        self.all_crop_corners: list[np.ndarray] = []

    def _expand(self, corners: np.ndarray) -> None:
        mpp = self.meters_per_pixel
        requested_min_x = math.floor(float(np.min(corners[:, 0])) / mpp) * mpp
        requested_max_x = math.ceil(float(np.max(corners[:, 0])) / mpp) * mpp
        requested_min_z = math.floor(float(np.min(corners[:, 1])) / mpp) * mpp
        requested_max_z = math.ceil(float(np.max(corners[:, 1])) / mpp) * mpp
        if self.minimum_x is None or self.maximum_z is None:
            columns = int(round((requested_max_x - requested_min_x) / mpp)) + 1
            rows = int(round((requested_max_z - requested_min_z) / mpp)) + 1
            self.minimum_x = requested_min_x
            self.maximum_z = requested_max_z
            shape = (rows, columns)
            self.complete = np.full(shape, UNKNOWN_VALUE, dtype=np.uint8)
            self.complete_known = np.zeros(shape, dtype=bool)
            self.observed = np.zeros(shape, dtype=bool)
            return

        old_min_x = self.minimum_x
        old_max_z = self.maximum_z
        old_max_x = old_min_x + (self.complete.shape[1] - 1) * mpp
        old_min_z = old_max_z - (self.complete.shape[0] - 1) * mpp
        new_min_x = min(old_min_x, requested_min_x)
        new_max_x = max(old_max_x, requested_max_x)
        new_min_z = min(old_min_z, requested_min_z)
        new_max_z = max(old_max_z, requested_max_z)
        if (
            abs(new_min_x - old_min_x) < mpp * 0.1
            and abs(new_max_x - old_max_x) < mpp * 0.1
            and abs(new_min_z - old_min_z) < mpp * 0.1
            and abs(new_max_z - old_max_z) < mpp * 0.1
        ):
            return
        new_columns = int(round((new_max_x - new_min_x) / mpp)) + 1
        new_rows = int(round((new_max_z - new_min_z) / mpp)) + 1
        column_offset = int(round((old_min_x - new_min_x) / mpp))
        row_offset = int(round((new_max_z - old_max_z) / mpp))
        slices = (
            slice(row_offset, row_offset + self.complete.shape[0]),
            slice(column_offset, column_offset + self.complete.shape[1]),
        )
        for name, fill in (
            ("complete", UNKNOWN_VALUE),
            ("complete_known", False),
            ("observed", False),
        ):
            old = getattr(self, name)
            expanded = np.full((new_rows, new_columns), fill, dtype=old.dtype)
            expanded[slices] = old
            setattr(self, name, expanded)
        self.minimum_x = new_min_x
        self.maximum_z = new_max_z

    def _ego_to_world_transform(
        self,
        position: Sequence[float],
        forward: Sequence[float],
        right: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.minimum_x is None or self.maximum_z is None:
            raise RuntimeError("accumulator bounds are not initialized")
        center = (self.size - 1) / 2.0
        mpp = self.meters_per_pixel
        matrix = np.asarray(
            [[forward[1], -forward[0]], [-right[1], right[0]]],
            dtype=np.float64,
        )
        offset = np.asarray(
            [
                center
                - (self.minimum_x - position[0]) * forward[0] / mpp
                - (self.maximum_z - position[2]) * forward[1] / mpp,
                center
                + (self.minimum_x - position[0]) * right[0] / mpp
                + (self.maximum_z - position[2]) * right[1] / mpp,
            ],
            dtype=np.float64,
        )
        return matrix, offset

    def update(
        self,
        complete: np.ndarray,
        masked: np.ndarray,
        extrinsic: dict[str, Any],
    ) -> None:
        if complete.shape != (self.size, self.size) or masked.shape != complete.shape:
            raise ValueError("BEV inputs must match the configured accumulator size")
        position = extrinsic["agent_position_world_m"]
        forward = extrinsic["bev_forward_xz"]
        right = extrinsic["bev_right_xz"]
        corners = crop_corners(position, forward, right, self.extent)
        self._expand(corners)
        self.all_crop_corners.append(corners)
        matrix, offset = self._ego_to_world_transform(position, forward, right)
        output_shape = self.complete.shape

        complete_warped = ndimage.affine_transform(
            complete,
            matrix,
            offset,
            output_shape=output_shape,
            output=np.uint8,
            order=0,
            mode="constant",
            cval=int(UNKNOWN_VALUE),
            prefilter=False,
        )
        # P1B merged GT updates content only inside each frame's geometric
        # FOV-complete support. Iteration is chronological, so newer frames
        # overwrite older labels in overlap instead of confidence blending.
        source_known = (masked != UNKNOWN_VALUE).astype(np.uint8)
        complete_mask = ndimage.affine_transform(
            source_known,
            matrix,
            offset,
            output_shape=output_shape,
            output=np.uint8,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ).astype(bool)
        self.complete[complete_mask] = complete_warped[complete_mask]
        self.complete_known |= complete_mask

        self.observed |= complete_mask

    def render_masked(
        self,
        extrinsic: dict[str, Any],
        merged_extent: float,
    ) -> np.ndarray:
        if not self.all_crop_corners:
            raise RuntimeError("cannot render an empty accumulator")
        if self.minimum_x is None or self.maximum_z is None:
            raise RuntimeError("accumulator bounds are unavailable")
        position = extrinsic["agent_position_world_m"]
        forward = np.asarray(extrinsic["bev_forward_xz"], dtype=np.float64)
        right = np.asarray(extrinsic["bev_right_xz"], dtype=np.float64)
        output_mpp = merged_extent / self.size
        center = (self.size - 1) / 2.0
        scale = output_mpp / self.meters_per_pixel
        matrix = np.asarray(
            [
                [forward[1] * scale, -right[1] * scale],
                [-forward[0] * scale, right[0] * scale],
            ],
            dtype=np.float64,
        )
        offset = np.asarray(
            [
                (
                    self.maximum_z
                    - position[2]
                    - center * output_mpp * (forward[1] - right[1])
                )
                / self.meters_per_pixel,
                (
                    position[0]
                    - self.minimum_x
                    + center * output_mpp * (forward[0] - right[0])
                )
                / self.meters_per_pixel,
            ],
            dtype=np.float64,
        )
        rendered = ndimage.affine_transform(
            self.complete,
            matrix,
            offset,
            output_shape=(self.size, self.size),
            output=np.uint8,
            order=0,
            mode="constant",
            cval=int(UNKNOWN_VALUE),
            prefilter=False,
        )
        rendered_observed = ndimage.affine_transform(
            (self.observed & self.complete_known).astype(np.uint8),
            matrix,
            offset,
            output_shape=(self.size, self.size),
            output=np.uint8,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ).astype(bool)
        rendered[~rendered_observed] = UNKNOWN_VALUE
        return rendered
