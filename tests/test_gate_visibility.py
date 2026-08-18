from __future__ import annotations

import numpy as np

from vggt_bev_method1.data.gate_visibility import (
    ConservativeVisibilityConfig,
    conservative_observed_gate_repair,
)


def _labels(size: int = 96) -> tuple[np.ndarray, np.ndarray]:
    complete = np.full((size, size), 255, dtype=np.uint8)
    masked = np.full((size, size), 112, dtype=np.uint8)
    masked[size // 2 - 25 : size // 2 + 1, 12 : size - 12] = 255
    return masked, complete


def _config() -> ConservativeVisibilityConfig:
    return ConservativeVisibilityConfig(angular_oversample=4)


def test_observed_island_is_removed() -> None:
    masked, complete = _labels()
    masked[8:12, 8:12] = 255

    result = conservative_observed_gate_repair(masked, complete, config=_config())

    assert result.disconnected_remove[8:12, 8:12].all()
    assert not result.repaired_origin_reachable[8:12, 8:12].any()


def test_unoccluded_broad_region_is_preserved() -> None:
    masked, complete = _labels()

    result = conservative_observed_gate_repair(masked, complete, config=_config())

    assert not result.repair_remove.any()
    assert np.array_equal(result.repaired_origin_reachable, masked == 255)


def test_partial_cell_fan_around_wall_corner_is_removed() -> None:
    size = 128
    complete = np.full((size, size), 255, dtype=np.uint8)
    masked = np.full((size, size), 112, dtype=np.uint8)
    origin = size // 2
    masked[origin - 20 : origin + 1, 20:108] = 255
    complete[origin - 22, 20:80] = 0
    ray_rows = np.rint(np.linspace(origin - 21, 10, 34)).astype(np.int64)
    ray_columns = np.rint(np.linspace(80, 100, 34)).astype(np.int64)
    masked[ray_rows, ray_columns] = 255
    ray = np.zeros_like(masked, dtype=bool)
    ray[ray_rows, ray_columns] = True

    result = conservative_observed_gate_repair(masked, complete, config=_config())

    assert (result.repair_remove & ray).sum() >= int(ray.sum() * 0.85)
    assert not result.repair_remove[origin - 15 : origin + 1].any()


def test_real_wide_opening_preserves_central_visibility() -> None:
    size = 128
    complete = np.full((size, size), 255, dtype=np.uint8)
    masked = np.full((size, size), 112, dtype=np.uint8)
    origin = size // 2
    complete[origin - 22, origin - 30 : origin + 31] = 0
    complete[origin - 22, origin - 5 : origin + 6] = 255
    masked[15 : origin + 1, origin - 4 : origin + 5] = 255
    masked[origin - 20 : origin + 1, origin - 30 : origin + 31] = 255

    result = conservative_observed_gate_repair(masked, complete, config=_config())

    assert result.repaired_origin_reachable[20 : origin - 24, origin].all()
