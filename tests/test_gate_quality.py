from __future__ import annotations

import numpy as np

from vggt_bev_method1.data.gate_quality import (
    GateLeakRepairConfig,
    repair_observed_gate_leaks,
)


def _base(size: int = 96) -> np.ndarray:
    labels = np.full((size, size), 112, dtype=np.uint8)
    labels[8 : size // 2 + 1, 8 : size - 8] = 255
    return labels


def _config() -> GateLeakRepairConfig:
    return GateLeakRepairConfig(
        maximum_surface_gap_pixels=1,
        minimum_lobe_pixels=16,
        angular_bins=1024,
        line_of_sight_margin_pixels=0.5,
        origin_seed_radius_pixels=5.0,
    )


def test_closed_one_pixel_wall_leak_is_repaired() -> None:
    labels = _base()
    wall_row = 30
    labels[wall_row, 8:88] = 0
    labels[wall_row, 48] = 255

    result = repair_observed_gate_leaks(labels, config=_config())

    assert result.inserted_surface[wall_row, 48]
    assert result.repair_remove[:wall_row].sum() > 500
    assert not result.repair_remove[wall_row + 5 :, :].any()
    assert result.accepted_component_count == 1


def test_real_opening_wider_than_repair_limit_is_preserved() -> None:
    labels = _base()
    wall_row = 30
    labels[wall_row, 8:88] = 0
    labels[wall_row, 47:50] = 255

    result = repair_observed_gate_leaks(labels, config=_config())

    assert not result.inserted_surface.any()
    assert not result.repair_remove.any()
    assert not result.ambiguous_ignore.any()


def test_finite_obstacle_does_not_create_a_repaired_lobe() -> None:
    labels = _base()
    labels[30, 28:68] = 0
    labels[30, 48] = 255

    result = repair_observed_gate_leaks(labels, config=_config())

    assert result.inserted_surface[30, 48]
    assert not result.repair_remove.any()


def test_boundary_depth_disagreement_without_surface_gap_is_preserved() -> None:
    labels = _base()
    labels[22:34, 20:30] = 0

    result = repair_observed_gate_leaks(labels, config=_config())

    assert not result.inserted_surface.any()
    assert not result.repair_remove.any()
    assert not result.ambiguous_ignore.any()


def test_long_thin_observed_ray_is_removed_as_one_branch() -> None:
    labels = np.full((128, 128), 112, dtype=np.uint8)
    labels[50:112, 20:108] = 255
    labels[5:51, 63:65] = 255

    result = repair_observed_gate_leaks(labels, config=_config())

    assert result.thin_branch_remove[8:45, 63:65].all()
    assert result.accepted_thin_branch_count >= 1


def test_broad_observed_extension_is_not_shaved() -> None:
    labels = np.full((128, 128), 112, dtype=np.uint8)
    labels[50:112, 20:108] = 255
    labels[5:51, 55:73] = 255

    result = repair_observed_gate_leaks(labels, config=_config())

    assert not result.thin_branch_remove.any()
