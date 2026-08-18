from __future__ import annotations

import inspect

import numpy as np
import pytest

from vggt_bev_method1.interactive_verifier import (
    MetricRaster,
    VerifierPlanningError,
    astar_no_inflation,
    compile_grid_path_to_open_loop_actions,
    inflate_occupied_mask,
    local_target_from_normalized_click,
    plan_metric_target,
    relative_camera_motion_metric,
    transform_target_by_predicted_motion,
)


def test_local_click_conversion_has_no_gt_raster_input() -> None:
    assert "gt_complete" not in inspect.signature(
        local_target_from_normalized_click
    ).parameters
    predicted = MetricRaster(size=12, extent_m=6.0)
    target = local_target_from_normalized_click(
        local_extent_m=6.0,
        click_u=0.75,
        click_v=0.25,
    )
    assert target["target_metric_m"] == pytest.approx([1.5, 1.5])
    assert predicted.metric_to_pixel(*target["target_metric_m"]) == (3, 9)
    assert target["coordinate_frame"] == "current ego: x=right, z=forward"


def test_astar_uses_one_cell_corridor_without_inflation() -> None:
    blocked = np.ones((9, 9), dtype=bool)
    blocked[4, 1:8] = False
    path = astar_no_inflation(blocked, (4, 1), (4, 7))
    assert path is not None
    assert all(row == 4 for row, _ in path)


def test_predicted_occupied_target_is_explicit_failure() -> None:
    predicted = np.full((16, 16), 255, dtype=np.uint8)
    predicted[4, 12] = 0
    with pytest.raises(VerifierPlanningError) as caught:
        plan_metric_target(
            predicted_semantic=predicted,
            predicted_extent_m=8.0,
            target_metric_m=[2.25, 1.75],
            frame_seq=7,
            model_key="p1b",
            start_anchor_max_m=1.0,
        )
    assert caught.value.code == "target_predicted_occupied"


def test_success_path_contains_no_gt_or_simulator_coordinates() -> None:
    predicted = np.full((16, 16), 255, dtype=np.uint8)
    plan = plan_metric_target(
        predicted_semantic=predicted,
        predicted_extent_m=8.0,
        target_metric_m=[2.0, 2.0],
        frame_seq=11,
        model_key="p1b",
        start_anchor_max_m=1.0,
    )
    assert plan["success"]
    assert plan["inflation_radius_m"] == 0.025
    assert plan["metric_alignment"]["mapping"] == (
        "ego-local target coordinate -> predicted BEV pixel"
    )
    assert "path_world_xz_m" not in plan
    assert "path_vggt_units" not in plan
    assert "gt_target_pixel" not in plan
    assert "simulator_pose" in plan["planner_forbidden_inputs"]


def test_unknown_is_blocked_and_explains_disconnected_path() -> None:
    predicted = np.full((16, 16), 255, dtype=np.uint8)
    predicted[7, :] = 112
    with pytest.raises(VerifierPlanningError) as caught:
        plan_metric_target(
            predicted_semantic=predicted,
            predicted_extent_m=8.0,
            target_metric_m=[2.0, 2.5],
            frame_seq=12,
            model_key="p1b",
            start_anchor_max_m=1.0,
        )
    assert caught.value.code == "no_predicted_astar_path"


def test_vggt_relative_translation_is_scaled_before_target_update() -> None:
    previous_camera_from_world = np.concatenate(
        [np.eye(3), np.zeros((3, 1))], axis=1
    )
    # The camera moved +1 VGGT unit along its forward/world-z direction, so a
    # fixed point is one native unit closer in the current camera frame.
    current_camera_from_world = np.concatenate(
        [np.eye(3), np.asarray([[0.0], [0.0], [-1.0]])], axis=1
    )
    current_from_previous = relative_camera_motion_metric(
        previous_camera_from_world,
        current_camera_from_world,
        lambda_m_per_vggt=2.0,
    )
    assert current_from_previous[:3, 3] == pytest.approx([0.0, 0.0, -2.0])
    assert transform_target_by_predicted_motion(
        [0.0, 3.0], current_from_previous
    ) == pytest.approx([0.0, 1.0])


def test_closed_loop_metric_target_uses_one_sided_inflation() -> None:
    predicted = np.full((16, 16), 255, dtype=np.uint8)
    # A five-cell opening becomes one cell wide after two-cell expansion from
    # both obstacle ends.  The requested 2.5 cm is a one-sided radius.
    predicted[7, :] = 0
    predicted[7, 6:11] = 255
    plan = plan_metric_target(
        predicted_semantic=predicted,
        predicted_extent_m=0.208,
        target_metric_m=[0.0065, 0.0585],
        frame_seq=21,
        model_key="p1b",
        start_anchor_max_m=0.05,
    )
    assert plan["success"]
    assert plan["inflation_radius_m"] == 0.025
    assert plan["inflation_radius_cells"] == 2
    assert plan["effective_inflation_radius_m"] == pytest.approx(0.026)
    assert plan["metric_alignment"]["mapping"] == (
        "ego-local target coordinate -> predicted BEV pixel"
    )
    assert [7, 8] in plan["path_pixels"]


def test_25mm_inflation_rounds_up_to_two_native_512_cells() -> None:
    occupied = np.zeros((512, 512), dtype=bool)
    occupied[256, 256] = True
    inflated, radius_cells, effective_radius_m = inflate_occupied_mask(
        occupied,
        cell_size_m=6.5 / 512,
        radius_m=0.025,
    )
    assert radius_cells == 2
    assert effective_radius_m == pytest.approx(0.025390625)
    assert inflated[256, 258]
    assert inflated[258, 256]
    assert not inflated[258, 258]


def test_planner_accepts_runtime_one_sided_inflation_radius() -> None:
    predicted = np.full((512, 512), 255, dtype=np.uint8)
    plan = plan_metric_target(
        predicted_semantic=predicted,
        predicted_extent_m=6.5,
        target_metric_m=[0.0, 0.25],
        frame_seq=24,
        model_key="p1b",
        inflation_radius_m=0.04,
    )
    assert plan["inflation_radius_m"] == pytest.approx(0.04)
    assert plan["inflation_radius_cells"] == 4
    assert plan["effective_inflation_radius_m"] == pytest.approx(0.05078125)


def test_predicted_free_target_inside_inflation_is_explicit_failure() -> None:
    predicted = np.full((512, 512), 255, dtype=np.uint8)
    predicted[230, 258] = 0
    raster = MetricRaster(size=512, extent_m=6.5)
    with pytest.raises(VerifierPlanningError) as caught:
        plan_metric_target(
            predicted_semantic=predicted,
            predicted_extent_m=6.5,
            target_metric_m=raster.pixel_to_metric(230, 256),
            frame_seq=23,
            model_key="p1b",
        )
    assert caught.value.code == "target_inside_inflation_margin"


def test_display_click_and_predicted_only_planning_are_separate() -> None:
    forbidden_parameters = {
        "gt_complete",
        "gt_extent_m",
        "world_from_bev_planar",
        "simulator_pose",
        "navmesh",
    }
    assert forbidden_parameters.isdisjoint(
        inspect.signature(plan_metric_target).parameters
    )
    target = local_target_from_normalized_click(
        local_extent_m=8.0,
        click_u=0.75,
        click_v=0.25,
    )
    predicted = np.full((16, 16), 255, dtype=np.uint8)
    plan = plan_metric_target(
        predicted_semantic=predicted,
        predicted_extent_m=8.0,
        target_metric_m=target["target_metric_m"],
        frame_seq=22,
        model_key="p1b",
        start_anchor_max_m=1.0,
    )
    assert plan["success"]
    assert "path_world_xz_m" not in plan
    assert "execution_path_world_xz_m" not in plan
    assert plan["path_pixels"][0] == plan["predicted_start_pixel"]


def test_strict_open_loop_compiler_emits_exact_grid_actions() -> None:
    actions = compile_grid_path_to_open_loop_actions(
        [(5, 5), (4, 5), (3, 6), (3, 7)]
    )
    assert actions == [
        "strict_forward_cardinal",
        "strict_turn_right_45",
        "strict_forward_diagonal",
        "strict_turn_right_45",
        "strict_forward_cardinal",
    ]


def test_strict_open_loop_compiler_rejects_non_adjacent_steps() -> None:
    with pytest.raises(ValueError, match="non-adjacent"):
        compile_grid_path_to_open_loop_actions([(5, 5), (3, 5)])
