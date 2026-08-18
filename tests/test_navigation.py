from __future__ import annotations

import random

import numpy as np

from vggt_bev_method1.navigation import (
    astar_path,
    ego_start_cell,
    geometric_fov_mask,
    plan_on_prediction,
    safe_confidence_cost_multiplier,
    select_gt_target,
)


def test_astar_does_not_cut_diagonal_obstacle_corners() -> None:
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[1, 2] = True
    blocked[2, 1] = True
    path = astar_path(blocked, (2, 2), (0, 0))
    assert path is not None
    assert path[1] != (1, 1)


def test_gt_target_is_free_reachable_and_inside_true_fov() -> None:
    complete = np.full((16, 16), 255, dtype=np.uint8)
    complete[:, 3] = 0
    trial = select_gt_target(
        complete,
        horizontal_fov_degrees=90.0,
        rng=random.Random(5),
        planning_size=8,
        extent_m=6.5,
        robot_radius_m=0.0,
        minimum_target_distance_m=0.5,
    )
    assert not trial.gt_blocked[trial.goal]
    assert trial.geometric_fov[trial.goal]
    assert trial.gt_path[0] == trial.start
    assert trial.gt_path[-1] == trial.goal


def test_gt_target_selection_is_geometric_not_visible_mask_dependent() -> None:
    complete = np.full((16, 16), 255, dtype=np.uint8)
    first = select_gt_target(
        complete,
        horizontal_fov_degrees=75.0,
        rng=random.Random(11),
        planning_size=8,
        robot_radius_m=0.0,
        minimum_target_distance_m=0.5,
    )
    second = select_gt_target(
        complete,
        horizontal_fov_degrees=75.0,
        rng=random.Random(11),
        planning_size=8,
        robot_radius_m=0.0,
        minimum_target_distance_m=0.5,
    )
    assert first.goal == second.goal


def test_predicted_occupied_gt_free_target_is_model_failure() -> None:
    complete = np.full((16, 16), 255, dtype=np.uint8)
    trial = select_gt_target(
        complete,
        horizontal_fov_degrees=100.0,
        rng=random.Random(3),
        planning_size=8,
        robot_radius_m=0.0,
        minimum_target_distance_m=0.5,
    )
    occupancy = np.zeros((16, 16), dtype=np.float32)
    support = np.ones((16, 16), dtype=np.float32)
    confidence = np.ones((16, 16), dtype=np.float32)
    occupancy[trial.goal_native_pixel] = 0.9
    result = plan_on_prediction(
        occupancy,
        support,
        confidence,
        trial,
        planning_inflation_radius_m=0.0,
        known_ego_pose_clearance_radius_m=0.0,
    )
    assert not result.success
    assert result.failure_reason == "target_predicted_occupied"


def test_predicted_path_through_gt_obstacle_is_failure() -> None:
    complete = np.full((16, 16), 255, dtype=np.uint8)
    trial = select_gt_target(
        complete,
        horizontal_fov_degrees=120.0,
        rng=random.Random(1),
        planning_size=8,
        robot_radius_m=0.0,
        minimum_target_distance_m=1.0,
    )
    # Add a GT-only collision directly on a handcrafted replacement trial path
    # by marking one cell in the stored collision validation raster.
    gt_blocked = trial.gt_blocked.copy()
    midpoint = trial.gt_path[len(trial.gt_path) // 2]
    if midpoint not in (trial.start, trial.goal):
        gt_blocked[midpoint] = True
        object.__setattr__(trial, "gt_blocked", gt_blocked)
    occupancy = np.zeros((16, 16), dtype=np.float32)
    support = np.ones((16, 16), dtype=np.float32)
    confidence = np.ones((16, 16), dtype=np.float32)
    result = plan_on_prediction(
        occupancy,
        support,
        confidence,
        trial,
        planning_inflation_radius_m=0.0,
        known_ego_pose_clearance_radius_m=0.0,
    )
    if midpoint not in (trial.start, trial.goal):
        assert not result.success
        assert result.failure_reason == "predicted_path_collides_with_gt"


def test_unknown_behind_camera_is_not_dilated_over_robot_origin() -> None:
    complete = np.full((16, 16), 255, dtype=np.uint8)
    trial = select_gt_target(
        complete,
        horizontal_fov_degrees=90.0,
        rng=random.Random(7),
        planning_size=8,
        robot_radius_m=0.0,
        minimum_target_distance_m=0.5,
    )
    occupancy = np.zeros((16, 16), dtype=np.float32)
    support = np.zeros((16, 16), dtype=np.float32)
    confidence = np.ones((16, 16), dtype=np.float32)
    support[:10] = 1.0
    result = plan_on_prediction(
        occupancy,
        support,
        confidence,
        trial,
        planning_inflation_radius_m=1.0,
        known_ego_pose_clearance_radius_m=1.0,
    )
    assert not result.predicted_blocked[trial.start]


def test_narrow_supported_fov_apex_remains_connected_after_reduction() -> None:
    complete = np.full((16, 16), 255, dtype=np.uint8)
    trial = select_gt_target(
        complete,
        horizontal_fov_degrees=90.0,
        rng=random.Random(13),
        planning_size=8,
        robot_radius_m=0.0,
        minimum_target_distance_m=0.5,
    )
    occupancy = np.zeros((16, 16), dtype=np.float32)
    support = np.zeros((16, 16), dtype=np.float32)
    confidence = np.ones((16, 16), dtype=np.float32)
    # One supported source pixel in the first planning cell ahead of the ego,
    # followed by a wider supported corridor.
    support[7, 8] = 1.0
    support[:7, 6:11] = 1.0
    result = plan_on_prediction(
        occupancy,
        support,
        confidence,
        trial,
        planning_inflation_radius_m=0.0,
        known_ego_pose_clearance_radius_m=0.0,
    )
    # The exact random target may lie outside this synthetic corridor, but the
    # first forward cell must no longer be deleted by ALL-pixel pooling.
    assert not result.predicted_blocked[3, 4]


def test_geometric_fov_faces_image_up() -> None:
    mask = geometric_fov_mask(8, 6.5, 90.0)
    assert mask[1, 4]
    assert not mask[6, 4]


def test_even_512_start_is_forward_and_inside_true_fov() -> None:
    extent_m = 6.5
    mask = geometric_fov_mask(512, extent_m, 60.0)
    start = ego_start_cell(mask)
    cell = extent_m / 512
    x_m = -extent_m / 2.0 + (start[1] + 0.5) * cell
    z_m = extent_m / 2.0 - (start[0] + 0.5) * cell
    assert mask[start]
    assert z_m > 0.0
    assert abs(x_m) < 0.01
    assert z_m < 0.025


def test_native_512_gt_trial_has_a_real_gt_path() -> None:
    complete = np.full((512, 512), 255, dtype=np.uint8)
    trial = select_gt_target(
        complete,
        horizontal_fov_degrees=75.0,
        rng=random.Random(19),
        planning_size=512,
        extent_m=6.5,
        robot_radius_m=0.05,
        minimum_target_distance_m=0.75,
    )
    assert trial.geometric_fov[trial.start]
    assert trial.gt_path[0] == trial.start
    assert trial.gt_path[-1] == trial.goal
    assert not any(trial.gt_blocked[cell] for cell in trial.gt_path)


def test_known_ego_footprint_is_cleared_at_native_resolution() -> None:
    complete = np.full((512, 512), 255, dtype=np.uint8)
    trial = select_gt_target(
        complete,
        horizontal_fov_degrees=60.0,
        rng=random.Random(23),
        planning_size=512,
        extent_m=6.5,
        robot_radius_m=0.05,
        minimum_target_distance_m=0.75,
    )
    occupancy = np.zeros((512, 512), dtype=np.float32)
    support = np.ones((512, 512), dtype=np.float32)
    confidence = np.ones((512, 512), dtype=np.float32)
    occupancy[252:260, 252:260] = 1.0
    result = plan_on_prediction(
        occupancy,
        support,
        confidence,
        trial,
        planning_inflation_radius_m=0.10,
        known_ego_pose_clearance_radius_m=0.05,
    )
    assert not result.predicted_blocked[trial.start]
    # More than the one start pixel is cleared, preventing a one-pixel island.
    assert np.count_nonzero(~result.predicted_blocked[252:260, 252:260]) > 1


def test_confidence_weighted_astar_avoids_low_confidence_shortcut() -> None:
    blocked = np.zeros((7, 7), dtype=bool)
    multiplier = np.ones((7, 7), dtype=np.float64)
    multiplier[3, 2:5] = 20.0
    path = astar_path(blocked, (3, 1), (3, 5), multiplier)
    assert path is not None
    assert not any(row == 3 and 2 <= column <= 4 for row, column in path)


def test_safe_confidence_formula_penalizes_uncertain_free_cells() -> None:
    occupancy = np.full((2, 2), 0.1, dtype=np.float32)
    navigation_confidence = np.array(
        [[0.9, 0.2], [0.9, 0.2]], dtype=np.float32
    )
    safe, multiplier = safe_confidence_cost_multiplier(
        occupancy,
        navigation_confidence,
        weight=1.0,
    )
    assert safe[0, 0] > safe[0, 1]
    assert multiplier[0, 0] < multiplier[0, 1]
    assert np.all(multiplier >= 1.0)
