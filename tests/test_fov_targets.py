from __future__ import annotations

import numpy as np
import torch

from vggt_bev_method1.config import LabelValues
from vggt_bev_method1.data import (
    cap_complete_and_visible_to_fov,
    fov_union_mask,
)


def test_single_fov_includes_occluded_forward_cells_but_not_rear_cells() -> None:
    identity = np.eye(3, dtype=np.float64)[None]
    support = fov_union_mask(
        identity,
        target_frame=0,
        horizontal_fov_degrees=90.0,
        output_size=64,
        output_extent_m=6.5,
    )
    assert support[8, 32]
    assert not support[56, 32]
    assert not support[8, 0]

    labels = LabelValues()
    complete = torch.full((64, 64), labels.free, dtype=torch.uint8)
    complete[8, 32] = labels.occupied
    visible = torch.full_like(complete, labels.unknown)
    target, clipped_visible, effective = cap_complete_and_visible_to_fov(
        complete,
        visible,
        support,
        labels=labels,
    )
    assert target[8, 32] == labels.occupied
    assert clipped_visible[8, 32] == labels.unknown
    assert effective[8, 32]
    assert target[56, 32] == labels.unknown


def test_merged_fov_is_union_of_historical_poses_in_latest_frame() -> None:
    first = np.eye(3, dtype=np.float64)
    second = np.eye(3, dtype=np.float64)
    second[0, 2] = 2.0
    transforms = np.stack((first, second))
    latest_only = fov_union_mask(
        transforms[1:],
        target_frame=0,
        horizontal_fov_degrees=70.0,
        output_size=100,
        output_extent_m=10.0,
    )
    merged = fov_union_mask(
        transforms,
        target_frame=1,
        horizontal_fov_degrees=70.0,
        output_size=100,
        output_extent_m=10.0,
    )
    assert int(merged.sum()) > int(latest_only.sum())
    assert bool((merged & ~latest_only).any())
