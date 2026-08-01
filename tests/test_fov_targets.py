from __future__ import annotations

import numpy as np
import torch

from vggt_bev_method1.config import LabelValues
from vggt_bev_method1.data.fov_targets import (
    cap_complete_and_visible_to_fov,
    fov_union_mask,
)


def test_single_uses_latest_fov_and_merged_uses_history_union() -> None:
    world_from_bev = np.stack(
        (
            np.eye(3),
            np.asarray(
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )
    )
    latest = fov_union_mask(
        world_from_bev[1:],
        target_frame=0,
        horizontal_fov_degrees=70.0,
        output_size=128,
        output_extent_m=10.0,
    )
    merged = fov_union_mask(
        world_from_bev,
        target_frame=1,
        horizontal_fov_degrees=70.0,
        output_size=128,
        output_extent_m=10.0,
    )
    assert torch.all(latest <= merged)
    assert int(merged.sum()) > int(latest.sum())


def test_fov_complete_keeps_occluded_truth_but_masks_outside() -> None:
    labels = LabelValues()
    complete = torch.full((4, 4), labels.free, dtype=torch.uint8)
    complete[1, 1] = labels.occupied
    visible = torch.full((4, 4), labels.unknown, dtype=torch.uint8)
    visible[2, 1] = labels.free
    support = torch.zeros(4, 4, dtype=torch.bool)
    support[1:3, 1:3] = True
    fov_complete, fov_visible, effective = cap_complete_and_visible_to_fov(
        complete,
        visible,
        support,
        labels=labels,
    )
    assert fov_complete[1, 1] == labels.occupied
    assert fov_visible[1, 1] == labels.unknown
    assert fov_complete[0, 0] == labels.unknown
    assert torch.equal(effective, support)
