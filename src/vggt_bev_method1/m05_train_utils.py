from __future__ import annotations

from vggt_bev_method1.m04_train_utils import build_m04_datasets, m04_collate


M05_REQUIRED_LATEST_TARGETS = (
    "latest_fov_complete_target",
    "latest_visible_target",
    "latest_fov_support_target",
    "latest_gt_valid_mask",
)


def m05_collate(samples: list[dict]) -> dict:
    output = m04_collate(samples)
    missing = [key for key in M05_REQUIRED_LATEST_TARGETS if key not in output]
    if missing:
        raise KeyError(f"M05 batch is missing latest-frame targets: {missing}")
    return output


def build_m05_datasets(config: dict, *, verify_manifest: bool = True):
    datasets = build_m04_datasets(
        config,
        verify_manifest=verify_manifest,
        include_latest_temporal_targets=True,
    )
    return datasets
