#!/usr/bin/env python3
"""Strictly load and collate every session in a TartanGround M05 gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vggt_bev_method1.data.dataset import VGGNAVMethod1Dataset
from vggt_bev_method1.data.manifest import (
    load_split_manifest,
    manifest_session_keys,
)
from vggt_bev_method1.m05_train_utils import m05_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=170901)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    payload = load_split_manifest(
        args.manifest,
        dataset_root=root,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        selection_order="lexicographic_session_key",
        verify_metadata=True,
        verify_artifacts=True,
    )
    train_keys, validation_keys = manifest_session_keys(payload)
    dataset = VGGNAVMethod1Dataset(
        root,
        session_keys=train_keys + validation_keys,
        minimum_history=10,
        maximum_history=10,
        merged_source_extent_m=10.0,
        merged_source_image_size=512,
        merged_complete_directory="merged_complete_10m",
        merged_masked_directory="merged_masked_10m",
        merged_bev_extent_m=10.0,
        merged_bev_output_size=512,
        include_single_targets=False,
        include_latest_temporal_targets=True,
    )
    required = (
        "images",
        "scale_gt_depth_m",
        "scale_gt_valid_mask",
        "scale_gt_intrinsics",
        "relative_pose_target",
        "merged_fov_complete_target",
        "merged_visible_target",
        "merged_fov_support_target",
        "merged_gt_valid_mask",
        "latest_fov_complete_target",
        "latest_visible_target",
        "latest_fov_support_target",
        "latest_gt_valid_mask",
    )
    depth_valid: list[float] = []
    latest_valid: list[float] = []
    merged_valid: list[float] = []
    pending: list[dict] = []
    collated_samples = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample["images"].shape != (10, 3, 384, 512):
            raise ValueError(f"unexpected RGB tensor at sample {index}")
        if sample["scale_gt_depth_m"].shape != (10, 384, 512):
            raise ValueError(f"unexpected depth tensor at sample {index}")
        for key in required:
            if not bool(torch.isfinite(sample[key]).all()):
                raise ValueError(f"non-finite tensor at sample {index}: {key}")
        depth_valid.append(float(sample["scale_gt_valid_mask"].float().mean()))
        latest_valid.append(float(sample["latest_gt_valid_mask"].float().mean()))
        merged_valid.append(float(sample["merged_gt_valid_mask"].float().mean()))
        pending.append(sample)
        if len(pending) == 4 or index + 1 == len(dataset):
            batch = m05_collate(pending)
            if batch["images"].shape[1:] != (10, 3, 384, 512):
                raise ValueError("M05 collate produced an unexpected RGB tensor")
            collated_samples += int(batch["images"].shape[0])
            pending.clear()

    print(
        json.dumps(
            {
                "status": "STRICT_M05_ALL_OK",
                "manifest_sha256": payload["content_sha256"],
                "sessions": len(dataset.sessions),
                "unique_scenes": len({record.scene_id for record in dataset.sessions}),
                "samples": len(dataset),
                "frames_loaded": len(dataset) * 10,
                "collated_samples": collated_samples,
                "depth_valid_fraction_min": min(depth_valid),
                "latest_gt_valid_fraction_min": min(latest_valid),
                "merged_gt_valid_fraction_min": min(merged_valid),
                "train_sessions": len(train_keys),
                "validation_sessions": len(validation_keys),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
