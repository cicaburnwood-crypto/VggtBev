from __future__ import annotations

import argparse
import json
from pathlib import Path

from vggt_bev_method1.config import load_config
from vggt_bev_method1.data import (
    RGBResizePad,
    VGGNAVMethod1Dataset,
    load_split_manifest,
    manifest_session_keys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate data_build_unlimited for Method I training"
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = config["data"]
    manifest = load_split_manifest(
        data["split_manifest"],
        dataset_root=data["root"],
        validation_fraction=float(data["validation_fraction"]),
        seed=int(data["split_seed"]),
    )
    train_keys, validation_keys = manifest_session_keys(manifest)
    preprocess = RGBResizePad(
        int(data["image_height"]),
        int(data["image_width"]),
    )
    common = {
        "root": data["root"],
        "supervision": data["supervision"],
        "preprocess": preprocess,
        "sample_stride": int(data["sample_stride"]),
        "minimum_history": int(data["minimum_history"]),
        "maximum_history": int(data["maximum_history"]),
    }
    train = VGGNAVMethod1Dataset(session_keys=train_keys, **common)
    validation = VGGNAVMethod1Dataset(session_keys=validation_keys, **common)
    overlap = train.scene_keys.intersection(validation.scene_keys)
    if overlap:
        raise RuntimeError(f"scene leakage detected: {sorted(overlap)[:10]}")
    first = train[0]
    last = train[-1]
    forbidden = {
        "depth",
        "intrinsics",
        "extrinsics",
        "trajectory",
        "camera_to_world",
        "world_to_camera",
        "single_extent_m",
        "merged_extent_m",
    }
    leaked = forbidden.intersection(first)
    if leaked:
        raise RuntimeError(f"GT geometry leaked into a dataset sample: {sorted(leaked)}")
    print(
        json.dumps(
            {
                "dataset_root": str(Path(data["root"]).resolve()),
                "supervision": data["supervision"],
                "train_sessions": len(train.sessions),
                "validation_sessions": len(validation.sessions),
                "train_scenes": len(train.scene_keys),
                "validation_scenes": len(validation.scene_keys),
                "train_samples": len(train),
                "validation_samples": len(validation),
                "scene_overlap": 0,
                "first_history": int(first["images"].shape[0]),
                "last_history": int(last["images"].shape[0]),
                "image_shape": list(first["images"].shape[1:]),
                "target_shapes": {
                    key: list(value.shape)
                    for key, value in first.items()
                    if key.endswith("_target")
                },
                "pipeline": "P1A",
                "coordinate_mode": (
                    "camera_height_anchored_fixed_normalized_scale"
                ),
                "extent_mode": "fixed_6p5_single_10_merged",
                "single_output_size": 512,
                "merged_output_size": 800,
                "single_output_extent_normalized_scale": 6.5,
                "merged_output_extent_normalized_scale": 10.0,
                "single_target_extent_m": float(
                    first["single_target_extent_m"]
                ),
                "merged_target_extent_m": float(
                    first["merged_target_extent_m"]
                ),
                "model_batch_camera_height": True,
                "model_forward_consumes_target_extents": False,
                "model_batch_gt_depth_pose_trajectory": False,
                "training_only_scale_validation_trajectory": True,
                "training_target_resampling": False,
                "training_geometry_source": (
                    "live VGGT-Omega estimates anchored by camera height"
                ),
                "split_manifest": str(
                    Path(data["split_manifest"]).expanduser().resolve()
                ),
                "split_manifest_sha256": manifest["content_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
