from __future__ import annotations

import torch

from vggt_bev_method1.data import (
    RGBResizePad,
    VGGNAVMethod1Dataset,
    load_split_manifest,
    manifest_session_keys,
)


def p1d_collate(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("cannot collate an empty P1D batch")
    frame_counts = {int(sample["images"].shape[0]) for sample in samples}
    if len(frame_counts) != 1:
        raise ValueError("one P1D batch must contain equal-length RGB histories")
    tensor_keys = (
        "images",
        "scale_gt_depth_m",
        "scale_gt_valid_mask",
        "merged_fov_complete_target",
        "merged_visible_target",
        "merged_fov_support_target",
        "merged_gt_valid_mask",
        "latest_observed_free_target",
        "latest_fov_support_target",
    )
    output = {
        key: torch.stack([sample[key] for sample in samples])
        for key in tensor_keys
    }
    output["metadata"] = [sample["metadata"] for sample in samples]
    if any(key.startswith("single_") for key in output):
        raise RuntimeError("P1D collate exposed a Single prediction target")
    return output


def build_p1d_datasets(
    config: dict,
    *,
    verify_manifest: bool = True,
) -> tuple[VGGNAVMethod1Dataset, VGGNAVMethod1Dataset]:
    data = config["data"]
    manifest = load_split_manifest(
        data["split_manifest"],
        dataset_root=data["root"],
        validation_fraction=float(data["validation_fraction"]),
        seed=int(data["split_seed"]),
        maximum_sessions=(
            int(data["maximum_sessions"])
            if data.get("maximum_sessions") is not None
            else None
        ),
        selection_order=data.get("session_selection_order"),
        verify_metadata=(
            verify_manifest
            and bool(data.get("verify_manifest_metadata_at_startup", True))
        ),
        verify_artifacts=(
            verify_manifest
            and bool(data.get("verify_manifest_artifacts_at_startup", True))
        ),
    )
    train_keys, validation_keys = manifest_session_keys(manifest)
    preprocess = RGBResizePad(int(data["image_height"]), int(data["image_width"]))
    common = dict(
        root=data["root"],
        supervision=data["supervision"],
        preprocess=preprocess,
        sample_stride=int(data["sample_stride"]),
        minimum_history=int(data["minimum_history"]),
        maximum_history=int(data["maximum_history"]),
        void_coverage_index=data.get("void_coverage_index"),
        expected_manifest_sha256=manifest["content_sha256"],
        merged_source_extent_m=float(data["merged_source_extent_m"]),
        merged_bev_extent_m=float(data["merged_source_extent_m"]),
        merged_bev_output_size=int(data["merged_source_output_size"]),
        include_single_targets=False,
        include_latest_temporal_targets=True,
    )
    train = VGGNAVMethod1Dataset(session_keys=train_keys, **common)
    validation = VGGNAVMethod1Dataset(session_keys=validation_keys, **common)
    train.split_manifest_sha256 = manifest["content_sha256"]
    validation.split_manifest_sha256 = manifest["content_sha256"]
    overlap = train.scene_keys.intersection(validation.scene_keys)
    if overlap:
        raise RuntimeError(f"scene leakage detected: {sorted(overlap)[:10]}")
    return train, validation
