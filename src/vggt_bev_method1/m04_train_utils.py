from __future__ import annotations

import json
from pathlib import Path

import torch

from vggt_bev_method1.data import (
    RGBResizePad,
    VGGNAVMethod1Dataset,
    load_split_manifest,
    manifest_session_keys,
)


def m04_collate(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("cannot collate an empty M04 batch")
    frame_counts = {int(sample["images"].shape[0]) for sample in samples}
    if len(frame_counts) != 1:
        raise ValueError("one M04 batch must contain equal-length RGB histories")
    tensor_keys = (
        "images",
        "scale_gt_depth_m",
        "scale_gt_valid_mask",
        "merged_fov_complete_target",
        "merged_visible_target",
        "merged_fov_support_target",
        "merged_gt_valid_mask",
    )
    optional_tensor_keys = (
        "latest_fov_complete_target",
        "latest_visible_target",
        "latest_observed_free_target",
        "latest_fov_support_target",
        "latest_gt_valid_mask",
    )
    output = {
        key: torch.stack([sample[key] for sample in samples])
        for key in tensor_keys
    }
    for key in optional_tensor_keys:
        present = [key in sample for sample in samples]
        if any(present) and not all(present):
            raise ValueError(f"M04 optional target {key} is only partially present")
        if all(present):
            output[key] = torch.stack([sample[key] for sample in samples])
    output["metadata"] = [sample["metadata"] for sample in samples]
    if any(key.startswith("single_") for key in output):
        raise RuntimeError("M04 collate exposed a Single prediction target")
    return output


def build_m04_datasets(
    config: dict,
    *,
    verify_manifest: bool = True,
    include_latest_temporal_targets: bool = False,
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
    incomplete_policy = str(data.get("incomplete_session_policy", "error"))
    if incomplete_policy not in {"error", "skip"}:
        raise ValueError("data.incomplete_session_policy must be error or skip")
    if incomplete_policy == "skip":
        index_path = Path(
            str(data.get("incomplete_session_index", ""))
        ).expanduser().resolve()
        if not index_path.is_file():
            raise FileNotFoundError(
                f"incomplete-session index is missing: {index_path}"
            )
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if int(payload.get("format_version", 0)) != 1:
            raise ValueError("incomplete-session index format is unsupported")
        if payload.get("manifest_content_sha256") != manifest["content_sha256"]:
            raise ValueError("incomplete-session index manifest SHA-256 mismatch")
        if Path(payload.get("dataset_root", "")).resolve() != Path(
            data["root"]
        ).expanduser().resolve():
            raise ValueError("incomplete-session index dataset root mismatch")
        incomplete = set(payload.get("incomplete_session_keys", ()))
        train_keys = [key for key in train_keys if key not in incomplete]
        validation_keys = [key for key in validation_keys if key not in incomplete]
        if not train_keys or not validation_keys:
            raise ValueError(
                "incomplete-session filtering emptied a dataset split"
            )
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
        merged_source_image_size=int(data["merged_source_image_size"]),
        merged_complete_directory=str(data["merged_complete_directory"]),
        merged_masked_directory=str(data["merged_masked_directory"]),
        merged_bev_extent_m=float(data["merged_source_extent_m"]),
        merged_bev_output_size=int(data["merged_source_output_size"]),
        include_single_targets=False,
        include_latest_temporal_targets=include_latest_temporal_targets,
        missing_depth_policy=str(data.get("missing_depth_policy", "error")),
    )
    train = VGGNAVMethod1Dataset(session_keys=train_keys, **common)
    validation = VGGNAVMethod1Dataset(session_keys=validation_keys, **common)
    train.split_manifest_sha256 = manifest["content_sha256"]
    validation.split_manifest_sha256 = manifest["content_sha256"]
    train.incomplete_session_count = len(incomplete) if incomplete_policy == "skip" else 0
    validation.incomplete_session_count = (
        len(incomplete) if incomplete_policy == "skip" else 0
    )
    overlap = train.scene_keys.intersection(validation.scene_keys)
    if overlap:
        raise RuntimeError(f"M04 scene leakage detected: {sorted(overlap)[:10]}")
    return train, validation
