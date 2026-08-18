from __future__ import annotations

import torch


def assert_runtime_model_inputs(model_inputs: dict) -> None:
    if set(model_inputs) != {"images"}:
        raise ValueError("new P1B runtime model input must contain RGB images only")


def method1_collate(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    frame_counts = {int(sample["images"].shape[0]) for sample in samples}
    if len(frame_counts) != 1:
        raise ValueError(
            "VGGT has no temporal padding mask; a batch may contain only "
            "equal-length histories"
        )
    tensor_keys = (
        "images",
        "scale_gt_depth_m",
        "scale_gt_valid_mask",
        "scale_gt_intrinsics",
        "single_fov_complete_target",
        "single_visible_target",
        "single_fov_support_target",
        "single_gt_valid_mask",
        "merged_fov_complete_target",
        "merged_visible_target",
        "merged_fov_support_target",
        "merged_gt_valid_mask",
    )
    output = {
        key: torch.stack([sample[key] for sample in samples])
        for key in tensor_keys
    }
    output["metadata"] = [sample["metadata"] for sample in samples]
    assert_runtime_model_inputs({"images": output["images"]})
    return output
