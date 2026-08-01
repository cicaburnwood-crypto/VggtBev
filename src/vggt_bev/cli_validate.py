from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from vggt_bev.config import LabelValues
from vggt_bev.data import CalibrationAwareResize, VGGNAVMethod2Dataset
from vggt_bev.geometry.frames import opencv_points_to_reference_bev


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate VGGNAV-to-Method-II alignment")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/user/Project/VGGNAV/output/random_sessions_test_3"),
    )
    parser.add_argument(
        "--extent-key",
        default="bev_5m",
        help="dataset extent directory, for example bev_5m or bev_6p5m",
    )
    parser.add_argument("--target-mode", choices=("single", "merged"), default="single")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--required-merged-fusion-version", type=int, default=None)
    return parser


def validate(args: argparse.Namespace) -> dict:
    dataset = VGGNAVMethod2Dataset(
        args.data_root,
        extent_key=args.extent_key,
        target_mode=args.target_mode,
        preprocess=CalibrationAwareResize(args.image_height, args.image_width),
        include_revealed_for_audit=True,
        required_merged_fusion_version=args.required_merged_fusion_version,
    )
    labels = LabelValues()
    value_counts: Counter[int] = Counter()
    observed_pixels = 0
    revealed_mismatches = 0
    maximum_forward_axis_error_m = 0.0
    history_mismatches = 0
    sample_count = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)

    for index in range(sample_count):
        sample = dataset[index]
        target = sample["target_labels"]
        revealed = sample["revealed_labels_audit_only"]
        unique, counts = torch.unique(target, return_counts=True)
        value_counts.update(
            {int(value): int(count) for value, count in zip(unique, counts, strict=True)}
        )
        observed = target != labels.unknown
        observed_pixels += int(observed.sum())
        revealed_mismatches += int(((target != revealed) & observed).sum())
        metadata = sample["metadata"]
        history_mismatches += int(
            metadata["history_frame_count"] != metadata["declared_history_frame_count"]
        )

        # A point one metre along OpenCV +z must become current-ego (right=0, forward=1).
        current_camera = sample["camera_to_world"][-1:][None]
        point = torch.tensor([[[[0.0, 0.0, 1.0]]]], dtype=torch.float32)
        point_bev = opencv_points_to_reference_bev(
            point,
            current_camera,
            sample["reference_world_from_bev"][None],
            sample["floor_y"][None],
        )[0, 0, 0, :2]
        axis_error = torch.linalg.vector_norm(point_bev - torch.tensor([0.0, 1.0]))
        maximum_forward_axis_error_m = max(maximum_forward_axis_error_m, float(axis_error))

    allowed = {labels.occupied, labels.unknown, labels.free}
    unexpected = sorted(set(value_counts) - allowed)
    errors: list[str] = []
    warnings: list[str] = []
    if unexpected:
        errors.append(f"unexpected label values: {unexpected}")
    if revealed_mismatches and args.target_mode == "single":
        errors.append(
            f"{revealed_mismatches} observed target pixels disagree with omniscient audit labels"
        )
    elif revealed_mismatches and args.required_merged_fusion_version == 2:
        errors.append(
            "visibility-v2 merged labels must agree with complete truth on every "
            f"known pixel; found {revealed_mismatches} mismatches"
        )
    elif revealed_mismatches:
        warnings.append(
            "merged_masked and merged_complete have different chronological overwrite histories; "
            f"{revealed_mismatches} known pixels differ and complete remains audit-only"
        )
    if history_mismatches:
        errors.append(f"{history_mismatches} cumulative-history counts are misaligned")
    if maximum_forward_axis_error_m > 1e-4:
        errors.append(
            f"Habitat/OpenCV forward-axis error is {maximum_forward_axis_error_m:.6g} m"
        )

    return {
        "passed": not errors,
        "data_root": str(dataset.root),
        "target_mode": args.target_mode,
        "extent_key": args.extent_key,
        "required_merged_fusion_version": args.required_merged_fusion_version,
        "session_count": len(dataset.session_names),
        "session_examples": list(dataset.session_names[:3]),
        "samples_checked": sample_count,
        "label_value_counts": dict(sorted(value_counts.items())),
        "observed_pixels": observed_pixels,
        "observed_vs_revealed_mismatches": revealed_mismatches,
        "observed_vs_revealed_mismatch_rate": revealed_mismatches / max(observed_pixels, 1),
        "history_count_mismatches": history_mismatches,
        "maximum_forward_axis_error_m": maximum_forward_axis_error_m,
        "warnings": warnings,
        "errors": errors,
    }


def main() -> None:
    result = validate(build_parser().parse_args())
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
