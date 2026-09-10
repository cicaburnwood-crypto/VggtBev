from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .coverage import VoidCoverageIndex

OCCUPIED = 0
UNKNOWN = 112
FREE = 255
ALLOWED_LABELS = {OCCUPIED, UNKNOWN, FREE}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    branch: str
    semantic_scope: str
    directory: str
    output_size: int
    extent_m: float


TARGETS = (
    TargetSpec("single_complete", "single", "complete", "complete", 512, 6.5),
    TargetSpec("single_observed", "single", "observed", "masked", 512, 6.5),
    TargetSpec(
        "merged_complete",
        "merged",
        "complete",
        "merged_complete_10m",
        800,
        10.0,
    ),
    TargetSpec(
        "merged_observed",
        "merged",
        "observed",
        "merged_masked_10m",
        800,
        10.0,
    ),
)


@dataclass(frozen=True)
class VoidMeasurement:
    grid_pixels: int
    void_pixels: int
    void_fraction_of_grid: float
    gt_known_pixels: int
    void_inside_gt_known_pixels: int
    void_fraction_of_gt_known: float


def measure_gt_bev(
    void_mask: np.ndarray,
    semantic_target: np.ndarray,
) -> VoidMeasurement:
    """Measure Void without changing or filtering the semantic target."""

    void = np.asarray(void_mask, dtype=bool)
    target = np.asarray(semantic_target, dtype=np.uint8)
    if void.shape != target.shape or void.ndim != 2:
        raise ValueError("Void and GT BEV must be aligned 2-D rasters")
    values = {int(value) for value in np.unique(target)}
    if not values <= ALLOWED_LABELS:
        raise ValueError(f"GT BEV contains invalid values: {sorted(values)}")
    known = target != UNKNOWN
    grid_pixels = int(void.size)
    void_pixels = int(void.sum())
    known_pixels = int(known.sum())
    void_known_pixels = int((void & known).sum())
    return VoidMeasurement(
        grid_pixels=grid_pixels,
        void_pixels=void_pixels,
        void_fraction_of_grid=(void_pixels / grid_pixels if grid_pixels else 0.0),
        gt_known_pixels=known_pixels,
        void_inside_gt_known_pixels=void_known_pixels,
        void_fraction_of_gt_known=(
            void_known_pixels / known_pixels if known_pixels else 0.0
        ),
    )


def _load_world_from_bev_planar(
    records: Sequence[dict],
    *,
    expected_frames: int,
) -> np.ndarray:
    matrices: list[np.ndarray] = []
    for expected_frame, record in enumerate(records):
        if int(record.get("frame_id", -1)) != expected_frame:
            raise ValueError("camera extrinsics are not in contiguous frame order")
        extrinsic = record.get("extrinsic", record)
        matrix = np.asarray(extrinsic.get("world_from_bev_planar"), dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("world_from_bev_planar is missing or invalid")
        if abs(float(np.linalg.det(matrix))) < 1e-8:
            raise ValueError("world_from_bev_planar is singular")
        matrices.append(matrix)
    if len(matrices) != expected_frames:
        raise ValueError("camera extrinsic count does not match session frame_count")
    return np.stack(matrices)


def _load_gt(path: Path, output_size: int) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (512, 512):
            raise ValueError(f"unexpected GT BEV raster: {path}")
        if output_size != 512:
            image = image.resize(
                (output_size, output_size),
                Image.Resampling.NEAREST,
            )
        target = np.asarray(image, dtype=np.uint8).copy()
    values = {int(value) for value in np.unique(target)}
    if not values <= ALLOWED_LABELS:
        raise ValueError(f"GT BEV contains invalid values at {path}: {sorted(values)}")
    return target


def _manifest_rows(payload: dict, split: str) -> list[tuple[str, dict]]:
    splits = ("train", "validation") if split == "all" else (split,)
    rows = [(name, dict(row)) for name in splits for row in payload.get(name, ())]
    if not rows:
        raise ValueError("manifest selection contains no sessions")
    keys = [str(row["key"]) for _, row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("manifest selection contains duplicate session keys")
    return rows


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _update_summary(groups: dict[str, dict[str, float]], row: dict) -> None:
    group = groups.setdefault(
        str(row["gt_bev"]),
        {
            "records": 0,
            "grid_pixels": 0,
            "void_pixels": 0,
            "gt_known_pixels": 0,
            "void_inside_gt_known_pixels": 0,
            "sum_void_fraction_of_grid": 0.0,
            "minimum_void_fraction_of_grid": 1.0,
            "maximum_void_fraction_of_grid": 0.0,
        },
    )
    group["records"] += 1
    for name in (
        "grid_pixels",
        "void_pixels",
        "gt_known_pixels",
        "void_inside_gt_known_pixels",
    ):
        group[name] += int(row[name])
    fraction = float(row["void_fraction_of_grid"])
    group["sum_void_fraction_of_grid"] += fraction
    group["minimum_void_fraction_of_grid"] = min(
        group["minimum_void_fraction_of_grid"], fraction
    )
    group["maximum_void_fraction_of_grid"] = max(
        group["maximum_void_fraction_of_grid"], fraction
    )


def _finalize_summary(groups: dict[str, dict[str, float]]) -> dict:
    total_records = sum(int(group["records"]) for group in groups.values())
    for group in groups.values():
        records = int(group["records"])
        grid = int(group["grid_pixels"])
        known = int(group["gt_known_pixels"])
        group["mean_void_fraction_of_grid"] = (
            float(group.pop("sum_void_fraction_of_grid")) / records
        )
        group["weighted_void_fraction_of_grid"] = (
            int(group["void_pixels"]) / grid if grid else 0.0
        )
        group["weighted_void_fraction_of_gt_known"] = (
            int(group["void_inside_gt_known_pixels"]) / known if known else 0.0
        )
    return {"record_count": total_records, "by_gt_bev": groups}


def audit_manifest(
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
    void_index_path: str | Path,
    output_dir: str | Path,
    split: str = "all",
    frame_mode: str = "final",
    maximum_history: int = 10,
    maximum_sessions: int | None = None,
    verify_artifacts: bool = True,
    log_every_sessions: int = 100,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict:
    """Write per-GT-BEV Void fractions as CSV plus an aggregate JSON report."""

    if split not in {"all", "train", "validation"}:
        raise ValueError("split must be all, train, or validation")
    if frame_mode not in {"final", "all"}:
        raise ValueError("frame_mode must be final or all")
    if maximum_history <= 0:
        raise ValueError("maximum_history must be positive")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    root = Path(dataset_root).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if Path(manifest["dataset_root"]).resolve() != root:
        raise ValueError("manifest dataset_root does not match --dataset-root")
    sessions = _manifest_rows(manifest, split)
    full_session_count = len(sessions)
    sessions = [
        row for index, row in enumerate(sessions) if index % shard_count == shard_index
    ]
    if maximum_sessions is not None:
        if maximum_sessions <= 0:
            raise ValueError("maximum_sessions must be positive")
        sessions = sessions[:maximum_sessions]
    coverage = VoidCoverageIndex(
        void_index_path,
        dataset_root=root,
        expected_manifest_sha256=str(manifest["content_sha256"]),
        require_complete=True,
        verify_artifacts=verify_artifacts,
    )
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "void_fraction_per_gt_bev.csv"
    temporary_csv = destination / f".{csv_path.name}.tmp-{os.getpid()}"
    fieldnames = (
        "session_key",
        "scene_key",
        "dataset",
        "split",
        "target_frame",
        "gt_bev",
        "branch",
        "semantic_scope",
        "source_path",
        "output_height",
        "output_width",
        "grid_pixels",
        "void_pixels",
        "void_fraction_of_grid",
        "gt_known_pixels",
        "void_inside_gt_known_pixels",
        "void_fraction_of_gt_known",
    )
    summary_groups: dict[str, dict[str, float]] = {}
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for session_number, (session_split, manifest_row) in enumerate(
            sessions, start=1
        ):
            session_key = str(manifest_row["key"])
            session = root / session_key
            metadata = json.loads(
                (session / "metadata.json").read_text(encoding="utf-8")
            )
            scene_key = f"{metadata['dataset']}:{metadata['scene_id']}"
            if scene_key != str(manifest_row["scene_key"]):
                raise ValueError(f"manifest/metadata scene mismatch: {session_key}")
            frame_count = int(metadata["frame_count"])
            usable_frames = min(frame_count, maximum_history)
            if usable_frames <= 0:
                raise ValueError(f"session contains no usable frames: {session_key}")
            frame_ids = (
                (usable_frames - 1,)
                if frame_mode == "final"
                else tuple(range(usable_frames))
            )
            extrinsic_path = session / str(
                metadata.get("camera_extrinsics_file", "camera_extrinsics.jsonl")
            )
            extrinsic_records = [
                json.loads(line)
                for line in extrinsic_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            transforms = _load_world_from_bev_planar(
                extrinsic_records,
                expected_frames=frame_count,
            )
            if str(metadata.get("dataset", "")).startswith("procthor"):
                floor_height_m = float(
                    metadata.get("voxel_statistics", {}).get(
                        "floor_y_world_m", float("nan")
                    )
                )
            else:
                path_start = metadata.get("path", {}).get("start")
                if not isinstance(path_start, list) or len(path_start) < 2:
                    raise ValueError(
                        f"session has no GT floor height: {session_key}"
                    )
                floor_height_m = float(path_start[1])
            if not np.isfinite(floor_height_m):
                raise ValueError(f"session GT floor height is invalid: {session_key}")
            valid_by_branch: dict[tuple[int, str], np.ndarray] = {}
            for frame_id in frame_ids:
                for branch, output_size, extent_m in (
                    ("single", 512, 6.5),
                    ("merged", 800, 10.0),
                ):
                    valid = coverage.render_valid_mask(
                        scene_key=scene_key,
                        floor_height_m=floor_height_m,
                        world_from_bev_planar=transforms[frame_id],
                        output_size=output_size,
                        output_extent_m=extent_m,
                    )
                    valid_by_branch[(frame_id, branch)] = valid.numpy()
                for target in TARGETS:
                    source = (
                        session
                        / "bev_6p5m"
                        / target.directory
                        / f"frame_{frame_id:06d}.png"
                    )
                    labels = _load_gt(source, target.output_size)
                    measurement = measure_gt_bev(
                        ~valid_by_branch[(frame_id, target.branch)],
                        labels,
                    )
                    row = {
                        "session_key": session_key,
                        "scene_key": scene_key,
                        "dataset": str(metadata["dataset"]),
                        "split": session_split,
                        "target_frame": frame_id,
                        "gt_bev": target.name,
                        "branch": target.branch,
                        "semantic_scope": target.semantic_scope,
                        "source_path": str(source),
                        "output_height": target.output_size,
                        "output_width": target.output_size,
                        **asdict(measurement),
                    }
                    writer.writerow(row)
                    _update_summary(summary_groups, row)
            if log_every_sessions > 0 and session_number % log_every_sessions == 0:
                print(
                    f"audited {session_number:,}/{len(sessions):,} sessions",
                    flush=True,
                )
    temporary_csv.replace(csv_path)
    aggregate = _finalize_summary(summary_groups)
    summary = {
        "format_version": 1,
        "module": "p1b_void_audit",
        "training_integration": False,
        "read_only": True,
        "dataset_root": str(root),
        "manifest": str(manifest_file),
        "manifest_content_sha256": str(manifest["content_sha256"]),
        "void_index": str(Path(void_index_path).expanduser().resolve()),
        "void_index_content_sha256": coverage.content_sha256,
        "coverage_algorithm": coverage.algorithm,
        "split": split,
        "frame_mode": frame_mode,
        "maximum_history": maximum_history,
        "session_count": len(sessions),
        "full_session_count": full_session_count,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "csv": str(csv_path),
        **aggregate,
    }
    summary_path = destination / "void_fraction_summary.json"
    _atomic_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    return summary
