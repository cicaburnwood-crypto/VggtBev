#!/usr/bin/env python3
"""Freeze a scene-disjoint M05 manifest across several BoQ data roots.

The regular manifest creator intentionally accepts one dataset root.  M05's
BoQ corpus is stored under several subtrees of one common data root, so this
utility freezes only the explicitly named subtrees while retaining keys that
the existing dataset loader can resolve relative to the common root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from vggt_bev_method1.data.manifest import FORMAT_VERSION, _canonical_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze one all-current-data M05 BoQ union manifest"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--include-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=170905)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def _fingerprint_one(common_root: Path, session_path: Path) -> dict:
    metadata_path = session_path / "metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    if metadata.get("status") != "complete":
        raise ValueError("metadata status is not complete")
    frame_count = int(metadata["frame_count"])
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    return {
        "entry": {
            "key": session_path.relative_to(common_root).as_posix(),
            "scene_key": f"{metadata['dataset']}:{metadata['scene_id']}",
            "frame_count": frame_count,
            "metadata_sha256": metadata_sha256,
            # Required format-v6 fields. Training is configured not to restat
            # all frame artifacts in this stopped-writer snapshot.
            "artifact_inventory_sha256": metadata_sha256,
            "artifact_count": 0,
        },
        "identity_sha256": metadata_sha256,
        "complete_mtime_ns": (session_path / "COMPLETE").stat().st_mtime_ns,
        "include_root": None,
    }


def _discover_complete_sessions(root: Path) -> list[Path]:
    """Find standard BoQ session markers without entering session artifacts."""

    sessions = [
        marker.parent.resolve()
        for marker in root.glob("dense_workers/*/*/*/session_*/COMPLETE")
        if marker.is_file()
    ]
    sessions.sort()
    if not sessions:
        raise ValueError(f"no committed complete sessions found under {root}")
    return sessions


def _iter_bounded_results(common_root: Path, paths: list[Path], workers: int):
    maximum_pending = max(workers * 4, 1)
    path_iter = iter(paths)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}
        for _ in range(min(maximum_pending, len(paths))):
            path = next(path_iter)
            pending[executor.submit(_fingerprint_one, common_root, path)] = path
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                path = pending.pop(future)
                try:
                    yield path, future.result(), None
                except Exception as error:  # preserve every invalid-session reason
                    yield path, None, f"{type(error).__name__}: {error}"
                try:
                    next_path = next(path_iter)
                except StopIteration:
                    continue
                pending[
                    executor.submit(_fingerprint_one, common_root, next_path)
                ] = next_path


def main() -> None:
    args = parse_args()
    common_root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output}")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation fraction must be in (0,1)")
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    include_roots = [path.expanduser().resolve() for path in args.include_root]
    for include_root in include_roots:
        include_root.relative_to(common_root)
    if len(set(include_roots)) != len(include_roots):
        raise ValueError("include roots must be unique")

    snapshots: list[tuple[Path, Path]] = []
    source_counts: dict[str, int] = {}
    seen_paths: set[Path] = set()
    for include_root in include_roots:
        paths = _discover_complete_sessions(include_root)
        relative_root = include_root.relative_to(common_root).as_posix()
        source_counts[relative_root] = len(paths)
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                snapshots.append((include_root, resolved))
    snapshots.sort(key=lambda item: item[1].relative_to(common_root).as_posix())
    print(
        json.dumps(
            {
                "event": "snapshot",
                "candidate_sessions": len(snapshots),
                "source_counts": source_counts,
            }
        ),
        flush=True,
    )

    root_for_path = {path: root for root, path in snapshots}
    valid_results: list[dict] = []
    invalid_count = 0
    invalid_sample: list[dict[str, str]] = []
    processed = 0
    for path, result, error in _iter_bounded_results(
        common_root,
        [path for _, path in snapshots],
        args.workers,
    ):
        processed += 1
        if error is not None:
            invalid_count += 1
            if len(invalid_sample) < 100:
                invalid_sample.append(
                    {
                        "key": path.relative_to(common_root).as_posix(),
                        "error": error,
                    }
                )
        else:
            result["include_root"] = root_for_path[path].relative_to(
                common_root
            ).as_posix()
            valid_results.append(result)
        if processed % args.progress_every == 0 or processed == len(snapshots):
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "processed": processed,
                        "total": len(snapshots),
                        "valid": len(valid_results),
                        "invalid": invalid_count,
                    }
                ),
                flush=True,
            )

    # Exact copied session metadata can occur while importing another server's
    # BoQ pool. Deduplicate those snapshots without retaining a huge map in the
    # manifest.
    valid_counts_by_root: dict[str, int] = defaultdict(int)
    valid_frames_by_root: dict[str, int] = defaultdict(int)
    for result in valid_results:
        valid_counts_by_root[result["include_root"]] += 1
        valid_frames_by_root[result["include_root"]] += int(
            result["entry"]["frame_count"]
        )
    unique: dict[tuple[str, str, int], dict] = {}
    duplicate_count = 0
    duplicate_sample: list[dict[str, str]] = []
    for result in valid_results:
        entry = result["entry"]
        identity = (
            entry["metadata_sha256"],
            result["identity_sha256"],
            int(entry["frame_count"]),
        )
        incumbent = unique.get(identity)
        if incumbent is None:
            unique[identity] = result
            continue
        duplicate_count += 1
        if len(duplicate_sample) < 100:
            duplicate_sample.append(
                {
                    "discarded_key": entry["key"],
                    "kept_key": incumbent["entry"]["key"],
                }
            )
    selected = sorted(unique.values(), key=lambda item: item["entry"]["key"])

    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in selected:
        grouped[result["entry"]["scene_key"]].append(result["entry"])
    scene_keys = sorted(grouped)
    if len(scene_keys) < 2:
        raise ValueError("at least two scenes are required for scene-disjoint split")
    random.Random(args.seed).shuffle(scene_keys)
    validation_scene_count = max(
        1, round(len(scene_keys) * args.validation_fraction)
    )
    validation_scene_count = min(validation_scene_count, len(scene_keys) - 1)
    validation_scenes = set(scene_keys[:validation_scene_count])
    train = sorted(
        (
            entry
            for scene, entries in grouped.items()
            if scene not in validation_scenes
            for entry in entries
        ),
        key=lambda entry: entry["key"],
    )
    validation = sorted(
        (
            entry
            for scene, entries in grouped.items()
            if scene in validation_scenes
            for entry in entries
        ),
        key=lambda entry: entry["key"],
    )
    payload = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(common_root),
        "included_roots": [
            path.relative_to(common_root).as_posix() for path in include_roots
        ],
        "included_root_candidate_counts": source_counts,
        "included_root_valid_counts": dict(valid_counts_by_root),
        "included_root_valid_frame_counts": dict(valid_frames_by_root),
        "source_session_count": len(snapshots),
        "invalid_complete_session_count": invalid_count,
        "invalid_complete_sessions_sample": invalid_sample,
        "exact_duplicate_session_count": duplicate_count,
        "exact_duplicate_sessions_sample": duplicate_sample,
        "maximum_sessions": None,
        "source_writer_count_at_freeze": 0,
        "snapshot_scope": "completed_sessions_from_explicit_boq_roots",
        "freeze_fingerprint_mode": (
            "sha256_metadata_plus_complete_marker; writers_stopped; "
            "full_frame_artifact_restat_disabled"
        ),
        "selection_order": "lexicographic_session_key",
        "selection_order_description": (
            "all COMPLETE sessions with readable complete metadata in explicit "
            "BoQ roots; identical metadata deduplicated before scene split"
        ),
        "validation_fraction": args.validation_fraction,
        "split_seed": args.seed,
        "session_count": len(selected),
        "frame_count": sum(
            int(result["entry"]["frame_count"]) for result in selected
        ),
        "scene_count": len(grouped),
        "train": train,
        "validation": validation,
    }
    payload["content_sha256"] = _canonical_digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.link(temporary, output)
    temporary.unlink()
    print(
        json.dumps(
            {
                "event": "complete",
                "manifest": str(output),
                "content_sha256": payload["content_sha256"],
                "candidate_sessions": len(snapshots),
                "valid_unique_sessions": len(selected),
                "valid_unique_frames": payload["frame_count"],
                "invalid_sessions": invalid_count,
                "exact_duplicates": duplicate_count,
                "scenes": len(grouped),
                "train_sessions": len(train),
                "validation_sessions": len(validation),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
