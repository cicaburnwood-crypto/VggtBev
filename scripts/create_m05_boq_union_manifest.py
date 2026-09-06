#!/usr/bin/env python3
"""Freeze a scene-disjoint M05 manifest across several BoQ data roots.

The regular manifest creator intentionally accepts one dataset root.  M05's
BoQ corpus is stored under several subtrees of one common data root, so this
utility freezes only the explicitly named subtrees while retaining keys that
the existing dataset loader can resolve relative to the common root.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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


def _entry_one(common_root: Path, session_path: Path) -> dict:
    parts = session_path.name.split("_", 3)
    if len(parts) != 4 or parts[0] != "session" or not parts[1].isdigit():
        raise ValueError(f"unexpected BoQ session directory name: {session_path}")
    dataset, scene_id = parts[2:]
    return {
        "key": session_path.relative_to(common_root).as_posix(),
        "scene_key": f"{dataset}:{scene_id}",
        # The fast snapshot deliberately does not open per-session files.
        # Training reconstructs the true frame count from metadata.json.
        "frame_count": 0,
        "metadata_sha256": "not_scanned_fast_freeze",
        "artifact_inventory_sha256": "not_scanned_fast_freeze",
        "artifact_count": 0,
    }


def _discover_complete_sessions(root: Path) -> list[Path]:
    """Find standard BoQ session markers without entering session artifacts."""

    completed = subprocess.run(
        (
            "find",
            os.fspath(root),
            "-mindepth",
            "6",
            "-maxdepth",
            "6",
            "-type",
            "f",
            "-path",
            "*/dense_workers/*/*/*/session_*/COMPLETE",
            "-print0",
        ),
        check=True,
        stdout=subprocess.PIPE,
    )
    sessions = [
        Path(os.fsdecode(value)).parent
        for value in completed.stdout.split(b"\0")
        if value
    ]
    sessions.sort()
    if not sessions:
        raise ValueError(f"no committed complete sessions found under {root}")
    return sessions


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
    with ThreadPoolExecutor(max_workers=min(3, len(include_roots))) as executor:
        discovered = executor.map(_discover_complete_sessions, include_roots)
        root_paths = list(
            zip(  # noqa: B905 - executor.map preserves input length
                include_roots, discovered
            )
        )
    for include_root, paths in root_paths:
        relative_root = include_root.relative_to(common_root).as_posix()
        source_counts[relative_root] = len(paths)
        snapshots.extend((include_root, path) for path in paths)
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

    selected = [_entry_one(common_root, path) for _, path in snapshots]

    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in selected:
        grouped[entry["scene_key"]].append(entry)
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
        "source_session_count": len(snapshots),
        "invalid_complete_session_count": 0,
        "exact_duplicate_session_count": 0,
        "deduplication_mode": "none_preserve_maximum_complete_sessions",
        "maximum_sessions": None,
        "source_writer_count_at_freeze": 0,
        "snapshot_scope": "completed_sessions_from_explicit_boq_roots",
        "freeze_fingerprint_mode": (
            "complete_marker_and_session_key_only; writers_stopped; "
            "per_session_files_not_opened"
        ),
        "selection_order": "lexicographic_session_key",
        "selection_order_description": (
            "all standard-layout COMPLETE session paths in explicit BoQ roots; "
            "no deduplication or per-session file scan"
        ),
        "validation_fraction": args.validation_fraction,
        "split_seed": args.seed,
        "session_count": len(selected),
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
                "invalid_sessions": 0,
                "exact_duplicates_removed": 0,
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
