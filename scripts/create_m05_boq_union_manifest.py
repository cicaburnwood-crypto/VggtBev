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
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from vggt_bev_method1.data.dataset import _load_record, discover_sessions
from vggt_bev_method1.data.manifest import (
    FORMAT_VERSION,
    _canonical_digest,
    _session_entry,
)


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
    record = _load_record(common_root, session_path)
    return {
        "entry": _session_entry(record),
        "complete_mtime_ns": (session_path / "COMPLETE").stat().st_mtime_ns,
        "include_root": None,
    }


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
        paths = discover_sessions(include_root)
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
    invalid: list[dict[str, str]] = []
    processed = 0
    for path, result, error in _iter_bounded_results(
        common_root,
        [path for _, path in snapshots],
        args.workers,
    ):
        processed += 1
        if error is not None:
            invalid.append(
                {"key": path.relative_to(common_root).as_posix(), "error": error}
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
                        "invalid": len(invalid),
                    }
                ),
                flush=True,
            )

    # Exact copied sessions can occur while importing another server's BoQ
    # pool.  A duplicate must match metadata and the complete artifact inventory
    # rather than merely sharing a human-readable session name.
    unique: dict[tuple[str, str, int], dict] = {}
    duplicates: list[dict[str, str]] = []
    for result in valid_results:
        entry = result["entry"]
        identity = (
            entry["metadata_sha256"],
            entry["artifact_inventory_sha256"],
            int(entry["artifact_count"]),
        )
        incumbent = unique.get(identity)
        if incumbent is None:
            unique[identity] = result
            continue
        duplicates.append(
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
        "source_session_count": len(snapshots),
        "invalid_complete_session_count": len(invalid),
        "invalid_complete_sessions": invalid,
        "exact_duplicate_session_count": len(duplicates),
        "exact_duplicate_sessions": duplicates,
        "maximum_sessions": None,
        "source_writer_count_at_freeze": 0,
        "snapshot_scope": "completed_sessions_from_explicit_boq_roots",
        "selection_order": "lexicographic_session_key",
        "selection_order_description": (
            "all structurally valid COMPLETE sessions in explicit BoQ roots; "
            "exact copied sessions deduplicated before scene split"
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
                "invalid_sessions": len(invalid),
                "exact_duplicates": len(duplicates),
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
