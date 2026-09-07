#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic blacklist of incomplete frozen sessions"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--session-record-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


def _existing_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _first_missing_frame(
    directory: Path,
    *,
    frame_count: int,
    suffixes: tuple[str, ...],
) -> str | None:
    try:
        available = {
            Path(entry.name).stem
            for entry in os.scandir(directory)
            if entry.is_file(follow_symlinks=False)
            and Path(entry.name).suffix.lower() in suffixes
        }
    except OSError:
        return str(directory)
    for frame in range(frame_count):
        stem = f"frame_{frame:06d}"
        if stem not in available:
            return str(directory / f"{stem}{suffixes[0]}")
    return None


def _first_missing(record) -> str | None:
    static_paths = (
        record.path / "COMPLETE",
        record.path / "metadata.json",
        record.path / "camera_intrinsics.json",
        record.path / "camera_extrinsics.jsonl",
    )
    for path in static_paths:
        if not _existing_file(path):
            return str(path)
    raster_directories = (
        record.path / "camera",
        record.path / "bev_6p5m/masked",
        record.path / "bev_6p5m/complete",
        record.path / "bev_6p5m/merged_masked_10m",
        record.path / "bev_6p5m/merged_complete_10m",
    )
    for directory in raster_directories:
        missing = _first_missing_frame(
            directory,
            frame_count=record.frame_count,
            suffixes=(".png", ".jpeg", ".jpg"),
        )
        if missing is not None:
            return missing
    missing_depth = _first_missing_frame(
        record.path / "depth",
        frame_count=record.frame_count,
        suffixes=(record.depth_suffix,),
    )
    if missing_depth is not None:
        return missing_depth
    return None


def _check_chunk(chunk: list[tuple[str, object]]) -> list[tuple[str, str]]:
    invalid: list[tuple[str, str]] = []
    for key, record in chunk:
        missing = _first_missing(record)
        if missing is not None:
            invalid.append((key, missing))
    return invalid


def main() -> None:
    arguments = parse_args()
    if arguments.workers <= 0 or arguments.chunk_size <= 0:
        raise ValueError("workers and chunk-size must be positive")
    manifest_path = arguments.manifest.expanduser().resolve()
    cache_path = arguments.session_record_cache.expanduser().resolve()
    output_path = arguments.output.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with cache_path.open("rb") as stream:
        cache = pickle.load(stream)
    if cache.get("manifest_content_sha256") != manifest.get("content_sha256"):
        raise ValueError("session-record cache manifest SHA-256 mismatch")
    dataset_root = Path(manifest["dataset_root"]).resolve()
    if Path(cache.get("dataset_root", "")).resolve() != dataset_root:
        raise ValueError("session-record cache dataset root mismatch")
    records_by_key = cache.get("records_by_key", {})
    entries = manifest["train"] + manifest["validation"]
    items = []
    for entry in entries:
        key = entry["key"]
        if key not in records_by_key:
            raise KeyError(f"session-record cache lacks manifest key: {key}")
        items.append((key, records_by_key[key]))
    chunks = [
        items[index : index + arguments.chunk_size]
        for index in range(0, len(items), arguments.chunk_size)
    ]
    started = time.monotonic()
    incomplete: list[tuple[str, str]] = []
    checked = 0
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        for result in executor.map(_check_chunk, chunks):
            incomplete.extend(result)
            checked += min(arguments.chunk_size, len(items) - checked)
            if checked % 16384 < arguments.chunk_size or checked == len(items):
                print(
                    json.dumps(
                        {
                            "completeness_checked_sessions": checked,
                            "completeness_total_sessions": len(items),
                            "incomplete_sessions": len(incomplete),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    incomplete.sort()
    payload = {
        "format_version": FORMAT_VERSION,
        "manifest_content_sha256": manifest["content_sha256"],
        "dataset_root": str(dataset_root),
        "checked_session_count": len(items),
        "complete_session_count": len(items) - len(incomplete),
        "incomplete_session_count": len(incomplete),
        "incomplete_session_keys": [key for key, _ in incomplete],
        "first_missing_artifact_by_session": dict(incomplete),
        "scan_elapsed_seconds": time.monotonic() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "completeness_index": str(output_path),
                "checked_session_count": len(items),
                "complete_session_count": len(items) - len(incomplete),
                "incomplete_session_count": len(incomplete),
                "scan_elapsed_seconds": payload["scan_elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
