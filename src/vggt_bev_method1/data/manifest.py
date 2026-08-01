from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .dataset import load_session_records

FORMAT_VERSION = 6


def _canonical_digest(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _training_artifacts(record) -> list[Path]:
    paths = [
        record.path / "camera_intrinsics.json",
        record.path / "camera_extrinsics.jsonl",
    ]
    for frame in range(record.frame_count):
        filename = f"frame_{frame:06d}.png"
        paths.append(record.path / "camera" / filename)
        paths.append(
            record.path / "depth" / f"frame_{frame:06d}{record.depth_suffix}"
        )
        paths.append(record.path / "bev_6p5m/masked" / filename)
        paths.append(record.path / "bev_6p5m/complete" / filename)
        paths.append(
            record.path / "bev_6p5m/merged_masked_10m" / filename
        )
        paths.append(
            record.path / "bev_6p5m/merged_complete_10m" / filename
        )
    return paths


def _artifact_inventory(record) -> tuple[str, int]:
    digest = hashlib.sha256()
    paths = _training_artifacts(record)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required training artifact is missing: {path}")
        stat = path.stat()
        relative = path.relative_to(record.path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(paths)


def _session_entry(record) -> dict:
    metadata_path = record.path / "metadata.json"
    artifact_digest, artifact_count = _artifact_inventory(record)
    return {
        "key": record.key,
        "scene_key": record.scene_key,
        "frame_count": record.frame_count,
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "artifact_inventory_sha256": artifact_digest,
        "artifact_count": artifact_count,
    }


def create_split_manifest(
    dataset_root: str | Path,
    manifest_path: str | Path,
    *,
    validation_fraction: float,
    seed: int,
    maximum_sessions: int | None = None,
    source_writer_count: int = 0,
) -> dict:
    """Freeze one exact dataset snapshot for new P1B."""

    root = Path(dataset_root).expanduser().resolve()
    path = Path(manifest_path).expanduser().resolve()
    if path.exists():
        return load_split_manifest(
            path,
            dataset_root=root,
            validation_fraction=validation_fraction,
            seed=seed,
            maximum_sessions=maximum_sessions,
            verify_metadata=True,
        )

    records = sorted(load_session_records(root), key=lambda record: record.key)
    source_session_count = len(records)
    if maximum_sessions is not None:
        if maximum_sessions <= 0:
            raise ValueError("maximum_sessions must be positive")
        if source_session_count < maximum_sessions:
            raise ValueError(
                f"dataset has {source_session_count} sessions, fewer than "
                f"requested first {maximum_sessions}"
            )
        records = records[:maximum_sessions]
    record_by_key = {record.key: record for record in records}
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record.scene_key].append(record.key)
    scene_keys = sorted(grouped)
    if len(scene_keys) < 2:
        raise ValueError("at least two scenes are required for a grouped split")
    random.Random(seed).shuffle(scene_keys)
    validation_scene_count = max(1, round(len(scene_keys) * validation_fraction))
    validation_scene_count = min(validation_scene_count, len(scene_keys) - 1)
    validation_scenes = set(scene_keys[:validation_scene_count])
    train_keys = sorted(
        key
        for scene, keys in grouped.items()
        if scene not in validation_scenes
        for key in keys
    )
    validation_keys = sorted(
        key
        for scene, keys in grouped.items()
        if scene in validation_scenes
        for key in keys
    )
    payload = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "source_session_count": source_session_count,
        "maximum_sessions": maximum_sessions,
        "source_writer_count_at_freeze": int(source_writer_count),
        "snapshot_scope": (
            "completed_session_prefix"
            if source_writer_count
            else "immutable_dataset"
        ),
        "selection_order": "lexicographic session key, first N before scene split",
        "validation_fraction": validation_fraction,
        "split_seed": seed,
        "session_count": len(records),
        "scene_count": len({record.scene_key for record in records}),
        "train": [_session_entry(record_by_key[key]) for key in train_keys],
        "validation": [
            _session_entry(record_by_key[key]) for key in validation_keys
        ],
    }
    payload["content_sha256"] = _canonical_digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        os.link(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        return load_split_manifest(
            path,
            dataset_root=root,
            validation_fraction=validation_fraction,
            seed=seed,
            maximum_sessions=maximum_sessions,
            verify_metadata=True,
        )
    temporary.unlink()
    return payload


def load_split_manifest(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path,
    validation_fraction: float,
    seed: int,
    maximum_sessions: int | None = None,
    verify_metadata: bool = True,
    verify_artifacts: bool = True,
) -> dict:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"split manifest does not exist: {path}; run vggt-bev-m1-freeze-split first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", 0)) != FORMAT_VERSION:
        raise ValueError(f"unsupported split manifest format: {path}")
    expected_digest = payload.pop("content_sha256", None)
    actual_digest = _canonical_digest(payload)
    payload["content_sha256"] = expected_digest
    if expected_digest != actual_digest:
        raise ValueError(f"split manifest checksum mismatch: {path}")

    root = Path(dataset_root).expanduser().resolve()
    if Path(payload["dataset_root"]).resolve() != root:
        raise ValueError("split manifest dataset root does not match the configuration")
    if int(payload["split_seed"]) != seed:
        raise ValueError("split manifest seed does not match the configuration")
    if float(payload["validation_fraction"]) != validation_fraction:
        raise ValueError(
            "split manifest validation fraction does not match the configuration"
        )
    if payload.get("maximum_sessions") != maximum_sessions:
        raise ValueError(
            "split manifest maximum_sessions does not match the configuration"
        )
    if maximum_sessions is not None and int(payload["session_count"]) != maximum_sessions:
        raise ValueError(
            "limited split manifest does not contain exactly maximum_sessions"
        )

    train = payload["train"]
    validation = payload["validation"]
    train_keys = {entry["key"] for entry in train}
    validation_keys = {entry["key"] for entry in validation}
    if len(train_keys) != len(train) or len(validation_keys) != len(validation):
        raise ValueError("split manifest contains duplicate session keys")
    if train_keys.intersection(validation_keys):
        raise ValueError("split manifest leaks sessions across train and validation")
    train_scenes = {entry["scene_key"] for entry in train}
    validation_scenes = {entry["scene_key"] for entry in validation}
    if train_scenes.intersection(validation_scenes):
        raise ValueError("split manifest leaks scenes across train and validation")
    if len(train) + len(validation) != int(payload["session_count"]):
        raise ValueError("split manifest session count is inconsistent")

    if verify_metadata:
        records = {
            record.key: record
            for record in load_session_records(
                root,
                [entry["key"] for entry in train + validation],
            )
        }
        for entry in train + validation:
            metadata_path = root / entry["key"] / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"manifest session metadata is missing: {metadata_path}"
                )
            digest = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
            if digest != entry["metadata_sha256"]:
                raise ValueError(
                    f"manifest session metadata changed after freezing: {metadata_path}"
                )
            if verify_artifacts:
                artifact_digest, artifact_count = _artifact_inventory(
                    records[entry["key"]]
                )
                if artifact_count != int(entry["artifact_count"]):
                    raise ValueError(
                        "manifest session artifact count changed after freezing: "
                        f"{entry['key']}"
                    )
                if artifact_digest != entry["artifact_inventory_sha256"]:
                    raise ValueError(
                        "manifest session training artifacts changed after freezing: "
                        f"{entry['key']}"
                    )
    return payload


def manifest_session_keys(payload: dict) -> tuple[list[str], list[str]]:
    return (
        [entry["key"] for entry in payload["train"]],
        [entry["key"] for entry in payload["validation"]],
    )
