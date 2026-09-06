#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

from vggt_bev_method1.data.dataset import (
    SESSION_RECORD_CACHE_FORMAT,
    SessionRecord,
    _load_record,
)


def _load_one(
    arguments: tuple[str, str],
) -> tuple[str, SessionRecord | None, str | None]:
    root_text, key = arguments
    root = Path(root_text)
    try:
        return key, _load_record(root, root / key), None
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        return key, None, f"{type(error).__name__}: {error}"


def _canonical_digest(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_filtered_manifest(
    source: dict,
    output: Path,
    valid_keys: set[str],
    invalid_sessions: list[dict[str, str]],
) -> dict:
    payload = {key: value for key, value in source.items() if key != "content_sha256"}
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["filtered_from_manifest_sha256"] = source["content_sha256"]
    payload["invalid_complete_session_count"] = len(invalid_sessions)
    payload["invalid_complete_sessions"] = invalid_sessions
    payload["train"] = [
        entry for entry in source["train"] if str(entry["key"]) in valid_keys
    ]
    payload["validation"] = [
        entry for entry in source["validation"] if str(entry["key"]) in valid_keys
    ]
    payload["session_count"] = len(valid_keys)
    payload["source_session_count"] = len(valid_keys)
    payload["scene_count"] = len(
        {
            str(entry["scene_key"])
            for entry in payload["train"] + payload["validation"]
        }
    )
    payload["content_sha256"] = _canonical_digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one validated compact SessionRecord cache for DDP ranks"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--filtered-manifest",
        type=Path,
        help="Write a manifest without invalid COMPLETE sessions and bind the cache to it",
    )
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    root = args.dataset_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(manifest["dataset_root"]).resolve() != root:
        raise ValueError("manifest and requested dataset root differ")
    entries = [*manifest["train"], *manifest["validation"]]
    keys = [str(entry["key"]) for entry in entries]
    if len(keys) != int(manifest["session_count"]) or len(set(keys)) != len(keys):
        raise ValueError("manifest session membership is inconsistent")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    records_by_key: dict[str, SessionRecord] = {}
    invalid_sessions: list[dict[str, str]] = []
    started = time.monotonic()
    context = mp.get_context("fork")
    try:
        with context.Pool(processes=args.workers) as pool:
            jobs = ((str(root), key) for key in keys)
            for completed, (key, record, error) in enumerate(
                pool.imap_unordered(_load_one, jobs, chunksize=16), start=1
            ):
                if record is None:
                    invalid_sessions.append({"key": key, "error": str(error)})
                    print(
                        json.dumps(
                            {"status": "invalid_session", "key": key, "error": error}
                        ),
                        flush=True,
                    )
                else:
                    records_by_key[record.key] = record
                if completed % 1000 == 0 or completed == len(keys):
                    elapsed = time.monotonic() - started
                    rate = completed / max(elapsed, 1e-6)
                    remaining = (len(keys) - completed) / max(rate, 1e-6)
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "total": len(keys),
                                "sessions_per_second": rate,
                                "eta_seconds": remaining,
                            }
                        ),
                        flush=True,
                    )
        if len(records_by_key) + len(invalid_sessions) != len(keys):
            raise RuntimeError("session-record cache builder lost records")
        cache_manifest = manifest
        if invalid_sessions:
            if args.filtered_manifest is None:
                raise RuntimeError(
                    "invalid sessions were found; provide --filtered-manifest"
                )
            cache_manifest = _write_filtered_manifest(
                manifest,
                args.filtered_manifest.expanduser().resolve(),
                set(records_by_key),
                invalid_sessions,
            )
        payload = {
            "format_version": SESSION_RECORD_CACHE_FORMAT,
            "dataset_root": str(root),
            "manifest_content_sha256": cache_manifest["content_sha256"],
            "session_count": len(records_by_key),
            "records_by_key": records_by_key,
        }
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": str(output),
                    "bytes": output.stat().st_size,
                    "sessions": len(records_by_key),
                    "invalid_sessions": len(invalid_sessions),
                    "manifest_content_sha256": cache_manifest["content_sha256"],
                    "elapsed_seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
