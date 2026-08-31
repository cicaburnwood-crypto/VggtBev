#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

from vggt_bev_method1.data.dataset import (
    SESSION_RECORD_CACHE_FORMAT,
    SessionRecord,
    _load_record,
)


def _load_one(arguments: tuple[str, str]) -> SessionRecord:
    root_text, key = arguments
    root = Path(root_text)
    return _load_record(root, root / key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one validated compact SessionRecord cache for DDP ranks"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
    started = time.monotonic()
    context = mp.get_context("fork")
    try:
        with context.Pool(processes=args.workers) as pool:
            jobs = ((str(root), key) for key in keys)
            for completed, record in enumerate(
                pool.imap_unordered(_load_one, jobs, chunksize=16), start=1
            ):
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
        if len(records_by_key) != len(keys):
            raise RuntimeError("session-record cache builder lost records")
        payload = {
            "format_version": SESSION_RECORD_CACHE_FORMAT,
            "dataset_root": str(root),
            "manifest_content_sha256": manifest["content_sha256"],
            "session_count": len(keys),
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
                    "sessions": len(keys),
                    "elapsed_seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
