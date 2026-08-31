#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_digest(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relocate an immutable split without changing its membership"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    arguments = parser.parse_args()

    payload = json.loads(arguments.source.read_text(encoding="utf-8"))
    source_digest = payload.pop("content_sha256", None)
    if source_digest != canonical_digest(payload):
        raise ValueError("source split manifest checksum is invalid")
    payload["dataset_root"] = str(arguments.dataset_root.resolve())
    payload["content_sha256"] = canonical_digest(payload)
    arguments.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.destination.with_suffix(
        arguments.destination.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(arguments.destination)
    print(
        json.dumps(
            {
                "source_content_sha256": source_digest,
                "relocated_content_sha256": payload["content_sha256"],
                "dataset_root": payload["dataset_root"],
                "sessions": payload["session_count"],
                "scenes": payload["scene_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
