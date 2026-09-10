from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _canonical_digest(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_audit_snapshot(
    *,
    dataset_root: str | Path,
    output_path: str | Path,
    expected_sessions: int | None = None,
    log_every_sessions: int = 10_000,
) -> dict:
    """Freeze all completed sessions into a lightweight read-only audit manifest.

    Unlike a training manifest, this snapshot intentionally fingerprints only
    the immutable session metadata needed to locate geometry and GT rasters.
    It never hashes RGB/depth payloads because the Void audit does not consume
    them.
    """

    root = Path(dataset_root).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    markers = sorted(
        marker
        for marker in root.glob("*/session_*/COMPLETE")
        if marker.is_file()
    )
    if expected_sessions is not None and len(markers) != expected_sessions:
        raise ValueError(
            f"expected {expected_sessions:,} complete sessions, found "
            f"{len(markers):,}"
        )
    rows: list[dict] = []
    for number, marker in enumerate(markers, start=1):
        session = marker.parent
        metadata_path = session / "metadata.json"
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
        if metadata.get("status") != "complete":
            raise ValueError(f"session is not marked complete: {session}")
        frame_count = int(metadata.get("frame_count", 0))
        if frame_count <= 0:
            raise ValueError(f"session has no frames: {session}")
        frame = frame_count - 1
        required = (
            session / str(
                metadata.get("camera_extrinsics_file", "camera_extrinsics.jsonl")
            ),
            session / "bev_6p5m" / "complete" / f"frame_{frame:06d}.png",
            session / "bev_6p5m" / "masked" / f"frame_{frame:06d}.png",
            session
            / "bev_6p5m"
            / "merged_complete_10m"
            / f"frame_{frame:06d}.png",
            session
            / "bev_6p5m"
            / "merged_masked_10m"
            / f"frame_{frame:06d}.png",
        )
        missing = next((path for path in required if not path.is_file()), None)
        if missing is not None:
            raise FileNotFoundError(f"audit input is missing: {missing}")
        dataset = str(metadata["dataset"])
        scene_id = str(metadata["scene_id"])
        rows.append(
            {
                "key": session.relative_to(root).as_posix(),
                "scene_key": f"{dataset}:{scene_id}",
                "frame_count": frame_count,
                "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            }
        )
        if log_every_sessions > 0 and number % log_every_sessions == 0:
            print(f"snapshotted {number:,}/{len(markers):,} sessions", flush=True)

    payload = {
        "format_version": 1,
        "module": "p1b_void_audit",
        "snapshot_scope": "all-completed-sessions-read-only-audit-v1",
        "dataset_root": str(root),
        "session_count": len(rows),
        "scene_count": len({row["scene_key"] for row in rows}),
        "train": rows,
        "validation": [],
    }
    payload["content_sha256"] = _canonical_digest(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a lightweight all-completed-session Void-audit snapshot."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sessions", type=int)
    parser.add_argument("--log-every-sessions", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = create_audit_snapshot(
        dataset_root=args.dataset_root,
        output_path=args.output,
        expected_sessions=args.expected_sessions,
        log_every_sessions=args.log_every_sessions,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output.expanduser().resolve()),
                "sessions": result["session_count"],
                "scenes": result["scene_count"],
                "content_sha256": result["content_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
