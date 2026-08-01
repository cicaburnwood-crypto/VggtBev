from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from vggt_bev_method1.config import load_config
from vggt_bev_method1.data.manifest import create_split_manifest
from vggt_bev_method1.p2b_config import load_p2b_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze one immutable Method I train/validation snapshot"
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def active_dataset_writers(dataset_root: str | Path) -> list[dict[str, str | int]]:
    root = str(Path(dataset_root).expanduser().resolve())
    writers = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8",
                errors="replace",
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if root not in command:
            continue
        if not any(
            name in command
            for name in ("collect_random_sessions.py", "run_unlimited_data_build.py")
        ):
            continue
        writers.append({"pid": int(process.name), "command": command.strip()})
    return writers


def load_split_config(path: str | Path) -> dict:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as stream:
        raw = tomllib.load(stream)
    pipeline = str(raw.get("training", {}).get("pipeline", ""))
    if pipeline in ("P2B-NLL", "P2B-BCE"):
        return load_p2b_config(resolved)
    return load_config(resolved)


def main() -> None:
    args = parse_args()
    config = load_split_config(args.config)
    data = config["data"]
    writers = active_dataset_writers(data["root"])
    allow_completed_snapshot = bool(
        data.get("allow_active_writer_completed_snapshot", False)
    )
    if writers and not allow_completed_snapshot:
        pids = [writer["pid"] for writer in writers]
        raise RuntimeError(
            "refusing to freeze a mutable dataset while collection processes "
            f"are writing under the configured root; active PIDs: {pids}"
        )
    if writers and data.get("maximum_sessions") is None:
        raise RuntimeError(
            "an active-writer completed-session snapshot requires a finite "
            "data.maximum_sessions"
        )
    payload = create_split_manifest(
        data["root"],
        data["split_manifest"],
        validation_fraction=float(data["validation_fraction"]),
        seed=int(data["split_seed"]),
        maximum_sessions=(
            int(data["maximum_sessions"])
            if data.get("maximum_sessions") is not None
            else None
        ),
        source_writer_count=len(writers),
    )
    print(
        json.dumps(
            {
                "manifest": str(Path(data["split_manifest"]).expanduser().resolve()),
                "content_sha256": payload["content_sha256"],
                "sessions": payload["session_count"],
                "scenes": payload["scene_count"],
                "train_sessions": len(payload["train"]),
                "validation_sessions": len(payload["validation"]),
                "artifact_fingerprint": "path + byte-size + nanosecond mtime",
                "source_writers_at_freeze": len(writers),
                "snapshot_scope": payload.get("snapshot_scope"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
