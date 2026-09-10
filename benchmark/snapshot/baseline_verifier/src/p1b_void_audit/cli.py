from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Void audit: report the Void fraction of every "
            "corresponding GT BEV without affecting P1B training."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--void-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation"),
        default="all",
    )
    parser.add_argument(
        "--frame-mode",
        choices=("final", "all"),
        default="final",
        help="final matches one-prefix-per-session training; all audits 1..N frames",
    )
    parser.add_argument("--maximum-history", type=int, default=10)
    parser.add_argument("--maximum-sessions", type=int)
    parser.add_argument("--log-every-sessions", type=int, default=100)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--skip-artifact-hash-verification",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_manifest(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        void_index_path=args.void_index,
        output_dir=args.output_dir,
        split=args.split,
        frame_mode=args.frame_mode,
        maximum_history=args.maximum_history,
        maximum_sessions=args.maximum_sessions,
        verify_artifacts=not args.skip_artifact_hash_verification,
        log_every_sessions=args.log_every_sessions,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
