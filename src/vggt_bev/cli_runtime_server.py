from __future__ import annotations

import argparse
from pathlib import Path

from vggt_bev.runtime import (
    MODEL_FILENAMES,
    Method2Runtime,
    MultiMethod2Runtime,
    default_model_path,
)
from vggt_bev.runtime_server import RuntimeHistory, serve_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve stateful Method II inference for the simulator comparison UI"
    )
    parser.add_argument("--model", choices=sorted(MODEL_FILENAMES), default="5m")
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="load the 3.5 m, 5 m, and 6.5 m heads with one shared VGGT backbone",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vggt-source", type=Path)
    parser.add_argument("--vggt-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-history", type=int, default=34)
    parser.add_argument(
        "--camera-height-m",
        type=float,
        default=None,
        help="legacy metric checkpoints only; vggt_raw requires no camera height",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.all_models:
        if args.checkpoint is not None:
            raise SystemExit("--checkpoint cannot be combined with --all-models")
        runtime = MultiMethod2Runtime(
            {
                model_key: default_model_path(model_key)
                for model_key in MODEL_FILENAMES
            },
            vggt_source=args.vggt_source,
            vggt_checkpoint=args.vggt_checkpoint,
            device=args.device,
        )
    else:
        checkpoint = args.checkpoint or default_model_path(args.model)
        runtime = Method2Runtime(
            checkpoint,
            vggt_source=args.vggt_source,
            vggt_checkpoint=args.vggt_checkpoint,
            device=args.device,
        )
    history = RuntimeHistory(
        runtime,
        max_history=args.max_history,
        camera_height_m=args.camera_height_m,
    )
    serve_runtime(history, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
