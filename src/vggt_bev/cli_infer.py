from __future__ import annotations

import argparse
import json
from pathlib import Path

from vggt_bev.runtime import (
    MODEL_FILENAMES,
    Method2Runtime,
    default_model_path,
    default_vggt_paths,
    load_vggnav_runtime_sequence,
    save_runtime_prediction,
)


def build_parser() -> argparse.ArgumentParser:
    default_source, default_backbone = default_vggt_paths()
    parser = argparse.ArgumentParser(
        description="Run a trained Method II dual-output checkpoint on a VGGNAV camera history"
    )
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_FILENAMES), default="5m")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="override the checkpoint selected by --model",
    )
    parser.add_argument("--target-frame", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vggt-source", type=Path, default=default_source)
    parser.add_argument("--vggt-checkpoint", type=Path, default=default_backbone)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--camera-height-m",
        type=float,
        default=None,
        help="legacy metric checkpoints only; vggt_raw requires no camera height",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    checkpoint = args.checkpoint or default_model_path(args.model)
    runtime = Method2Runtime(
        checkpoint,
        vggt_source=args.vggt_source,
        vggt_checkpoint=args.vggt_checkpoint,
        device=args.device,
    )
    sequence = load_vggnav_runtime_sequence(
        args.session,
        target_frame=args.target_frame,
        image_height=runtime.image_height,
        image_width=runtime.image_width,
        camera_height_m=args.camera_height_m,
    )
    prediction = runtime.predict(sequence, threshold=args.threshold)
    outputs = save_runtime_prediction(
        prediction,
        args.output_dir,
        sequence=sequence,
        checkpoint_path=checkpoint,
        threshold=args.threshold,
    )
    report = {
        "passed": True,
        "model": args.model,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "checkpoint_epoch": prediction.checkpoint_epoch,
        "checkpoint_global_step": prediction.checkpoint_global_step,
        "history_frame_count": len(sequence.frame_ids),
        "coordinate_mode": prediction.coordinate_mode,
        "single_extent": prediction.single_extent_m,
        "merged_extent": prediction.merged_extent_m,
        "output_size": prediction.output_size,
        "depth_scale": prediction.depth_scale,
        "outputs": outputs,
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
