from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vggt_bev.data import (
    CalibrationAwareResize,
    VGGNAVMethod2Dataset,
    method2_collate,
)
from vggt_bev.losses import method2_loss
from vggt_bev.models import FrozenVGGTAdapter, Method2BEVHead, Method2System


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one low-resolution live VGGT train step")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/user/Project/VGGNAV/output/random_sessions_test_3"),
    )
    parser.add_argument(
        "--vggt-source", type=Path, default=Path("/home/user/Project/vggt")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/user/Project/vggt/checkpoints/VGGT-Omega-1B-512/model.pt"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--extent-key", default="bev_5m")
    parser.add_argument("--target-mode", choices=("single", "merged", "both"), default="single")
    parser.add_argument("--history", choices=("first", "longest"), default="first")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--bev-feature-size", type=int, default=32)
    parser.add_argument("--ray-steps", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.image_height % 16 or args.image_width % 16:
        raise ValueError("image dimensions must be divisible by 16")

    dataset = VGGNAVMethod2Dataset(
        args.data_root,
        extent_key=args.extent_key,
        target_mode=args.target_mode,
        preprocess=CalibrationAwareResize(args.image_height, args.image_width),
        geometry_source="vggt",
    )
    sample_index = (
        max(
            range(len(dataset)),
            key=lambda index: dataset.samples[index].target_frame_index,
        )
        if args.history == "longest"
        else 0
    )
    batch = method2_collate([dataset[sample_index]])
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

    adapter = FrozenVGGTAdapter(args.vggt_source, args.checkpoint, device=device)
    head = Method2BEVHead(
        feature_dim=2048,
        hidden_dim=args.hidden_dim,
        output_size=512,
        ray_steps=args.ray_steps,
    ).to(device)
    merged_head = (
        Method2BEVHead(
            feature_dim=2048,
            hidden_dim=args.hidden_dim,
            output_size=512,
            ray_steps=args.ray_steps,
        ).to(device)
        if args.target_mode == "both"
        else None
    )
    model = Method2System(
        adapter,
        head,
        merged_head=merged_head,
        bev_feature_size=args.bev_feature_size,
        learn_depth_scale=False,
        metric_scale_mode="camera_height",
        stabilize_intrinsics=True,
        ground_minimum_points=4,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4)

    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch)
    if args.target_mode == "both":
        single_loss = method2_loss(prediction["single"], batch["single_target_labels"])
        merged_loss = method2_loss(prediction["merged"], batch["merged_target_labels"])
        loss = 0.5 * (single_loss["loss"] + merged_loss["loss"])
        output_shape: dict | list = {
            "single": list(prediction["single"]["occupancy_logit"].shape),
            "merged": list(prediction["merged"]["occupancy_logit"].shape),
        }
    else:
        losses = method2_loss(prediction, batch["target_labels"])
        loss = losses["loss"]
        output_shape = list(prediction["occupancy_logit"].shape)
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in trainable
    )
    optimizer.step()
    report = {
        "passed": bool(torch.isfinite(loss) and finite_gradients),
        "device": str(device),
        "target_mode": args.target_mode,
        "history_frame_count": int(batch["frame_valid"].sum()),
        "input_shape": list(batch["images"].shape),
        "output_shape": output_shape,
        "loss": float(loss.detach()),
        "metric_scale": float(
            prediction["depth_scale"].detach().reshape(-1).mean()
        ),
        "finite_gradients": finite_gradients,
    }
    if device.type == "cuda":
        report["peak_cuda_memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
