#!/usr/bin/env python3
"""Small, repeatable M05 capacity/loss diagnostic on cached frozen-VGGT features."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from vggt_bev_method1.cli_eval_p1d import _sample_metrics, _summarize
from vggt_bev_method1.cli_train_m05 import build_model
from vggt_bev_method1.cli_train_metric import _build_scale_target, scale_fit_config
from vggt_bev_method1.m04_losses import m04_scale_loss
from vggt_bev_method1.m05_config import load_m05_config
from vggt_bev_method1.m05_losses import m05_bev_loss, m05_loss_weights
from vggt_bev_method1.m05_train_utils import (
    build_m05_datasets,
    fixed_metric_m05_target,
    m05_collate,
)
from vggt_bev_method1.train_utils import move_batch, seed_everything


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overfit a tiny fixed M05 batch and compare train/holdout BEV metrics"
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=1)
    parser.add_argument("--holdout-samples", type=int, default=1)
    parser.add_argument(
        "--cache-micro-batch-size",
        type=int,
        default=1,
        help=(
            "Frozen-VGGT extraction micro-batch. Cached features are concatenated "
            "before every M05-head optimizer step, so this does not change the "
            "true diagnostic training batch or its gradients."
        ),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-steps", default="0,1,5,20,50,100")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--latest-auxiliary-weight", type=float, default=0.5)
    parser.add_argument("--history-gate-initial-bias", type=float)
    parser.add_argument("--gate-boundary-emphasis", type=float, default=0.0)
    parser.add_argument("--support-boundary-emphasis", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=170907)
    parser.add_argument("--maximum-visual-samples", type=int, default=4)
    return parser.parse_args()


def _fixed_samples(
    dataset,
    count: int,
    history: int,
    seed: int,
) -> list[dict]:
    candidates: dict[str, list[int]] = {}
    sessions: set[int] = set()
    for index, record in enumerate(dataset.samples):
        if record.target_frame + 1 != history or record.session_index in sessions:
            continue
        sessions.add(record.session_index)
        source = dataset.sessions[record.session_index].dataset
        candidates.setdefault(source, []).append(index)
    total = sum(len(indices) for indices in candidates.values())
    if total < count:
        raise ValueError(
            f"dataset has only {total} unique sessions with history={history}"
        )

    sources = sorted(candidates)
    random.Random(seed).shuffle(sources)
    for source_index, source in enumerate(sources):
        random.Random(seed + 1009 * (source_index + 1)).shuffle(
            candidates[source]
        )
    selected_indices: list[int] = []
    cursors = {source: 0 for source in sources}
    while len(selected_indices) < count:
        progressed = False
        for source in sources:
            cursor = cursors[source]
            if cursor >= len(candidates[source]):
                continue
            selected_indices.append(candidates[source][cursor])
            cursors[source] = cursor + 1
            progressed = True
            if len(selected_indices) == count:
                break
        if not progressed:
            raise RuntimeError("stratified fixed-sample selection stalled")
    return [dataset[index] for index in selected_indices]


def _fixed_target(
    batch: dict,
    config: dict,
    *,
    prefix: str,
) -> dict[str, torch.Tensor]:
    return fixed_metric_m05_target(
        batch[f"{prefix}_fov_complete_target"],
        batch[f"{prefix}_visible_target"],
        batch[f"{prefix}_fov_support_target"],
        batch[f"{prefix}_gt_valid_mask"],
        extent_m=float(config["model"]["merged_bev_extent_m"]),
        latest_observed_free=batch["latest_observed_free_target"],
        latest_support=batch["latest_fov_support_target"],
    )


@torch.no_grad()
def _cache_batch(model, samples: list[dict], config: dict, device: torch.device) -> dict:
    batch = m05_collate(samples)
    rgb = batch["images"][:, -1].clone()
    metadata = list(batch["metadata"])
    batch = move_batch(batch, device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        enabled=device.type == "cuda",
    ):
        extraction = model.extract(batch["images"])
        teacher = model.decode_scale_teacher(extraction)
        scale_target = _build_scale_target(
            batch, teacher, scale_fit_config(config)
        )
    for key in ("_aggregated", "_patch_start", "_images"):
        extraction.pop(key, None)
    if use_bf16:
        extraction["tokens"] = {
            layer: value.to(dtype=torch.bfloat16)
            for layer, value in extraction["tokens"].items()
        }
        extraction["camera_register_tokens"] = extraction[
            "camera_register_tokens"
        ].to(dtype=torch.bfloat16)
    return {
        "extraction": extraction,
        "scale_target": scale_target,
        "merged_target": _fixed_target(batch, config, prefix="merged"),
        "latest_target": _fixed_target(batch, config, prefix="latest"),
        "rgb": rgb,
        "metadata": metadata,
    }


def _concatenate_cached(parts: list[dict]) -> dict:
    if not parts:
        raise ValueError("at least one cached micro-batch is required")

    def concatenate(values: list, path: str):
        first = values[0]
        if isinstance(first, torch.Tensor):
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise TypeError(f"inconsistent cached values at {path}")
            return torch.cat(values, dim=0)
        if isinstance(first, dict):
            keys = tuple(first)
            if not all(tuple(value) == keys for value in values):
                raise KeyError(f"inconsistent cached dictionary at {path}")
            return {
                key: concatenate(
                    [value[key] for value in values],
                    f"{path}.{key}",
                )
                for key in keys
            }
        if isinstance(first, list):
            if not all(isinstance(value, list) for value in values):
                raise TypeError(f"inconsistent cached lists at {path}")
            return [item for value in values for item in value]
        if not all(value == first for value in values[1:]):
            raise ValueError(f"non-batched cached value changed at {path}")
        return first

    return concatenate(parts, "cache")


@torch.no_grad()
def _cache_samples(
    model,
    samples: list[dict],
    config: dict,
    device: torch.device,
    micro_batch_size: int,
) -> dict:
    if micro_batch_size <= 0:
        raise ValueError("cache micro-batch size must be positive")
    parts = [
        _cache_batch(model, samples[start : start + micro_batch_size], config, device)
        for start in range(0, len(samples), micro_batch_size)
    ]
    return _concatenate_cached(parts)


def _panel(array: np.ndarray, title: str, size: int = 288) -> Image.Image:
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    image = Image.fromarray(array.astype(np.uint8))
    image = image.resize((size, size), Image.Resampling.NEAREST)
    output = Image.new("RGB", (size, size + 24), "white")
    output.paste(image, (0, 24))
    ImageDraw.Draw(output).text((6, 5), title, fill="black")
    return output


def _probability_rgb(probability: torch.Tensor) -> np.ndarray:
    value = probability.detach().float().cpu().clamp(0, 1).numpy()
    red = (255.0 * value).astype(np.uint8)
    blue = (255.0 * (1.0 - value)).astype(np.uint8)
    green = (255.0 * (1.0 - np.abs(2.0 * value - 1.0))).astype(np.uint8)
    return np.stack((red, green, blue), axis=-1)


def _save_visuals(
    output_dir: Path,
    split: str,
    step: int,
    cached: dict,
    prediction: dict,
    maximum: int,
) -> None:
    directory = output_dir / "visuals" / split
    directory.mkdir(parents=True, exist_ok=True)
    count = min(maximum, cached["rgb"].shape[0])
    for sample in range(count):
        rgb = (
            cached["rgb"][sample].permute(1, 2, 0).numpy().clip(0, 1) * 255.0
        ).astype(np.uint8)
        merged = prediction["merged_bev"]
        latest = prediction["latest_auxiliary_bev"]
        arrays = (
            (rgb, "latest RGB"),
            (
                cached["merged_target"]["complete_target"][sample].cpu().numpy(),
                "merged GT",
            ),
            (
                merged["fov_complete_semantic"][sample].cpu().numpy(),
                "merged prediction",
            ),
            (
                _probability_rgb(merged["occupancy_probability"][sample]),
                "merged P(occupied)",
            ),
            (
                cached["latest_target"]["complete_target"][sample].cpu().numpy(),
                "latest GT",
            ),
            (
                latest["fov_complete_semantic"][sample].cpu().numpy(),
                "latest prediction",
            ),
        )
        panels = [_panel(array, title) for array, title in arrays]
        width = panels[0].width * 3
        height = panels[0].height * 2
        canvas = Image.new("RGB", (width, height), (230, 230, 230))
        for index, panel in enumerate(panels):
            canvas.paste(panel, ((index % 3) * panel.width, (index // 3) * panel.height))
        canvas.save(directory / f"step_{step:04d}_sample_{sample:02d}.png")


@torch.no_grad()
def _evaluate(
    model,
    cached: dict,
    output_dir: Path,
    split: str,
    step: int,
    maximum_visual_samples: int,
) -> dict:
    model.eval()
    device = cached["extraction"]["camera_register_tokens"].device
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        enabled=device.type == "cuda",
    ):
        prediction = model.forward_head(
            cached["extraction"],
            include_merged=True,
            include_scale=True,
            include_latest_auxiliary=True,
            assemble_runtime_outputs=True,
        )
    merged_records, merged_aggregate = _sample_metrics(
        prediction["merged_bev"],
        cached["merged_target"],
        prediction["scale"],
        cached["scale_target"],
    )
    latest_records, latest_aggregate = _sample_metrics(
        prediction["latest_auxiliary_bev"],
        cached["latest_target"],
        prediction["scale"],
        cached["scale_target"],
    )
    _save_visuals(
        output_dir,
        split,
        step,
        cached,
        prediction,
        maximum_visual_samples,
    )
    valid = cached["merged_target"]["gt_valid_mask"].bool()
    outside = ~valid
    outside_count = outside.sum().clamp_min(1)
    outside_support_mean = (
        prediction["merged_bev"]["fov_support_probability"].float()
        * outside.float()
    ).sum() / outside_count
    outside_occupancy_mean = (
        prediction["merged_bev"]["guessed"]["occupancy_probability"].float()
        * outside.float()
    ).sum() / outside_count
    return {
        "step": step,
        "split": split,
        "sample_ids": [item["sample_id"] for item in cached["metadata"]],
        "merged": _summarize(
            merged_records,
            merged_aggregate["calibration"].cpu().tolist(),
            include_groups=False,
        ),
        "latest": _summarize(
            latest_records,
            latest_aggregate["calibration"].cpu().tolist(),
            include_groups=False,
        ),
        "history_gate_mean": float(
            prediction["history_update_gate_mean"].float().mean().cpu()
        ),
        "coordinate_coverage_fraction": float(
            cached["merged_target"]["coordinate_coverage_fraction"].mean().cpu()
        ),
        "metric_extent_m": float(
            cached["merged_target"]["metric_extent_m"].mean().cpu()
        ),
        "outside_gt_support_probability_mean": float(
            outside_support_mean.cpu()
        ),
        "outside_gt_occupancy_probability_mean": float(
            outside_occupancy_mean.cpu()
        ),
    }


def main() -> None:
    args = _arguments()
    if args.steps <= 0 or args.train_samples <= 0 or args.holdout_samples <= 0:
        raise ValueError("steps and sample counts must be positive")
    if args.cache_micro_batch_size <= 0:
        raise ValueError("cache micro-batch size must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("optimizer arguments are invalid")
    eval_steps = {int(value) for value in args.eval_steps.split(",") if value.strip()}
    eval_steps.update((0, args.steps))
    if min(eval_steps) < 0 or max(eval_steps) > args.steps:
        raise ValueError("eval steps must fall inside [0,steps]")

    config = load_m05_config(args.config)
    if args.history_gate_initial_bias is not None:
        config["model"]["history_gate_initial_bias"] = (
            args.history_gate_initial_bias
        )
    config["training"]["observed_gate_boundary_emphasis"] = (
        args.gate_boundary_emphasis
    )
    config["training"]["support_boundary_emphasis"] = (
        args.support_boundary_emphasis
    )
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    train_dataset, holdout_dataset = build_m05_datasets(config)
    model = build_model(config, device)
    torch.cuda.reset_peak_memory_stats(device)
    for parameter in model.unwrapped_head().parameters():
        parameter.requires_grad_(True)
    train = _cache_samples(
        model,
        _fixed_samples(
            train_dataset,
            args.train_samples,
            args.history,
            args.seed,
        ),
        config,
        device,
        args.cache_micro_batch_size,
    )
    holdout = _cache_samples(
        model,
        _fixed_samples(
            holdout_dataset,
            args.holdout_samples,
            args.history,
            args.seed + 1,
        ),
        config,
        device,
        args.cache_micro_batch_size,
    )
    del train_dataset, holdout_dataset

    scale_prefixes = (
        "scale_token_projector.",
        "scale_frame_reliability.",
        "scale_decoder.",
    )
    named = list(model.unwrapped_head().named_parameters())
    scale_parameters = [p for name, p in named if name.startswith(scale_prefixes)]
    bev_parameters = [p for name, p in named if not name.startswith(scale_prefixes)]
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        **vars(args),
        "config": str(args.config.expanduser().resolve()),
        "output_dir": str(output_dir),
        "eval_steps": sorted(eval_steps),
        "cached_frozen_vggt": True,
        "runtime_model_unchanged": True,
        "sample_selection": "deterministic_round_robin_by_dataset_source",
        "train_datasets": [item["dataset"] for item in train["metadata"]],
        "holdout_datasets": [item["dataset"] for item in holdout["metadata"]],
    }
    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    metrics_path = output_dir / "metrics.jsonl"
    training_path = output_dir / "training.jsonl"
    weights = m05_loss_weights(config["training"])
    started = time.monotonic()

    def evaluate(step: int) -> None:
        with metrics_path.open("a", encoding="utf-8") as stream:
            for split, cached in (("train", train), ("holdout", holdout)):
                result = _evaluate(
                    model,
                    cached,
                    output_dir,
                    split,
                    step,
                    args.maximum_visual_samples,
                )
                stream.write(json.dumps(result, sort_keys=True) + "\n")
                print(
                    json.dumps(
                        {
                            "event": "evaluation",
                            "step": step,
                            "split": split,
                            "merged_fused_iou": result["merged"]["pixel_metrics"][
                                "fused_occupied"
                            ]["iou"],
                            "merged_support_boundary_f1_r2": result["merged"][
                                "boundary_metrics"
                            ]["support"]["radius_2_px"]["f1"],
                            "scale_relative_error": result["merged"]["scale"][
                                "relative_error_mean"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    evaluate(0)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        use_bf16 = torch.cuda.is_bf16_supported()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            prediction = model.forward_head(
                train["extraction"],
                include_merged=True,
                include_scale=True,
                include_latest_auxiliary=True,
                assemble_runtime_outputs=False,
            )
            bev = m05_bev_loss(
                prediction["merged_bev"],
                prediction["latest_auxiliary_bev"],
                train["merged_target"],
                train["latest_target"],
                weights=weights,
                global_step=step - 1,
                total_steps=args.steps,
                wrong_evidence_zero_fraction=float(
                    config["training"]["wrong_evidence_zero_fraction"]
                ),
                wrong_evidence_ramp_fraction=float(
                    config["training"]["wrong_evidence_ramp_fraction"]
                ),
                hidden_occupied_zero_fraction=float(
                    config["training"]["hidden_occupied_zero_fraction"]
                ),
                hidden_occupied_ramp_fraction=float(
                    config["training"]["hidden_occupied_ramp_fraction"]
                ),
                latest_auxiliary_weight=args.latest_auxiliary_weight,
            )
            scale = m04_scale_loss(
                prediction["scale"],
                train["scale_target"],
                degrees_of_freedom=float(
                    config["training"]["scale_student_t_degrees_of_freedom"]
                ),
                minimum_sigma_log=float(
                    config["training"]["scale_minimum_sigma_log"]
                ),
                maximum_sigma_log=float(
                    config["training"]["scale_maximum_sigma_log"]
                ),
            )
            loss = bev["loss"] + scale["loss"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("M05 overfit diagnostic produced non-finite loss")
        loss.backward()
        clip = float(config["training"]["gradient_clip_norm"])
        bev_norm = torch.nn.utils.clip_grad_norm_(
            bev_parameters, clip, foreach=True
        )
        scale_norm = torch.nn.utils.clip_grad_norm_(
            scale_parameters, clip, foreach=True
        )
        optimizer.step()
        with training_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "bev_loss": float(bev["loss"].detach().cpu()),
                        "merged_loss": float(bev["merged_loss"].detach().cpu()),
                        "latest_loss": float(
                            bev["latest_auxiliary_loss"].detach().cpu()
                        ),
                        "scale_loss": float(scale["loss"].detach().cpu()),
                        "bev_gradient_norm": float(bev_norm.detach().cpu()),
                        "scale_gradient_norm": float(scale_norm.detach().cpu()),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        if step in eval_steps:
            evaluate(step)

    print(
        json.dumps(
            {
                "event": "complete",
                "steps": args.steps,
                "elapsed_seconds": time.monotonic() - started,
                "peak_allocated_gib": (
                    torch.cuda.max_memory_allocated(device) / 2**30
                ),
                "peak_reserved_gib": (
                    torch.cuda.max_memory_reserved(device) / 2**30
                ),
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
