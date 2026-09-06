#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from vggt_bev_method1.cli_train_m05 import (
    _configure_m05_execution,
    _forward_losses,
)
from vggt_bev_method1.cli_train_m05 import build_model as build_m05_model
from vggt_bev_method1.cli_train_m05_plus import build_model as build_m05_plus_model
from vggt_bev_method1.cli_train_metric import _configure_head_compilation
from vggt_bev_method1.data import RGBResizePad, VGGNAVMethod1Dataset
from vggt_bev_method1.m05_config import load_m05_config
from vggt_bev_method1.m05_plus_config import load_m05_plus_config
from vggt_bev_method1.m05_train_utils import build_m05_datasets, m05_collate
from vggt_bev_method1.train_utils import move_batch


def _timed_cuda_call(totals: dict[str, float], name: str, function, *args, **kwargs):
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = function(*args, **kwargs)
    torch.cuda.synchronize()
    totals[name] = totals.get(name, 0.0) + time.perf_counter() - started
    return output


def _sample_with_history(dataset, history: int) -> dict:
    for index, record in enumerate(dataset.samples):
        if record.target_frame + 1 == history:
            return dataset[index]
    raise ValueError(f"no training sample has exactly {history} history frames")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Short real-data M05 forward/backward benchmark without checkpoints"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--m05-plus",
        action="store_true",
        help="Load the M05+ contract and model instead of the legacy M05 model",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override the configured dataset root for a diagnostic manifest",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Override the configured split manifest for a diagnostic run",
    )
    parser.add_argument(
        "--session-selection-order",
        help="Override the configured session order for a diagnostic manifest",
    )
    parser.add_argument(
        "--maximum-sessions",
        type=int,
        help="Match a diagnostic manifest that freezes a finite prefix",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        help="Override the configured split seed for a diagnostic manifest",
    )
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--history", required=True, type=int)
    parser.add_argument(
        "--session-key",
        help="Load one diagnostic session directly, without a split manifest",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--query-chunk-size", type=int)
    parser.add_argument("--history-proposal-batch-size", type=int)
    parser.add_argument(
        "--attention-checkpoint-fraction",
        type=float,
        default=1.0,
        help="Fraction of deformable-query chunks rematerialized in backward",
    )
    parser.add_argument("--profile-components", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.history <= 0 or args.steps <= 0:
        raise ValueError("batch size, history, and steps must be positive")
    if not 0.0 <= args.attention_checkpoint_fraction <= 1.0:
        raise ValueError("attention checkpoint fraction must be in [0,1]")

    if args.m05_plus:
        config = load_m05_plus_config(args.config)
        build_model = build_m05_plus_model
    else:
        config = load_m05_config(args.config)
        build_model = build_m05_model
    if args.data_root is not None:
        config["data"]["root"] = str(args.data_root)
    if args.split_manifest is not None:
        config["data"]["split_manifest"] = str(args.split_manifest)
    if args.session_selection_order is not None:
        config["data"]["session_selection_order"] = args.session_selection_order
    if args.maximum_sessions is not None:
        if args.maximum_sessions <= 0:
            raise ValueError("maximum sessions must be positive")
        config["data"]["maximum_sessions"] = args.maximum_sessions
    if args.split_seed is not None:
        config["data"]["split_seed"] = args.split_seed
    if args.query_chunk_size is not None:
        if args.query_chunk_size <= 0:
            raise ValueError("query chunk size must be positive")
        config["model"]["cross_query_chunk_size"] = args.query_chunk_size
    if args.history_proposal_batch_size is not None:
        if args.history_proposal_batch_size <= 0:
            raise ValueError("history proposal batch size must be positive")
        config["model"]["history_proposal_batch_size"] = (
            args.history_proposal_batch_size
        )
    # Compilation is deliberately disabled here: the server's current Inductor
    # stack is not stable, and eager BF16 is the production-safe comparison.
    config["training"]["compile_head"] = False

    if args.session_key is None:
        train_dataset, _ = build_m05_datasets(config, verify_manifest=False)
    else:
        data = config["data"]
        train_dataset = VGGNAVMethod1Dataset(
            root=data["root"],
            supervision=data["supervision"],
            preprocess=RGBResizePad(
                int(data["image_height"]), int(data["image_width"])
            ),
            session_keys=[args.session_key],
            sample_stride=int(data["sample_stride"]),
            minimum_history=int(data["minimum_history"]),
            maximum_history=int(data["maximum_history"]),
            merged_source_extent_m=float(data["merged_source_extent_m"]),
            merged_source_image_size=int(data["merged_source_image_size"]),
            merged_complete_directory=str(data["merged_complete_directory"]),
            merged_masked_directory=str(data["merged_masked_directory"]),
            merged_bev_extent_m=float(data["merged_source_extent_m"]),
            merged_bev_output_size=int(data["merged_source_output_size"]),
            include_single_targets=False,
            include_latest_temporal_targets=True,
        )
    sample = _sample_with_history(train_dataset, args.history)
    batch = m05_collate([sample for _ in range(args.batch_size)])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl")
    primary = not distributed or dist.get_rank() == 0
    model = build_model(config, device)
    for parameter in model.unwrapped_head().parameters():
        parameter.requires_grad_(True)
    config["training"]["attention_checkpoint_fraction"] = (
        args.attention_checkpoint_fraction
    )
    execution = _configure_m05_execution(model, config["training"])
    compilation = _configure_head_compilation(model, config["training"])
    if distributed:
        model.head = DistributedDataParallel(
            model.head,
            device_ids=[device.index],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
        )

    named_trainable = list(model.unwrapped_head().named_parameters())
    scale_prefixes = (
        "scale_token_projector.",
        "scale_frame_reliability.",
        "scale_decoder.",
    )
    scale_parameters = [
        parameter
        for name, parameter in named_trainable
        if name.startswith(scale_prefixes)
    ]
    bev_parameters = [
        parameter
        for name, parameter in named_trainable
        if not name.startswith(scale_prefixes)
    ]
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.02)),
        fused=True,
    )
    batch = move_batch(batch, device)
    del train_dataset, sample
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()

    step_seconds: list[float] = []
    component_seconds: dict[str, float] = {}
    values = None
    loss = None
    for step in range(args.steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        should_profile = args.profile_components and step > 0
        if should_profile:
            original_extract = model.extract
            original_teacher = model.decode_scale_teacher
            original_head = model.forward_head

            def timed_extract(
                _self,
                *call_args,
                _original=original_extract,
                **call_kwargs,
            ):
                return _timed_cuda_call(
                    component_seconds,
                    "vggt_aggregate",
                    _original,
                    *call_args,
                    **call_kwargs,
                )

            def timed_teacher(
                _self,
                *call_args,
                _original=original_teacher,
                **call_kwargs,
            ):
                return _timed_cuda_call(
                    component_seconds,
                    "vggt_depth_teacher",
                    _original,
                    *call_args,
                    **call_kwargs,
                )

            def timed_head(
                _self,
                *call_args,
                _original=original_head,
                **call_kwargs,
            ):
                return _timed_cuda_call(
                    component_seconds,
                    "m05_head",
                    _original,
                    *call_args,
                    **call_kwargs,
                )

            model.extract = MethodType(timed_extract, model)
            model.decode_scale_teacher = MethodType(timed_teacher, model)
            model.forward_head = MethodType(timed_head, model)
            torch.cuda.synchronize()
            forward_started = time.perf_counter()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, values = _forward_losses(
                model,
                batch,
                config,
                global_step=step,
                total_steps=max(args.steps, 1),
            )
        if should_profile:
            torch.cuda.synchronize()
            component_seconds["forward_total"] = component_seconds.get(
                "forward_total", 0.0
            ) + time.perf_counter() - forward_started
            model.extract = original_extract
            model.decode_scale_teacher = original_teacher
            model.forward_head = original_head
            _timed_cuda_call(component_seconds, "backward", loss.backward)
            _timed_cuda_call(
                component_seconds,
                "bev_gradient_clip",
                torch.nn.utils.clip_grad_norm_,
                bev_parameters,
                float(config["training"]["gradient_clip_norm"]),
                foreach=True,
            )
            _timed_cuda_call(
                component_seconds,
                "scale_gradient_clip",
                torch.nn.utils.clip_grad_norm_,
                scale_parameters,
                float(config["training"]["gradient_clip_norm"]),
                foreach=True,
            )
            _timed_cuda_call(component_seconds, "optimizer", optimizer.step)
        else:
            loss.backward()
            clip = float(config["training"]["gradient_clip_norm"])
            torch.nn.utils.clip_grad_norm_(bev_parameters, clip, foreach=True)
            torch.nn.utils.clip_grad_norm_(scale_parameters, clip, foreach=True)
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        step_seconds.append(elapsed)
        if primary:
            print(
                json.dumps(
                    {
                        "event": "step",
                        "step": step + 1,
                        "seconds": elapsed,
                        "loss": float(loss.detach().cpu()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    assert loss is not None and values is not None
    measured_steps = max(args.steps - 1, 1)
    steady = step_seconds[1:] if len(step_seconds) > 1 else step_seconds
    mean_steady = sum(steady) / len(steady)
    if primary:
        print(
            json.dumps(
            {
                "event": "summary",
                "batch_size": args.batch_size,
                "history": args.history,
                "steps": args.steps,
                "loss": float(loss.detach().cpu()),
                "finite": bool(torch.isfinite(loss)),
                "first_step_seconds": step_seconds[0],
                "mean_steady_step_seconds": mean_steady,
                "steady_samples_per_second": args.batch_size / mean_steady,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                "query_chunk_size": int(config["model"]["cross_query_chunk_size"]),
                "attention_checkpoint_fraction": args.attention_checkpoint_fraction,
                "deformable_attention_modules": execution["attention_modules"],
                "execution": execution,
                "effective_supervision": float(
                    values["target_effective_supervision_fraction"].detach().cpu()
                ),
                "history_update_gate_mean": float(
                    values["history_update_gate_mean"].detach().cpu()
                ),
                "head_compilation": compilation,
                "mean_profiled_component_seconds": {
                    key: value / measured_steps
                    for key, value in sorted(component_seconds.items())
                },
            },
            sort_keys=True,
        ),
            flush=True,
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
