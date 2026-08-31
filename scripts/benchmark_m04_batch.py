#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from types import MethodType
from pathlib import Path

import torch

from vggt_bev_method1.cli_train_m04 import (
    _configure_head_compilation,
    _forward_losses,
    _set_trainable_head,
    build_model,
)
from vggt_bev_method1.data import RGBResizePad, VGGNAVMethod1Dataset
from vggt_bev_method1.m04_config import load_m04_config
from vggt_bev_method1.m04_train_utils import m04_collate
from vggt_bev_method1.train_utils import move_batch


def _timed_cuda_call(totals: dict[str, float], name: str, function, *args, **kwargs):
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = function(*args, **kwargs)
    torch.cuda.synchronize()
    totals[name] = totals.get(name, 0.0) + time.perf_counter() - started
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One worst-history M04 train step for batch-size selection"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--session-key", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--query-chunk-size", type=int)
    parser.add_argument("--compile-head", action="store_true")
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument("--torch-profile", type=Path)
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
    )
    arguments = parser.parse_args()
    if arguments.batch_size <= 0 or arguments.history <= 0 or arguments.steps <= 0:
        raise ValueError("batch size, history, and steps must be positive")

    config = load_m04_config(arguments.config)
    if arguments.query_chunk_size is not None:
        if arguments.query_chunk_size <= 0:
            raise ValueError("query chunk size must be positive")
        config["model"]["cross_query_chunk_size"] = arguments.query_chunk_size
    data = config["data"]
    dataset = VGGNAVMethod1Dataset(
        data["root"],
        supervision=data["supervision"],
        preprocess=RGBResizePad(int(data["image_height"]), int(data["image_width"])),
        session_keys=[arguments.session_key],
        minimum_history=arguments.history,
        maximum_history=arguments.history,
        merged_source_extent_m=float(data["merged_source_extent_m"]),
        merged_source_image_size=int(data["merged_source_image_size"]),
        merged_complete_directory=str(data["merged_complete_directory"]),
        merged_masked_directory=str(data["merged_masked_directory"]),
        merged_bev_extent_m=float(data["merged_source_extent_m"]),
        merged_bev_output_size=int(data["merged_source_output_size"]),
        include_single_targets=False,
    )
    sample = dataset[0]
    batch = m04_collate([sample for _ in range(arguments.batch_size)])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = build_model(config, device)
    _set_trainable_head(model)
    if arguments.compile_head:
        config["training"].update(
            {
                "compile_head": True,
                "compile_head_scope": "deformable_query_chunks",
                "compile_head_backend": "inductor",
                "compile_head_mode": arguments.compile_mode,
                "compile_head_dynamic": True,
            }
        )
    compilation = _configure_head_compilation(model, config["training"])
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.02)),
        fused=True,
    )
    batch = move_batch(batch, device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    step_seconds = []
    component_seconds: dict[str, float] = {}
    values = None
    loss = None
    for step in range(arguments.steps):
        operation_profiler = None
        if arguments.torch_profile is not None and step == 1:
            operation_profiler = torch.profiler.profile(
                activities=(
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            operation_profiler.start()
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        should_profile = arguments.profile_components and step > 0
        if should_profile:
            original_extract = model.extract
            original_teacher = model.decode_scale_teacher
            original_head = model.forward_head

            def timed_extract(_self, *args, **kwargs):
                return _timed_cuda_call(
                    component_seconds, "vggt_aggregate", original_extract,
                    *args, **kwargs,
                )

            def timed_teacher(_self, *args, **kwargs):
                return _timed_cuda_call(
                    component_seconds, "vggt_depth_teacher", original_teacher,
                    *args, **kwargs,
                )

            def timed_head(_self, *args, **kwargs):
                return _timed_cuda_call(
                    component_seconds, "m04_head", original_head,
                    *args, **kwargs,
                )

            model.extract = MethodType(timed_extract, model)
            model.decode_scale_teacher = MethodType(timed_teacher, model)
            model.forward_head = MethodType(timed_head, model)
            torch.cuda.synchronize()
            forward_started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, values = _forward_losses(model, batch, config)
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
                "gradient_clip",
                torch.nn.utils.clip_grad_norm_,
                model.head.parameters(),
                1.0,
                foreach=True,
            )
            _timed_cuda_call(component_seconds, "optimizer", optimizer.step)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.head.parameters(), 1.0, foreach=True
            )
            optimizer.step()
        if operation_profiler is not None:
            operation_profiler.stop()
            arguments.torch_profile.parent.mkdir(parents=True, exist_ok=True)
            operation_profiler.export_chrome_trace(str(arguments.torch_profile))
            print(
                operation_profiler.key_averages().table(
                    sort_by="self_cuda_time_total",
                    row_limit=40,
                ),
                flush=True,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        step_seconds.append(elapsed)
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
    steady = step_seconds[1:] if len(step_seconds) > 1 else step_seconds
    mean_steady_seconds = sum(steady) / len(steady)
    print(
        json.dumps(
            {
                "batch_size": arguments.batch_size,
                "history": arguments.history,
                "loss": float(loss.detach().cpu()),
                "finite": bool(torch.isfinite(loss)),
                "steps": arguments.steps,
                "first_step_seconds": step_seconds[0],
                "mean_steady_step_seconds": mean_steady_seconds,
                "steady_samples_per_second": (
                    arguments.batch_size / mean_steady_seconds
                ),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / 2**30,
                "query_chunk_size": int(
                    config["model"]["cross_query_chunk_size"]
                ),
                "effective_supervision": float(
                    values["target_effective_supervision_fraction"].detach().cpu()
                ),
                "head_compilation": compilation,
                "mean_profiled_component_seconds": {
                    key: value / max(arguments.steps - 1, 1)
                    for key, value in sorted(component_seconds.items())
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
