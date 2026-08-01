from __future__ import annotations

import argparse
import json
import os
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vggt_bev.data import (
    CalibrationAwareResize,
    VGGNAVMethod2Dataset,
    method2_collate,
    split_sessions,
)
from vggt_bev.losses import Method2LossWeights, method2_loss
from vggt_bev.metrics import method2_metrics
from vggt_bev.models import FrozenVGGTAdapter, Method2BEVHead, Method2System


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def _initialize_distributed(device_name: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(False, 0, 0, 1, "none")
    if not device_name.startswith("cuda"):
        raise ValueError("distributed Method II training requires --device cuda")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    backend = os.environ.get("VGGTBEV_DISTRIBUTED_BACKEND", "nccl")
    if backend not in {"nccl", "gloo"}:
        raise ValueError(
            "VGGTBEV_DISTRIBUTED_BACKEND must be 'nccl' or 'gloo'"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(True, rank, local_rank, world_size, backend)


def _distributed_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    reduced = value.detach().clone()
    if context.enabled:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= context.world_size
    return reduced


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Method II observed-area BEV heads")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/method2_observed.toml"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="build and inspect aligned train/validation datasets without loading VGGT",
    )
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--batch-size-per-gpu",
        type=int,
        default=None,
        help="override training.batch_size_per_gpu",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override training.output_dir (useful for an isolated smoke run)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="skip the validation epoch, useful only for a one-step integration smoke test",
    )
    return parser


def _read_config(path: Path) -> dict:
    with path.expanduser().resolve().open("rb") as file:
        config = tomllib.load(file)
    for section in ("data", "model", "training"):
        if section not in config:
            raise ValueError(f"configuration is missing [{section}]")
    geometry_source = config["data"].get("geometry_source", "vggt")
    if geometry_source != "vggt":
        raise ValueError(
            "Method II training now requires data.geometry_source='vggt'; "
            "simulator GT geometry is label/audit data only"
        )
    config["data"]["geometry_source"] = "vggt"
    vggt_execution = config["model"].get("vggt_execution", "live")
    if vggt_execution != "live":
        raise ValueError(
            "Method II training requires model.vggt_execution='live'; "
            "cached simulator or VGGT geometry inputs are not accepted"
        )
    config["model"]["vggt_execution"] = "live"
    metric_scale_mode = config["model"].get(
        "metric_scale_mode",
        "learned_global",
    )
    if metric_scale_mode in (
        "camera_height",
        "vggt_raw",
        "vggt_normalized",
    ):
        if config["model"].get("learn_depth_scale", False):
            raise ValueError(
                f"{metric_scale_mode} requires "
                "model.learn_depth_scale=false"
            )
        if not config["model"].get("stabilize_intrinsics", False):
            raise ValueError(
                "strict Method II requires model.stabilize_intrinsics=true"
            )
    coordinate_mode = config["data"].get("coordinate_mode", "metric")
    if metric_scale_mode in ("vggt_raw", "vggt_normalized"):
        if coordinate_mode != metric_scale_mode:
            raise ValueError(
                f"model.metric_scale_mode={metric_scale_mode!r} requires "
                f"data.coordinate_mode={metric_scale_mode!r}"
            )
    elif coordinate_mode in ("vggt_raw", "vggt_normalized"):
        raise ValueError(
            f"data.coordinate_mode={coordinate_mode!r} requires the same "
            "model.metric_scale_mode"
        )
    if metric_scale_mode == "vggt_normalized":
        single_size = int(config["model"].get("single_output_size", 512))
        merged_size = int(config["model"].get("merged_output_size", 800))
        if single_size != int(config["data"].get("single_target_size", 512)):
            raise ValueError(
                "single target and model output sizes must match"
            )
        if merged_size != int(config["data"].get("merged_target_size", 800)):
            raise ValueError(
                "merged target and model output sizes must match"
            )
    return config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_dataset(config: dict, session_names: list[str]) -> VGGNAVMethod2Dataset:
    data = config["data"]
    return VGGNAVMethod2Dataset(
        data["root"],
        extent_key=data["extent_key"],
        target_mode=data["target_mode"],
        preprocess=CalibrationAwareResize(data["image_height"], data["image_width"]),
        session_names=session_names,
        sample_stride=data.get("sample_stride", 1),
        include_revealed_for_audit=False,
        geometry_source=data["geometry_source"],
        coordinate_mode=data.get("coordinate_mode", "metric"),
        expected_merged_extent_m=data.get("expected_merged_extent_m"),
        required_merged_fusion_version=data.get(
            "required_merged_fusion_version"
        ),
        validate_paths_on_init=bool(
            data.get("validate_paths_on_init", True)
        ),
        single_target_size=data.get("single_target_size"),
        merged_target_size=data.get("merged_target_size"),
    )


def _move_batch(batch: dict, device: torch.device) -> dict:
    forbidden = {
        key for key in batch if "revealed_labels" in key or "complete" in key
    }
    if forbidden:
        raise RuntimeError(f"omniscient labels are forbidden in the training batch: {forbidden}")
    gt_geometry = {
        "intrinsics",
        "camera_to_world",
        "reference_world_from_bev",
        "floor_y",
    } & batch.keys()
    if gt_geometry:
        raise RuntimeError(
            "simulator GT geometry is forbidden in the training batch: "
            f"{sorted(gt_geometry)}"
        )
    if batch.get("use_predicted_geometry") is not True:
        raise RuntimeError(
            "training batches must enable VGGT-predicted geometry"
        )
    coordinate_mode = batch.get("coordinate_mode", "metric")
    if coordinate_mode in ("vggt_raw", "vggt_normalized"):
        if "camera_height_m" in batch:
            raise RuntimeError(
                "VGGT non-metric training forbids camera_height_m in the "
                "model batch"
            )
        if coordinate_mode == "vggt_normalized":
            forbidden_extents = {
                "target_extent_m",
                "single_target_extent_m",
                "merged_target_extent_m",
            } & batch.keys()
            if forbidden_extents:
                raise RuntimeError(
                    "normalized-VGGT training forbids fixed model extents: "
                    f"{sorted(forbidden_extents)}"
                )
    elif "camera_height_m" not in batch:
        raise RuntimeError(
            "metric VGGT geometry requires physical camera_height_m"
        )
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _compute_losses(
    predictions: dict,
    batch: dict,
    loss_weights: Method2LossWeights,
    target_mode: str,
    *,
    single_task_weight: float,
    merged_task_weight: float,
) -> dict[str, torch.Tensor]:
    if target_mode != "both":
        return method2_loss(predictions, batch["target_labels"], weights=loss_weights)

    single = method2_loss(
        predictions["single"],
        batch["single_target_labels"],
        weights=loss_weights,
    )
    merged = method2_loss(
        predictions["merged"],
        batch["merged_target_labels"],
        weights=loss_weights,
    )
    return {
        "loss": single_task_weight * single["loss"] + merged_task_weight * merged["loss"],
        **{f"single_{name}": value for name, value in single.items()},
        **{f"merged_{name}": value for name, value in merged.items()},
    }


def _compute_metrics(predictions: dict, batch: dict, target_mode: str) -> dict[str, torch.Tensor]:
    if target_mode != "both":
        return method2_metrics(predictions, batch["target_labels"])
    metrics: dict[str, torch.Tensor] = {}
    for task in ("single", "merged"):
        task_metrics = method2_metrics(
            predictions[task],
            batch[f"{task}_target_labels"],
        )
        metrics.update({f"{task}_{name}": value for name, value in task_metrics.items()})
    return metrics


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_weights: Method2LossWeights,
    target_mode: str,
    single_task_weight: float,
    merged_task_weight: float,
    context: DistributedContext,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            predictions = model(batch)
            losses = _compute_losses(
                predictions,
                batch,
                loss_weights,
                target_mode,
                single_task_weight=single_task_weight,
                merged_task_weight=merged_task_weight,
            )
            metrics = _compute_metrics(predictions, batch, target_mode)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1
    names = sorted(totals)
    values = torch.tensor(
        [*(totals[name] for name in names), float(batches)],
        dtype=torch.float64,
        device=device,
    )
    if context.enabled:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_batches = max(float(values[-1]), 1.0)
    return {
        name: float(values[index]) / total_batches
        for index, name in enumerate(names)
    }


def _save_checkpoint(
    path: Path,
    model: Method2System,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: dict,
    *,
    epoch_complete: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Save only trainable Method II state; the frozen 4.4 GB VGGT checkpoint stays external.
    torch.save(
        {
            "epoch": epoch,
            "epoch_complete": epoch_complete,
            "global_step": global_step,
            "head": model.head.state_dict(),
            "merged_head": (
                model.merged_head.state_dict() if model.merged_head is not None else None
            ),
            "normalizer": model.normalizer.state_dict(),
            "log_depth_scale": model.log_depth_scale.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        temporary,
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    model: Method2System,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    state = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    model.head.load_state_dict(state["head"], strict=True)
    if model.merged_head is not None:
        if state.get("merged_head") is None:
            raise ValueError("dual-output model cannot resume from a single-output checkpoint")
        model.merged_head.load_state_dict(state["merged_head"], strict=True)
    if model.metric_scale_mode == "vggt_normalized":
        if state.get("normalizer") is None:
            raise ValueError(
                "normalized-VGGT model cannot resume without normalizer state"
            )
        model.normalizer.load_state_dict(state["normalizer"], strict=True)
    model.log_depth_scale.data.copy_(state["log_depth_scale"].to(model.log_depth_scale.device))
    optimizer.load_state_dict(state["optimizer"])
    start_epoch = int(state["epoch"])
    if bool(state.get("epoch_complete", False)):
        start_epoch += 1
    return start_epoch, int(state.get("global_step", 0))


def run(args: argparse.Namespace) -> None:
    config = _read_config(args.config)
    training = config["training"]
    model_config = config["model"]
    context = _initialize_distributed(args.device)
    batch_size_per_gpu = (
        args.batch_size_per_gpu
        if args.batch_size_per_gpu is not None
        else int(training.get("batch_size_per_gpu", 1))
    )
    if batch_size_per_gpu <= 0:
        raise ValueError("batch size per GPU must be positive")
    if args.num_workers < 0:
        raise ValueError("num workers must be non-negative")
    seed = int(training.get("seed", 7))
    _set_seed(seed + context.rank)

    data_config = config["data"]
    train_sessions, validation_sessions = split_sessions(
        data_config["root"],
        validation_fraction=float(
            data_config.get("validation_fraction", 0.2)
        ),
        seed=seed,
        group_by=data_config.get("split_group", "scene"),
        strategy=data_config.get("split_strategy", "stable_hash"),
    )
    train_dataset = _make_dataset(config, train_sessions)
    validation_dataset = _make_dataset(config, validation_sessions)
    dataset_report = {
        "train_session_count": len(train_sessions),
        "validation_session_count": len(validation_sessions),
        "train_session_examples": train_sessions[:3],
        "validation_session_examples": validation_sessions[:3],
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "target_mode": config["data"]["target_mode"],
        "extent_key": config["data"]["extent_key"],
        "geometry_source": config["data"]["geometry_source"],
        "coordinate_mode": config["data"].get(
            "coordinate_mode",
            "metric",
        ),
        "expected_merged_extent_m": config["data"].get(
            "expected_merged_extent_m"
        ),
        "single_target_size": config["data"].get("single_target_size"),
        "merged_target_size": config["data"].get("merged_target_size"),
        "required_merged_fusion_version": config["data"].get(
            "required_merged_fusion_version"
        ),
        "validate_paths_on_init": bool(
            config["data"].get("validate_paths_on_init", True)
        ),
        "split_group": data_config.get("split_group", "scene"),
        "split_strategy": data_config.get(
            "split_strategy",
            "stable_hash",
        ),
        "vggt_execution": model_config["vggt_execution"],
        "metric_scale_mode": model_config.get(
            "metric_scale_mode",
            "learned_global",
        ),
        "stabilize_intrinsics": bool(
            model_config.get("stabilize_intrinsics", False)
        ),
        "single_bev_feature_size": int(
            model_config.get(
                "single_bev_feature_size",
                model_config.get("bev_feature_size", 128),
            )
        ),
        "merged_bev_feature_size": int(
            model_config.get(
                "merged_bev_feature_size",
                model_config.get("bev_feature_size", 128),
            )
        ),
        "distributed_world_size": context.world_size,
        "distributed_backend": context.backend,
        "batch_size_per_gpu": batch_size_per_gpu,
        "effective_batch_size": batch_size_per_gpu * context.world_size,
        "num_workers_per_rank": args.num_workers,
    }
    if context.is_primary:
        print(json.dumps({"dataset": dataset_report}), flush=True)
    if args.data_only:
        if context.is_primary:
            sample = train_dataset[0]
            if config["data"]["target_mode"] == "both":
                target_report = {
                    "single": list(sample["single_target_labels"].shape),
                    "merged": list(sample["merged_target_labels"].shape),
                }
            else:
                target_report = list(sample["target_labels"].shape)
            print(
                json.dumps(
                    {
                        "first_sample": {
                            "images": list(sample["images"].shape),
                            "camera_height_m": (
                                float(sample["camera_height_m"])
                                if "camera_height_m" in sample
                                else None
                            ),
                            "contains_simulator_gt_geometry": bool(
                                {
                                    "intrinsics",
                                    "camera_to_world",
                                    "reference_world_from_bev",
                                    "floor_y",
                                }
                                & sample.keys()
                            ),
                            "target": target_report,
                            "metadata": sample["metadata"],
                        }
                    },
                    indent=2,
                ),
                flush=True,
            )
        return

    device = (
        torch.device("cuda", context.local_rank)
        if context.enabled
        else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; use --data-only for CPU validation"
        )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=seed,
        )
        if context.enabled
        else None
    )
    validation_sampler = (
        DistributedSampler(
            validation_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
        )
        if context.enabled
        else None
    )
    worker_options = (
        {
            "persistent_workers": True,
            "prefetch_factor": int(
                training.get("prefetch_factor", 2)
            ),
        }
        if args.num_workers > 0
        else {}
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=method2_collate,
        **worker_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        sampler=validation_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=method2_collate,
        **worker_options,
    )

    adapter = FrozenVGGTAdapter(
        model_config["vggt_source"],
        model_config["checkpoint"],
        device=device,
        patch_size=int(model_config.get("patch_size", 16)),
    )
    def build_head(output_size: int) -> Method2BEVHead:
        return Method2BEVHead(
            feature_dim=int(model_config.get("feature_dim", 2048)),
            hidden_dim=int(model_config.get("hidden_dim", 96)),
            output_size=output_size,
            filter_by_height=bool(
                model_config.get("filter_by_height", True)
            ),
            ray_steps=int(model_config.get("ray_steps", 32)),
        ).to(device)

    single_output_size = int(
        model_config.get(
            "single_output_size",
            model_config.get("output_size", 512),
        )
    )
    merged_output_size = int(
        model_config.get(
            "merged_output_size",
            model_config.get("output_size", 512),
        )
    )
    head = build_head(single_output_size)
    merged_head = (
        build_head(merged_output_size)
        if config["data"]["target_mode"] == "both"
        else None
    )
    model = Method2System(
        adapter,
        head,
        merged_head=merged_head,
        bev_feature_size=int(
            model_config.get(
                "single_bev_feature_size",
                model_config.get("bev_feature_size", 128),
            )
        ),
        merged_bev_feature_size=int(
            model_config.get(
                "merged_bev_feature_size",
                model_config.get("bev_feature_size", 128),
            )
        ),
        initial_depth_scale=float(model_config.get("initial_depth_scale", 1.0)),
        learn_depth_scale=bool(model_config.get("learn_depth_scale", True)),
        metric_scale_mode=model_config.get(
            "metric_scale_mode",
            "learned_global",
        ),
        normalizer_hidden_dim=int(
            model_config.get("normalizer_hidden_dim", 96)
        ),
        initial_normalized_single_span=float(
            model_config.get("initial_normalized_single_span", 2.0)
        ),
        stabilize_intrinsics=bool(
            model_config.get("stabilize_intrinsics", False)
        ),
        minimum_confidence=float(
            model_config.get("minimum_confidence", 0.05)
        ),
        minimum_depth_m=float(model_config.get("minimum_depth_m", 0.05)),
        maximum_depth_m=float(model_config.get("maximum_depth_m", 20.0)),
        ground_minimum_points=int(
            model_config.get("ground_minimum_points", 48)
        ),
        ground_candidate_quantile=float(
            model_config.get("ground_candidate_quantile", 0.55)
        ),
        ground_maximum_candidate_quantile=float(
            model_config.get(
                "ground_maximum_candidate_quantile",
                0.995,
            )
        ),
        ground_irls_iterations=int(
            model_config.get("ground_irls_iterations", 5)
        ),
        ground_huber_delta=float(
            model_config.get("ground_huber_delta", 2.5)
        ),
        ground_maximum_tilt_degrees=float(
            model_config.get("ground_maximum_tilt_degrees", 35.0)
        ),
    ).to(device)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    loss_weights = Method2LossWeights(
        occupancy=float(training.get("occupancy_weight", 1.0)),
        observation=float(training.get("observation_weight", 0.5)),
        dice=float(training.get("dice_weight", 0.25)),
    )
    single_task_weight = float(training.get("single_task_weight", 0.5))
    merged_task_weight = float(training.get("merged_task_weight", 0.5))
    if config["data"]["target_mode"] == "both" and (
        single_task_weight <= 0.0 or merged_task_weight <= 0.0
    ):
        raise ValueError("dual-output task weights must both be positive")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(training["output_dir"])
    ).expanduser().resolve()
    start_epoch = 1
    global_step = 0
    if args.resume is not None:
        start_epoch, global_step = _load_checkpoint(args.resume, model, optimizer)
        if context.is_primary:
            print(
                json.dumps(
                    {
                        "resumed_from": str(args.resume.expanduser().resolve()),
                        "start_epoch": start_epoch,
                        "global_step": global_step,
                    }
                ),
                flush=True,
            )
    training_model: torch.nn.Module = model
    if context.enabled:
        # VGGT is frozen and independently loaded from the same checkpoint on
        # every rank. Excluding it avoids a redundant multi-gigabyte startup
        # broadcast; DDP still synchronizes every trainable Method II parameter.
        ignored = [
            name
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        ]
        ignored.extend(name for name, _ in model.named_buffers())
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            model,
            ignored,
        )
        training_model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
        )
    log_every_steps = int(training.get("log_every_steps", 10))
    checkpoint_every_steps = int(training.get("checkpoint_every_steps", 500))
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        stopped_early = False
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        training_model.train()
        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            predictions = training_model(batch)
            losses = _compute_losses(
                predictions,
                batch,
                loss_weights,
                config["data"]["target_mode"],
                single_task_weight=single_task_weight,
                merged_task_weight=merged_task_weight,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()
            global_step += 1
            if global_step == 1 or global_step % log_every_steps == 0:
                log_record = {
                    "epoch": epoch,
                    "step": global_step,
                    "world_size": context.world_size,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "effective_batch_size": (
                        batch_size_per_gpu * context.world_size
                    ),
                    "loss": float(
                        _distributed_mean(losses["loss"], context)
                    ),
                    **{
                        name: float(_distributed_mean(value, context))
                        for name, value in losses.items()
                        if name != "loss"
                    },
                    "metric_scale": float(
                        _distributed_mean(
                            predictions["depth_scale"]
                            .detach()
                            .reshape(-1)
                            .mean(),
                            context,
                        )
                    ),
                    "history_frame_count_mean": float(
                        _distributed_mean(
                            batch["frame_valid"]
                            .sum(dim=1)
                            .to(torch.float32)
                            .mean(),
                            context,
                        )
                    ),
                }
                if "ground" in predictions:
                    ground_tilt = torch.rad2deg(
                        torch.acos(
                            (
                                -predictions["ground"]["normal"][:, 1]
                            ).clamp(-1.0, 1.0)
                        )
                    ).mean()
                    log_record.update(
                        {
                            "ground_inlier_fraction": float(
                                _distributed_mean(
                                    predictions["ground"][
                                        "inlier_fraction"
                                    ].mean(),
                                    context,
                                )
                            ),
                            "ground_fallback_fraction": float(
                                _distributed_mean(
                                    predictions["ground"][
                                        "fallback_used"
                                    ]
                                    .to(torch.float32)
                                    .mean(),
                                    context,
                                )
                            ),
                            "ground_tilt_degrees": float(
                                _distributed_mean(ground_tilt, context)
                            ),
                            "predicted_camera_height": float(
                                _distributed_mean(
                                    predictions["ground"][
                                        "predicted_camera_height"
                                    ].mean(),
                                    context,
                                )
                            ),
                        }
                    )
                if "normalization" in predictions:
                    normalization = predictions["normalization"]
                    log_record.update(
                        {
                            "reference_scale_vggt": float(
                                _distributed_mean(
                                    normalization[
                                        "reference_scale_vggt"
                                    ].mean(),
                                    context,
                                )
                            ),
                            "normalized_units_per_output_pixel": float(
                                _distributed_mean(
                                    normalization[
                                        "normalized_units_per_output_pixel"
                                    ].mean(),
                                    context,
                                )
                            ),
                            "single_span_normalized": float(
                                _distributed_mean(
                                    normalization[
                                        "single_span_normalized"
                                    ].mean(),
                                    context,
                                )
                            ),
                            "merged_span_normalized": float(
                                _distributed_mean(
                                    normalization[
                                        "merged_span_normalized"
                                    ].mean(),
                                    context,
                                )
                            ),
                        }
                    )
                if context.is_primary:
                    print(
                        json.dumps(log_record),
                        flush=True,
                    )
            if (
                context.is_primary
                and checkpoint_every_steps > 0
                and global_step % checkpoint_every_steps == 0
            ):
                _save_checkpoint(
                    output_dir / "latest.pt",
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    config,
                    epoch_complete=False,
                )
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                stopped_early = True
                break

        if not args.skip_validation:
            validation = _evaluate(
                training_model,
                validation_loader,
                device,
                loss_weights,
                config["data"]["target_mode"],
                single_task_weight,
                merged_task_weight,
                context,
            )
            if context.is_primary:
                print(
                    json.dumps({"epoch": epoch, "validation": validation}),
                    flush=True,
                )
        if context.is_primary:
            _save_checkpoint(
                output_dir / "latest.pt",
                model,
                optimizer,
                epoch,
                global_step,
                config,
                epoch_complete=not stopped_early,
            )
        if context.enabled:
            dist.barrier()
        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break


def main() -> None:
    try:
        run(build_parser().parse_args())
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
