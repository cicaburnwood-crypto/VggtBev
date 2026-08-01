from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from vggt_bev_method1.config import load_config
from vggt_bev_method1.data import (
    RGBResizePad,
    VGGNAVMethod1Dataset,
    load_split_manifest,
    manifest_session_keys,
    method1_collate,
)
from vggt_bev_method1.losses import LossWeights, dual_method1_loss
from vggt_bev_method1.metrics import branch_metrics
from vggt_bev_method1.models import LiveVGGTOmegaAdapter, Method1System
from vggt_bev_method1.nccl import configure_and_validate_nccl
from vggt_bev_method1.training_state import (
    DistributedEpochShuffleSampler,
    EpochShuffleSampler,
    capture_rng_state,
    restore_rng_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the runtime-aligned VGGT-Ω Method I BEV pipeline"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument(
        "--smoke-first-sample",
        action="store_true",
        help="Use the deterministic one-frame first sample for integration testing",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-observed", type=Path)
    parser.add_argument("--resume-complete", type=Path)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(
    config: dict,
    *,
    verify_manifest: bool = True,
) -> tuple[VGGNAVMethod1Dataset, VGGNAVMethod1Dataset]:
    data = config["data"]
    manifest = load_split_manifest(
        data["split_manifest"],
        dataset_root=data["root"],
        validation_fraction=float(data["validation_fraction"]),
        seed=int(data["split_seed"]),
        verify_metadata=verify_manifest,
        verify_artifacts=verify_manifest,
    )
    train_keys, validation_keys = manifest_session_keys(manifest)
    preprocess = RGBResizePad(
        int(data["image_height"]),
        int(data["image_width"]),
    )
    common = {
        "root": data["root"],
        "supervision": data["supervision"],
        "preprocess": preprocess,
        "sample_stride": int(data["sample_stride"]),
        "minimum_history": int(data["minimum_history"]),
        "maximum_history": int(data["maximum_history"]),
    }
    train = VGGNAVMethod1Dataset(session_keys=train_keys, **common)
    validation = VGGNAVMethod1Dataset(session_keys=validation_keys, **common)
    train.split_manifest_sha256 = manifest["content_sha256"]
    validation.split_manifest_sha256 = manifest["content_sha256"]
    overlap = train.scene_keys.intersection(validation.scene_keys)
    if overlap:
        raise RuntimeError(f"scene leakage detected: {sorted(overlap)[:10]}")
    return train, validation


def build_model(config: dict, device: torch.device) -> Method1System:
    model_config = config["model"]
    cached_layers = tuple(int(value) for value in model_config["cached_layers"])
    spatial_scales = tuple(float(value) for value in model_config["spatial_scales"])
    adapter = LiveVGGTOmegaAdapter(
        model_config["vggt_source"],
        model_config["checkpoint"],
        device=device,
        patch_size=int(model_config["patch_size"]),
        cached_layers=cached_layers,
    )
    if int(model_config["geometry_cue_dim"]) != adapter.geometry_cue_dim:
        raise ValueError("configured geometry_cue_dim disagrees with live VGGT adapter")
    return Method1System(
        adapter,
        supervision=config["data"]["supervision"],
        cached_layers=cached_layers,
        spatial_scales=spatial_scales,
        vggt_token_dim=int(model_config["vggt_token_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        geometry_cue_dim=int(model_config["geometry_cue_dim"]),
        heads=int(model_config["attention_heads"]),
        decoder_layers=int(model_config["decoder_layers"]),
        self_attention_mode=str(model_config["self_attention_mode"]),
        cross_attention_mode=str(model_config["cross_attention_mode"]),
        deformable_samples=int(model_config["deformable_samples"]),
        cross_query_chunk_size=int(model_config["cross_query_chunk_size"]),
        query_parameter_chunk_size=int(
            model_config.get("query_parameter_chunk_size", 32768)
        ),
        single_output_size=int(model_config["single_output_size"]),
        merged_output_size=int(model_config["merged_output_size"]),
        gradient_checkpointing=bool(model_config["gradient_checkpointing"]),
    ).to(device)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def head_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    if not isinstance(model, Method1System):
        raise TypeError("expected a Method1System checkpoint source")
    return {
        key: value.detach().cpu()
        for key, value in model.unwrapped_head().state_dict().items()
    }


def load_head_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    if not isinstance(model, Method1System):
        raise TypeError("expected a Method1System checkpoint destination")
    model.unwrapped_head().load_state_dict(state, strict=True)


def resume_contract(config: dict, split_manifest_sha256: str) -> dict:
    data_keys = (
        "supervision",
        "coordinate_mode",
        "single_target_extent_m",
        "merged_target_extent_m",
        "image_height",
        "image_width",
        "sample_stride",
        "minimum_history",
        "maximum_history",
    )
    training_keys = (
        "batch_size",
        "seed",
        "geometry_warmup_epochs",
        "geometry_ramp_epochs",
    )
    training_contract = {key: config["training"][key] for key in training_keys}
    if "required_cuda_devices" in config["training"]:
        training_contract["required_cuda_devices"] = config["training"][
            "required_cuda_devices"
        ]
    training_contract["distributed_backend"] = config["training"].get(
        "distributed_backend", "nccl"
    )
    training_contract["require_same_numa"] = config["training"].get(
        "require_same_numa", True
    )
    training_contract["nccl_p2p_level"] = config["training"].get(
        "nccl_p2p_level", "AUTO"
    )
    training_contract["ddp_bucket_cap_mb"] = config["training"].get(
        "ddp_bucket_cap_mb", 25.0
    )
    return {
        "split_manifest_sha256": split_manifest_sha256,
        "data": {key: config["data"][key] for key in data_keys},
        "model": config["model"],
        "training": training_contract,
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict,
    epoch: int,
    global_step: int,
    next_epoch: int,
    next_batch_in_epoch: int,
    split_manifest_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 10,
            "method": (
                "Method I joint normalized observed+complete direct token-to-BEV"
                if config["data"]["supervision"] == "joint"
                else "Method I normalized direct token-to-BEV"
            ),
            "pdf_contract": (
                "Method I direct decoder; Equations 7, 16-20, 24-27, and "
                "32-33; metric calibration intentionally omitted"
            ),
            "epoch": epoch,
            "global_step": global_step,
            "next_epoch": next_epoch,
            "next_batch_in_epoch": next_batch_in_epoch,
            "split_manifest_sha256": split_manifest_sha256,
            "resume_contract": resume_contract(config, split_manifest_sha256),
            "config": config,
            "head_state_dict": head_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "grad_scaler_state_dict": scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "runtime_input_contract": {
                "rgb_history": True,
                "live_vggt_tokens": True,
                "live_vggt_depth": True,
                "live_vggt_intrinsics": True,
                "live_vggt_extrinsics": True,
                "camera_height": False,
                "sequence_constant_live_intrinsics": True,
                "metric_calibration": False,
                "coordinate_mode": "vggt_scene_radius_normalized",
                "internal_normalization_unit": "live_vggt_scene_radius",
                "extent_mode": "learned_per_sample_unbounded_positive",
                "single_output_size": config["model"]["single_output_size"],
                "merged_output_size": config["model"]["merged_output_size"],
                "ground_truth_depth": False,
                "ground_truth_intrinsics": False,
                "ground_truth_extrinsics": False,
                "ground_truth_trajectory": False,
            },
        },
        temporary,
    )
    temporary.replace(path)


def build_loss_weights(config: dict) -> LossWeights:
    training = config["training"]
    return LossWeights(
        occupancy=float(training["occupancy_weight"]),
        observation=float(training["observation_weight"]),
        dice=float(training["dice_weight"]),
        single=float(training["single_task_weight"]),
        merged=float(training["merged_task_weight"]),
        observed=float(training.get("observed_task_weight", 0.5)),
        complete=float(training.get("complete_task_weight", 0.5)),
    )


def geometry_gate_for_epoch(
    epoch: int,
    *,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    """Predicted-only Stage A/B transition without any GT geometry input."""

    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return 1.0
    progress = (epoch - warmup_epochs + 1) / ramp_epochs
    return max(0.0, min(1.0, progress))


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    supervision: str,
    weights: LossWeights,
    maximum_batches: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        batch = move_batch(batch, device)
        prediction = model(batch["images"], geometry_gate=1.0)
        losses = dual_method1_loss(
            prediction,
            batch,
            supervision=supervision,
            weights=weights,
        )
        values: dict[str, torch.Tensor] = dict(losses)
        tasks = ("observed", "complete") if supervision == "joint" else (supervision,)
        for task in tasks:
            single_prediction = (
                prediction["single"][task]
                if supervision == "joint"
                else prediction["single"]
            )
            merged_prediction = (
                prediction["merged"][task]
                if supervision == "joint"
                else prediction["merged"]
            )
            single_target = (
                batch[f"single_{task}_target"]
                if supervision == "joint"
                else batch["single_target"]
            )
            merged_target = (
                batch[f"merged_{task}_target"]
                if supervision == "joint"
                else batch["merged_target"]
            )
            single_metrics = branch_metrics(
                single_prediction,
                single_target,
                supervision=task,
                target_extent_m=batch["single_target_extent_m"],
            )
            merged_metrics = branch_metrics(
                merged_prediction,
                merged_target,
                supervision=task,
                target_extent_m=batch["merged_target_extent_m"],
            )
            prefix = f"{task}_" if supervision == "joint" else ""
            values.update(
                {
                    f"{prefix}single_{key}": value
                    for key, value in single_metrics.items()
                }
            )
            values.update(
                {
                    f"{prefix}merged_{key}": value
                    for key, value in merged_metrics.items()
                }
            )
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        batches += 1
    if dist.is_initialized():
        reduced = {}
        for key, total in totals.items():
            value = torch.tensor(
                [total, float(batches)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            reduced[key] = float(value[0] / value[1].clamp_min(1.0))
        result = reduced
    else:
        result = {key: value / max(batches, 1) for key, value in totals.items()}
    model.train()
    return result


def describe_data(train: VGGNAVMethod1Dataset, validation: VGGNAVMethod1Dataset) -> dict:
    return {
        "train_sessions": len(train.sessions),
        "validation_sessions": len(validation.sessions),
        "train_scenes": len(train.scene_keys),
        "validation_scenes": len(validation.scene_keys),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "scene_overlap": len(train.scene_keys.intersection(validation.scene_keys)),
        "geometry_source": (
            "live VGGT-Omega geometry plus camera-height anchor in every forward"
        ),
        "coordinate_mode": "camera_height_anchored_fixed_normalized_scale",
        "extent_mode": "fixed_6p5_single_10_merged",
        "single_output_size": 512,
        "merged_output_size": 800,
        "single_output_extent_normalized_scale": 6.5,
        "merged_output_extent_normalized_scale": 10.0,
        "camera_height_in_model_batch": True,
        "metric_target_extents_used_by_model": False,
        "gt_depth_pose_trajectory_in_model_batch": False,
        "split_manifest_sha256": train.split_manifest_sha256,
    }


def distributed_runtime(
    training: dict,
) -> tuple[bool, int, int, int, torch.device, dict]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    required_devices = int(training.get("required_cuda_devices", 1))
    if world_size > 1:
        if str(training["device"]) != "cuda":
            raise ValueError("distributed Method I training requires training.device='cuda'")
        if world_size != required_devices:
            raise ValueError(
                f"configuration requires {required_devices} CUDA processes, got {world_size}"
            )
        if torch.cuda.device_count() != world_size:
            raise RuntimeError(
                "torchrun world size must match the CUDA_VISIBLE_DEVICES count"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        nccl_preflight = configure_and_validate_nccl(
            world_size=world_size,
            require_same_numa=bool(training.get("require_same_numa", True)),
            p2p_level=str(training.get("nccl_p2p_level", "AUTO")),
        )
        dist.init_process_group(
            backend="nccl",
            device_id=device,
            timeout=timedelta(
                seconds=int(training.get("nccl_timeout_seconds", 300))
            ),
        )
        return True, rank, world_size, local_rank, device, nccl_preflight
    if required_devices > 1:
        raise RuntimeError(
            f"this configuration requires torchrun with {required_devices} processes"
        )
    device = torch.device(str(training["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    return False, rank, world_size, local_rank, device, {
        "backend": "none",
        "physical_devices": [],
        "numa_affinities": [],
    }


def smoke_subset(dataset: VGGNAVMethod1Dataset, sample_count: int) -> Subset:
    indices = [
        index
        for index, sample in enumerate(dataset.samples)
        if sample.target_frame == 0
    ][:sample_count]
    if len(indices) != sample_count:
        raise ValueError("dataset does not contain enough one-frame smoke samples")
    return Subset(dataset, indices)


def report_rank_stage(rank: int, stage: str, device: torch.device) -> None:
    print(
        json.dumps(
            {
                "rank": rank,
                "stage": stage,
                "logical_device": str(device),
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config["training"].get("pipeline") == "paired_evidential":
        if args.resume is not None:
            raise ValueError(
                "paired training uses --resume-observed and --resume-complete"
            )
        from vggt_bev_method1.cli_train_paired import run_paired_training

        run_paired_training(args, config)
        return
    training = config["training"]
    distributed, rank, world_size, local_rank, device, nccl_preflight = (
        distributed_runtime(training)
    )
    primary = rank == 0
    # Standard DDP performs the authoritative rank-0 head-state broadcast.
    seed_everything(int(training["seed"]))

    if distributed:
        if primary:
            train_dataset, validation_dataset = build_datasets(
                config,
                verify_manifest=not args.smoke_first_sample,
            )
        dist.barrier()
        if not primary:
            train_dataset, validation_dataset = build_datasets(
                config,
                verify_manifest=False,
            )
    else:
        train_dataset, validation_dataset = build_datasets(config)

    if primary:
        description = describe_data(train_dataset, validation_dataset)
        description.update(
            {
                "distributed": distributed,
                "world_size": world_size,
                "distributed_backend": nccl_preflight["backend"],
                "nccl_preflight": nccl_preflight,
                "physical_devices_from_cuda_visible_devices": True,
            }
        )
        print(json.dumps(description, indent=2), flush=True)
    if args.data_only:
        if primary:
            sample = train_dataset[0]
            target_shapes = {
                key: list(value.shape)
                for key, value in sample.items()
                if key.endswith("_target") and torch.is_tensor(value)
            }
            print(
                json.dumps(
                    {
                        "images": list(sample["images"].shape),
                        "target_shapes": target_shapes,
                        "coordinate_mode": "vggt_scene_radius_normalized",
                        "extent_mode": "learned_per_sample_unbounded_positive",
                        "single_target_extent_m": float(
                            sample["single_target_extent_m"]
                        ),
                        "merged_target_extent_m": float(
                            sample["merged_target_extent_m"]
                        ),
                        "camera_height_used": False,
                        "metric_scale_used": False,
                        "metadata": sample["metadata"],
                    },
                    indent=2,
                ),
                flush=True,
            )
        if distributed:
            dist.destroy_process_group()
        return

    if distributed:
        report_rank_stage(rank, "before_model_build", device)
    model = build_model(config, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if distributed:
        report_rank_stage(rank, "after_model_build", device)
        report_rank_stage(rank, "before_head_ddp_wrap", device)
        model.head = DistributedDataParallel(
            model.head,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            bucket_cap_mb=float(training.get("ddp_bucket_cap_mb", 25.0)),
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        torch.cuda.synchronize(device)
        report_rank_stage(rank, "after_head_ddp_wrap", device)
    parameters = [
        parameter
        for parameter in model.unwrapped_head().parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    if distributed:
        report_rank_stage(rank, "after_optimizer_build", device)
    starting_epoch = 0
    starting_batch_in_epoch = 0
    global_step = 0
    resume_rng_state = None
    checkpoint_state = None
    if args.resume is not None:
        checkpoint_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        load_head_state(model, checkpoint_state["head_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        global_step = int(checkpoint_state["global_step"])
        if int(checkpoint_state.get("format_version", 0)) >= 4:
            current_contract = resume_contract(
                config,
                train_dataset.split_manifest_sha256,
            )
            if checkpoint_state.get("resume_contract") != current_contract:
                raise ValueError(
                    "resume checkpoint data/model/order contract does not match "
                    "the current configuration"
                )
            starting_epoch = int(checkpoint_state["next_epoch"])
            starting_batch_in_epoch = int(
                checkpoint_state["next_batch_in_epoch"]
            )
            resume_rng_state = checkpoint_state["rng_state"]
        else:
            starting_epoch = int(checkpoint_state["epoch"]) + 1

    train_source = (
        smoke_subset(train_dataset, world_size)
        if args.smoke_first_sample
        else train_dataset
    )
    if distributed:
        train_sampler = DistributedEpochShuffleSampler(
            train_source,
            num_replicas=world_size,
            rank=rank,
            seed=int(training["seed"]),
            shuffle=not args.smoke_first_sample,
            drop_last=True,
        )
        validation_sampler = DistributedEpochShuffleSampler(
            validation_dataset,
            num_replicas=world_size,
            rank=rank,
            seed=int(training["seed"]),
            shuffle=False,
            drop_last=True,
        )
    else:
        train_sampler = EpochShuffleSampler(
            train_source,
            seed=int(training["seed"]),
            shuffle=not args.smoke_first_sample,
        )
        validation_sampler = None
    loader_generator = torch.Generator()
    train_loader = DataLoader(
        train_source,
        batch_size=int(training["batch_size"]),
        sampler=train_sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=method1_collate,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        sampler=validation_sampler,
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=method1_collate,
    )
    if distributed:
        report_rank_stage(rank, "after_loader_build", device)
    weights = build_loss_weights(config)
    output_dir = Path(training["output_dir"]).expanduser().resolve()
    if args.smoke_first_sample:
        output_dir = output_dir / "smoke"
    if primary:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
        report_rank_stage(rank, "after_output_barrier", device)
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16
    )
    if checkpoint_state is not None and "grad_scaler_state_dict" in checkpoint_state:
        scaler.load_state_dict(checkpoint_state["grad_scaler_state_dict"])
    first_source_reported = False
    model.train()
    stop = False
    resume_rng_pending = resume_rng_state is not None
    for epoch in range(starting_epoch, int(training["epochs"])):
        train_sampler.set_epoch(epoch)
        loader_generator.manual_seed(
            int(training["seed"]) + epoch * world_size + rank
        )
        geometry_gate = geometry_gate_for_epoch(
            epoch,
            warmup_epochs=int(training["geometry_warmup_epochs"]),
            ramp_epochs=int(training["geometry_ramp_epochs"]),
        )
        next_epoch = epoch + 1
        next_batch_in_epoch = 0
        for batch_index, batch in enumerate(train_loader):
            if epoch == starting_epoch and batch_index < starting_batch_in_epoch:
                continue
            if resume_rng_pending:
                restore_rng_state(resume_rng_state)
                resume_rng_pending = False
            step_started = time.monotonic()
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                if distributed and global_step == 0:
                    report_rank_stage(rank, "before_forward", device)
                prediction = model(
                    batch["images"],
                    geometry_gate=geometry_gate,
                )
                if distributed and global_step == 0:
                    torch.cuda.synchronize(device)
                    report_rank_stage(rank, "after_forward", device)
                losses = dual_method1_loss(
                    prediction,
                    batch,
                    supervision=config["data"]["supervision"],
                    weights=weights,
                )
            if distributed and global_step == 0:
                report_rank_stage(rank, "before_backward", device)
            scaler.scale(losses["loss"]).backward()
            if distributed and global_step == 0:
                torch.cuda.synchronize(device)
                report_rank_stage(rank, "after_backward", device)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if batch_index + 1 < len(train_loader):
                next_epoch = epoch
                next_batch_in_epoch = batch_index + 1
            else:
                next_epoch = epoch + 1
                next_batch_in_epoch = 0
            if not first_source_reported:
                if primary:
                    print(
                        json.dumps(
                            {
                                "live_vggt_verified": True,
                                "geometry_source": prediction["geometry_source"],
                                "estimated_geometry_cue_shape": list(
                                    prediction["estimated_geometry_cue"].shape
                                ),
                                "coordinate_mode": prediction[
                                    "coordinate_mode"
                                ],
                                "camera_height_used": prediction[
                                    "camera_height_used"
                                ],
                                "metric_scale_used": prediction[
                                    "metric_scale_used"
                                ],
                                "scene_radius_vggt": float(
                                    prediction["scene_radius_vggt"][0]
                                    .detach()
                                    .cpu()
                                ),
                                "single_predicted_extent_vggt_normalized": prediction[
                                    "single_predicted_extent_vggt_normalized"
                                ][0].detach().cpu().item(),
                                "merged_predicted_extent_vggt_normalized": prediction[
                                    "merged_predicted_extent_vggt_normalized"
                                ][0].detach().cpu().item(),
                                "joint_outputs": config["data"]["supervision"]
                                == "joint",
                                "world_size": world_size,
                                "distributed_backend": nccl_preflight["backend"],
                                "nccl_p2p_level": nccl_preflight.get(
                                    "nccl_p2p_level"
                                ),
                                "gt_depth_pose_trajectory_in_forward": False,
                                "geometry_gate": geometry_gate,
                            }
                        ),
                        flush=True,
                    )
                first_source_reported = True
            if primary and global_step % int(training["log_every_steps"]) == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "loss": float(losses["loss"].detach().cpu()),
                            "observed_loss": float(
                                losses.get("observed_loss", losses["loss"])
                                .detach()
                                .cpu()
                            ),
                            "complete_loss": float(
                                losses.get("complete_loss", losses["loss"])
                                .detach()
                                .cpu()
                            ),
                            "seconds": time.monotonic() - step_started,
                            "history_frames": int(batch["images"].shape[1]),
                            "effective_global_batch": (
                                int(training["batch_size"]) * world_size
                            ),
                            "geometry_gate": geometry_gate,
                        }
                    ),
                    flush=True,
                )
            if (
                primary
                and global_step % int(training["checkpoint_every_steps"]) == 0
            ):
                save_checkpoint(
                    output_dir / f"step_{global_step:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    next_epoch=next_epoch,
                    next_batch_in_epoch=next_batch_in_epoch,
                    split_manifest_sha256=train_dataset.split_manifest_sha256,
                )
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                stop = True
                break
        epoch_completed = next_epoch > epoch
        if not args.skip_validation and epoch_completed:
            result = validate(
                model,
                validation_loader,
                device=device,
                supervision=config["data"]["supervision"],
                weights=weights,
                maximum_batches=int(training["validation_batches"]),
            )
            if primary:
                print(json.dumps({"epoch": epoch, "validation": result}), flush=True)
        if primary:
            save_checkpoint(
                output_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                epoch=epoch,
                global_step=global_step,
                next_epoch=next_epoch,
                next_batch_in_epoch=next_batch_in_epoch,
                split_manifest_sha256=train_dataset.split_manifest_sha256,
            )
        if stop:
            break
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
