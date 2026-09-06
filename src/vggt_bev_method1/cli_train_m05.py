from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from vggt_bev_method1.cli_train_metric import (
    _base_dataset,
    _build_scale_target,
    _configure_head_compilation,
    _sampler,
    learning_rate_factor,
    scale_fit_config,
)
from vggt_bev_method1.m04_losses import m04_scale_loss
from vggt_bev_method1.m05_config import (
    CHECKPOINT_SCHEMA,
    PIPELINE_ID,
    load_m05_config,
)
from vggt_bev_method1.m05_losses import m05_bev_loss, m05_loss_weights
from vggt_bev_method1.m05_train_utils import (
    build_m05_datasets,
    fixed_metric_m05_target,
    m05_collate,
)
from vggt_bev_method1.models import LiveVGGTOmegaAdapter, M05System
from vggt_bev_method1.train_utils import (
    distributed_runtime,
    move_batch,
    seed_everything,
)
from vggt_bev_method1.training_state import (
    EpochOffsetSampler,
    StratifiedValidationSampler,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train M05 latest-anchored reverse-gated Merged + Scale"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(config: dict, device: torch.device) -> M05System:
    values = config["model"]
    layers = tuple(int(value) for value in values["cached_layers"])
    adapter = LiveVGGTOmegaAdapter(
        values["vggt_source"],
        values["checkpoint"],
        device=device,
        patch_size=int(values["patch_size"]),
        cached_layers=layers,
    )
    return M05System(
        adapter,
        cached_layers=layers,
        spatial_scales=tuple(float(value) for value in values["spatial_scales"]),
        vggt_token_dim=int(values["vggt_token_dim"]),
        hidden_dim=int(values["hidden_dim"]),
        heads=int(values["attention_heads"]),
        latest_decoder_layers=int(values["latest_decoder_layers"]),
        history_update_layers=int(values["history_update_layers"]),
        scale_decoder_layers=int(values["scale_decoder_layers"]),
        self_attention_mode=str(values["self_attention_mode"]),
        deformable_samples=int(values["deformable_samples"]),
        cross_query_chunk_size=int(values["cross_query_chunk_size"]),
        merged_bev_size=int(values["merged_bev_output_size"]),
        merged_extent_m=float(values["merged_bev_extent_m"]),
        query_fourier_bands=int(values["query_fourier_bands"]),
        refinement_layers=int(values["refinement_layers"]),
        predict_scale_uncertainty=bool(
            values.get("predict_scale_uncertainty", True)
        ),
        implicit_geometry_hidden_dim=int(values["implicit_geometry_hidden_dim"]),
        implicit_geometry_heads=int(values["implicit_geometry_heads"]),
        implicit_geometry_layers=int(values["implicit_geometry_layers"]),
        maximum_history=int(values["maximum_history"]),
        maximum_prefix_tokens=int(values["maximum_prefix_tokens"]),
        frame_reliability_hidden_dim=int(values["frame_reliability_hidden_dim"]),
        frame_reliability_minimum=float(values["frame_reliability_minimum"]),
        frame_reliability_maximum=float(values["frame_reliability_maximum"]),
        history_gate_initial_bias=float(values["history_gate_initial_bias"]),
        structured_prefix_readout=bool(values["structured_prefix_readout"]),
        structured_frame_reliability=bool(
            values["structured_frame_reliability"]
        ),
    ).to(device)


def _target(
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
    )


def _latest_auxiliary_execution(
    training: dict,
    *,
    global_step: int,
    total_steps: int,
    force: bool = False,
) -> tuple[bool, float]:
    """Choose the training-only latest branch without changing its expectation."""

    multiplier = float(
        training.get(
            "latest_auxiliary_multiplier",
            training.get("latest_auxiliary_loss_weight", 0.50),
        )
    )
    interval = int(training.get("latest_auxiliary_interval", 1))
    full_fraction = float(training.get("latest_auxiliary_full_fraction", 1.0))
    if interval <= 0:
        raise ValueError("latest auxiliary interval must be positive")
    if not 0.0 <= full_fraction <= 1.0:
        raise ValueError("latest auxiliary full fraction must be in [0,1]")
    full_steps = min(total_steps, int(total_steps * full_fraction + 0.999999))
    full_phase = global_step < full_steps
    periodic = (global_step - full_steps) % interval == 0
    active = force or full_phase or interval == 1 or periodic
    effective_multiplier = multiplier
    if active and not force and not full_phase and interval > 1:
        effective_multiplier *= interval
    return active, effective_multiplier if active else 0.0


def _configure_m05_execution(model: M05System, training: dict) -> dict[str, object]:
    """Apply execution controls that preserve the model and loss semantics."""

    fraction = float(training.get("attention_checkpoint_fraction", 1.0))
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("attention checkpoint fraction must be in [0,1]")
    attention_modules = 0
    for module in model.unwrapped_head().modules():
        if hasattr(module, "memory_efficient_checkpoint_fraction"):
            module.memory_efficient_checkpoint_fraction = fraction
            module.memory_efficient_training = fraction > 0.0
            attention_modules += 1
    channels_last = bool(training.get("channels_last", False))
    if channels_last:
        model.unwrapped_head().to(memory_format=torch.channels_last)
        if hasattr(model.unwrapped_head(), "channels_last_spatial"):
            model.unwrapped_head().channels_last_spatial = True
    cudnn_benchmark = bool(training.get("cudnn_benchmark", False))
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.set_float32_matmul_precision(
        str(training.get("float32_matmul_precision", "highest"))
    )
    return {
        "attention_checkpoint_fraction": fraction,
        "attention_modules": attention_modules,
        "channels_last": channels_last,
        "cudnn_benchmark": cudnn_benchmark,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _forward_losses(
    model: M05System,
    batch: dict,
    config: dict,
    *,
    global_step: int,
    total_steps: int,
    force_latest_auxiliary: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    training = config["training"]
    extraction = model.extract(batch["images"])
    teacher = model.decode_scale_teacher(extraction)
    scale_target = _build_scale_target(
        batch, teacher, scale_fit_config(config)
    )
    for private_key in ("_aggregated", "_patch_start", "_images"):
        extraction.pop(private_key, None)
    del teacher
    if batch["images"].device.type == "cuda" and torch.cuda.is_bf16_supported():
        extraction["tokens"] = {
            layer: value.to(dtype=torch.bfloat16)
            for layer, value in extraction["tokens"].items()
        }
        extraction["camera_register_tokens"] = extraction[
            "camera_register_tokens"
        ].to(dtype=torch.bfloat16)
    latest_active, latest_multiplier = _latest_auxiliary_execution(
        training,
        global_step=global_step,
        total_steps=total_steps,
        force=force_latest_auxiliary,
    )
    prediction = model.forward_head(
        extraction,
        include_merged=True,
        include_scale=True,
        include_latest_auxiliary=latest_active,
        assemble_runtime_outputs=False,
    )
    merged_target = _target(batch, config, prefix="merged")
    latest_target = (
        _target(batch, config, prefix="latest") if latest_active else None
    )
    bev = m05_bev_loss(
        prediction["merged_bev"],
        prediction.get("latest_auxiliary_bev"),
        merged_target,
        latest_target,
        weights=m05_loss_weights(training),
        global_step=global_step,
        total_steps=total_steps,
        wrong_evidence_zero_fraction=float(
            training["wrong_evidence_zero_fraction"]
        ),
        wrong_evidence_ramp_fraction=float(
            training["wrong_evidence_ramp_fraction"]
        ),
        hidden_occupied_zero_fraction=float(
            training["hidden_occupied_zero_fraction"]
        ),
        hidden_occupied_ramp_fraction=float(
            training["hidden_occupied_ramp_fraction"]
        ),
        latest_auxiliary_weight=latest_multiplier,
        loss_combination=str(
            training.get("latest_auxiliary_loss_mode", "convex")
        ),
    )
    scale = m04_scale_loss(
        prediction["scale"],
        scale_target,
        degrees_of_freedom=float(training["scale_student_t_degrees_of_freedom"]),
        minimum_sigma_log=float(training["scale_minimum_sigma_log"]),
        maximum_sigma_log=float(training["scale_maximum_sigma_log"]),
    )
    total = bev["loss"] + scale["loss"]
    gates = prediction["history_update_gate_mean"]
    gate_mean = gates.sum() / max(gates.numel(), 1)
    values = {
        **{f"bev_{key}": value for key, value in bev.items()},
        **{f"scale_{key}": value for key, value in scale.items()},
        "target_coordinate_coverage_fraction": merged_target[
            "coordinate_coverage_fraction"
        ].mean(),
        "target_effective_supervision_fraction": merged_target[
            "gt_valid_mask"
        ].float().mean(),
        "latest_effective_supervision_fraction": (
            latest_target["gt_valid_mask"].float().mean()
            if latest_target is not None
            else merged_target["gt_valid_mask"].new_zeros((), dtype=torch.float32)
        ),
        "target_metric_extent_m": merged_target["metric_extent_m"].mean(),
        "history_update_gate_mean": gate_mean,
        "frame_reliability_mean": prediction["frame_reliability"].float().mean(),
        "scale_frame_reliability_mean": prediction[
            "scale_frame_reliability"
        ].float().mean(),
        "loss": total,
    }
    return total, values


def _contract(
    config: dict,
    manifest_sha256: str,
    vggt_sha256: str,
    *,
    pipeline_id: str = PIPELINE_ID,
    checkpoint_schema: str = CHECKPOINT_SCHEMA,
) -> dict:
    model = config["model"]
    data = config["data"]
    half_extent = float(model["merged_bev_extent_m"]) / 2.0
    return {
        "pipeline_id": pipeline_id,
        "checkpoint_schema": checkpoint_schema,
        "manifest_sha256": manifest_sha256,
        "vggt_checkpoint_sha256": vggt_sha256,
        "single_bev_present": False,
        "latest_auxiliary_training_only": True,
        "merged_coordinate_mode": "fixed_metric",
        "merged_extent_m": float(model["merged_bev_extent_m"]),
        "merged_bounds_m": [
            -half_extent,
            half_extent,
            -half_extent,
            half_extent,
        ],
        "merged_cell_size_m": (
            float(model["merged_bev_extent_m"])
            / int(model["merged_bev_output_size"])
        ),
        "merged_output_size": int(model["merged_bev_output_size"]),
        "merged_source_extent_m": float(data["merged_source_extent_m"]),
        "bev_scale_dependency": "none",
        "void_invalid_loss_policy": "hard_ignore",
        "scale_unit": "meter_per_vggt_runtime_unit",
        "scale_is_merged_input": False,
        "geometry_conditioning": "latest_anchor_reverse_gated_history",
        "temporal_execution": "one_window_reverse_history_unroll",
        "history_order": "newest_to_oldest_after_latest_anchor",
        "extrinsic_input_present": False,
        "camera_height_input_present": False,
        "runtime_postprocessing_present": False,
        "runtime_passes": 1,
        "runtime_external_inputs": ["rgb_window"],
        "navigation_objective_present": False,
    }


def _save_checkpoint(
    path: Path,
    *,
    model: M05System,
    bev_optimizer: torch.optim.Optimizer,
    scale_optimizer: torch.optim.Optimizer,
    bev_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scale_scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict,
    contract: dict,
    epoch: int,
    global_step: int,
    batch_in_epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            **contract,
            "format_version": 2,
            "epoch": epoch,
            "global_step": global_step,
            "batch_in_epoch": batch_in_epoch,
            "config": config,
            "head": model.unwrapped_head().state_dict(),
            "bev_optimizer": bev_optimizer.state_dict(),
            "scale_optimizer": scale_optimizer.state_dict(),
            "bev_scheduler": bev_scheduler.state_dict(),
            "scale_scheduler": scale_scheduler.state_dict(),
            "trained_outputs": [
                "merged_bev_fixed_10m",
                "merged_confidence",
                "fov_support",
                "observed_gate",
                "scale_m_per_vggt",
                "scale_uncertainty",
                "frame_reliability",
            ],
            "training_only_outputs": ["latest_auxiliary_bev"],
        },
        temporary,
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    model: M05System,
    bev_optimizer: torch.optim.Optimizer,
    scale_optimizer: torch.optim.Optimizer,
    bev_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scale_scheduler: torch.optim.lr_scheduler.LRScheduler,
    contract: dict,
) -> tuple[int, int, int]:
    state = torch.load(
        path.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    for key, expected in contract.items():
        if state.get(key) != expected:
            raise ValueError(f"M05 resume contract mismatch for {key}")
    model.unwrapped_head().load_state_dict(state["head"], strict=True)
    bev_optimizer.load_state_dict(state["bev_optimizer"])
    scale_optimizer.load_state_dict(state["scale_optimizer"])
    bev_scheduler.load_state_dict(state["bev_scheduler"])
    scale_scheduler.load_state_dict(state["scale_scheduler"])
    return (
        int(state["epoch"]),
        int(state["global_step"]),
        int(state.get("batch_in_epoch", 0)),
    )


@torch.no_grad()
def _validate(
    model: M05System,
    loader: DataLoader,
    config: dict,
    device: torch.device,
    *,
    global_step: int,
    total_steps: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            enabled=device.type == "cuda",
        ):
            _, values = _forward_losses(
                model,
                batch,
                config,
                global_step=global_step,
                total_steps=total_steps,
                force_latest_auxiliary=True,
            )
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        count += 1
    if dist.is_initialized():
        keys = sorted(totals)
        payload = torch.tensor(
            [*(totals[key] for key in keys), float(count)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(payload)
        count = int(payload[-1].item())
        totals = {key: float(payload[index].item()) for index, key in enumerate(keys)}
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def run_training(
    *,
    config_loader=load_m05_config,
    model_builder=build_model,
    pipeline_id: str = PIPELINE_ID,
    checkpoint_schema: str = CHECKPOINT_SCHEMA,
    checkpoint_prefix: str = "m05",
    contract_builder=None,
) -> None:
    arguments = parse_args()
    config = config_loader(arguments.config)
    training = config["training"]
    seed_everything(int(training["seed"]))
    train_dataset, validation_dataset = build_m05_datasets(config)
    if arguments.data_only:
        print(
            json.dumps(
                {
                    "pipeline": pipeline_id,
                    "train_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                    "source_gt": (
                        "existing 10x10 m 512x512 categorical Merged rasters"
                    ),
                    "latest_auxiliary": "derived on the fly from existing GT",
                    "new_data_collection_required": False,
                    "runtime_external_inputs": ["rgb_window"],
                },
                indent=2,
            )
        )
        return

    distributed, rank, world_size, _, device, preflight = distributed_runtime(
        training
    )
    primary = rank == 0
    model = model_builder(config, device)
    for parameter in model.unwrapped_head().parameters():
        parameter.requires_grad_(True)
    execution = _configure_m05_execution(model, training)
    compilation = _configure_head_compilation(model, training)
    if distributed:
        model.head = DistributedDataParallel(
            model.head,
            device_ids=[device.index],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=bool(training["ddp_find_unused_parameters"]),
            static_graph=bool(training["ddp_static_graph"]),
        )
    named_trainable = [
        (name, parameter)
        for name, parameter in model.unwrapped_head().named_parameters()
        if parameter.requires_grad
    ]
    scale_prefixes = (
        "scale_token_projector.",
        "scale_frame_reliability.",
        "scale_decoder.",
    )
    scale_trainable = [
        parameter
        for name, parameter in named_trainable
        if name.startswith(scale_prefixes)
    ]
    bev_trainable = [
        parameter
        for name, parameter in named_trainable
        if not name.startswith(scale_prefixes)
    ]
    optimizer_arguments = dict(
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.02)),
        fused=device.type == "cuda" and bool(training.get("fused_optimizer", True)),
    )
    bev_optimizer = torch.optim.AdamW(bev_trainable, **optimizer_arguments)
    scale_optimizer = torch.optim.AdamW(scale_trainable, **optimizer_arguments)
    sampler = EpochOffsetSampler(
        _sampler(
            train_dataset,
            config,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
    )
    workers = int(training["num_workers"])
    worker_options = {}
    if workers > 0:
        worker_options = {
            "persistent_workers": bool(training.get("persistent_workers", True)),
            "prefetch_factor": int(training.get("prefetch_factor", 4)),
        }
    loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=m05_collate,
        **worker_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        sampler=StratifiedValidationSampler(
            validation_dataset,
            maximum_samples=int(training["validation_batches"]),
            seed=int(training["seed"]),
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
        ),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=m05_collate,
        **worker_options,
    )
    total_steps = max(1, len(loader) * int(training["epochs"]))
    bev_scheduler = torch.optim.lr_scheduler.LambdaLR(
        bev_optimizer,
        lambda step: learning_rate_factor(step, total_steps, training),
    )
    scale_scheduler = torch.optim.lr_scheduler.LambdaLR(
        scale_optimizer,
        lambda step: learning_rate_factor(step, total_steps, training),
    )
    if contract_builder is None:
        contract_builder = _contract
    contract = contract_builder(
        config,
        _base_dataset(train_dataset).split_manifest_sha256,
        _sha256(config["model"]["checkpoint"]),
        pipeline_id=pipeline_id,
        checkpoint_schema=checkpoint_schema,
    )
    start_epoch = global_step = batch_offset = 0
    if arguments.resume is not None:
        start_epoch, global_step, batch_offset = _load_checkpoint(
            arguments.resume,
            model=model,
            bev_optimizer=bev_optimizer,
            scale_optimizer=scale_optimizer,
            bev_scheduler=bev_scheduler,
            scale_scheduler=scale_scheduler,
            contract=contract,
        )
    output_dir = Path(training["output_dir"]).expanduser().resolve()
    if primary:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    **contract,
                    "world_size": world_size,
                    "local_batch_size": int(training["batch_size"]),
                    "global_batch_size": (
                        int(training["batch_size"]) * world_size
                    ),
                    "nccl": preflight,
                    "train_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                    "total_steps": total_steps,
                    "vggt_runs_per_batch": 1,
                    "runtime_forward_passes": 1,
                    "history_update_weight_sharing": True,
                    "separate_bev_scale_gradient_clipping": True,
                    "separate_bev_scale_optimizers": True,
                    "scale_default_runtime_output": False,
                    "head_compilation": compilation,
                    "execution": execution,
                    "history_cap_by_epoch": training.get(
                        "history_cap_by_epoch", []
                    ),
                    "latest_auxiliary_interval": int(
                        training.get("latest_auxiliary_interval", 1)
                    ),
                    "latest_auxiliary_full_fraction": float(
                        training.get("latest_auxiliary_full_fraction", 1.0)
                    ),
                }
            ),
            flush=True,
        )
    model.train()
    started = time.monotonic()
    stop = False
    for epoch in range(start_epoch, int(training["epochs"])):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        offset = batch_offset if epoch == start_epoch else 0
        sampler.set_start_index(offset * int(training["batch_size"]))
        for batch_index, batch in enumerate(loader, start=offset):
            batch_started = time.monotonic()
            batch = move_batch(batch, device)
            bev_optimizer.zero_grad(set_to_none=True)
            scale_optimizer.zero_grad(set_to_none=True)
            use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=device.type == "cuda",
            ):
                loss, values = _forward_losses(
                    model,
                    batch,
                    config,
                    global_step=global_step,
                    total_steps=total_steps,
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("M05 produced a non-finite training loss")
            loss.backward()
            clip = float(training["gradient_clip_norm"])
            torch.nn.utils.clip_grad_norm_(bev_trainable, clip, foreach=True)
            torch.nn.utils.clip_grad_norm_(scale_trainable, clip, foreach=True)
            bev_optimizer.step()
            scale_optimizer.step()
            bev_scheduler.step()
            scale_scheduler.step()
            global_step += 1
            if primary and global_step % int(training["log_every_steps"]) == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "batch_seconds": time.monotonic() - batch_started,
                            "elapsed_seconds": time.monotonic() - started,
                            "bev_learning_rate": bev_scheduler.get_last_lr()[0],
                            "scale_learning_rate": scale_scheduler.get_last_lr()[0],
                            **{
                                key: float(value.detach().cpu())
                                for key, value in values.items()
                            },
                        }
                    ),
                    flush=True,
                )
            if primary and global_step % int(training["checkpoint_every_steps"]) == 0:
                _save_checkpoint(
                    output_dir / f"{checkpoint_prefix}_step_{global_step:08d}.pt",
                    model=model,
                    bev_optimizer=bev_optimizer,
                    scale_optimizer=scale_optimizer,
                    bev_scheduler=bev_scheduler,
                    scale_scheduler=scale_scheduler,
                    config=config,
                    contract=contract,
                    epoch=epoch,
                    global_step=global_step,
                    batch_in_epoch=batch_index + 1,
                )
            if arguments.max_train_steps is not None and global_step >= arguments.max_train_steps:
                stop = True
                break
        batch_offset = 0
        if not arguments.skip_validation:
            metrics = _validate(
                model,
                validation_loader,
                config,
                device,
                global_step=global_step,
                total_steps=total_steps,
            )
            if primary:
                print(json.dumps({"epoch": epoch + 1, "validation": metrics}), flush=True)
        if primary:
            _save_checkpoint(
                output_dir / f"{checkpoint_prefix}_latest.pt",
                model=model,
                bev_optimizer=bev_optimizer,
                scale_optimizer=scale_optimizer,
                bev_scheduler=bev_scheduler,
                scale_scheduler=scale_scheduler,
                config=config,
                contract=contract,
                epoch=epoch + 1,
                global_step=global_step,
                batch_in_epoch=0,
            )
        if distributed:
            dist.barrier()
        if stop:
            break
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()
