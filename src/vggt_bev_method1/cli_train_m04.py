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
from vggt_bev_method1.data.vggt_unit_targets import (
    regrid_merged_metric_targets_to_vggt_units,
)
from vggt_bev_method1.m04_config import (
    CHECKPOINT_SCHEMA,
    PIPELINE_ID,
    load_m04_config,
)
from vggt_bev_method1.m04_losses import m04_bev_loss, m04_scale_loss
from vggt_bev_method1.m04_train_utils import (
    build_m04_datasets,
    m04_collate,
)
from vggt_bev_method1.models import LiveVGGTOmegaAdapter, M04System
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
        description="Train the M04 parallel latest/history Merged + Scale model"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume only an exact M04 schema checkpoint.",
    )
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(config: dict, device: torch.device) -> M04System:
    values = config["model"]
    layers = tuple(int(value) for value in values["cached_layers"])
    adapter = LiveVGGTOmegaAdapter(
        values["vggt_source"],
        values["checkpoint"],
        device=device,
        patch_size=int(values["patch_size"]),
        cached_layers=layers,
    )
    return M04System(
        adapter,
        cached_layers=layers,
        spatial_scales=tuple(float(value) for value in values["spatial_scales"]),
        vggt_token_dim=int(values["vggt_token_dim"]),
        hidden_dim=int(values["hidden_dim"]),
        heads=int(values["attention_heads"]),
        latest_decoder_layers=int(values["latest_decoder_layers"]),
        history_decoder_layers=int(values["history_decoder_layers"]),
        scale_decoder_layers=int(values["scale_decoder_layers"]),
        self_attention_mode=str(values["self_attention_mode"]),
        deformable_samples=int(values["deformable_samples"]),
        cross_query_chunk_size=int(values["cross_query_chunk_size"]),
        merged_bev_size=int(values["merged_bev_output_size"]),
        merged_extent_vggt=float(values["merged_bev_extent_vggt"]),
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
    ).to(device)


def _set_trainable_head(model: M04System) -> None:
    for parameter in model.unwrapped_head().parameters():
        parameter.requires_grad_(True)


def _forward_losses(
    model: M04System,
    batch: dict,
    config: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    training = config["training"]
    extraction = model.extract(batch["images"])
    # Training-only teacher geometry constructs lambda*. It never enters the
    # M04 runtime head and is released before dense 512x512 decoding.
    scale_teacher_geometry = model.decode_scale_teacher(extraction)
    scale_target = _build_scale_target(
        batch,
        scale_teacher_geometry,
        scale_fit_config(config),
    )
    for private_key in ("_aggregated", "_patch_start", "_images"):
        extraction.pop(private_key, None)
    del scale_teacher_geometry
    if batch["images"].device.type == "cuda" and torch.cuda.is_bf16_supported():
        extraction["tokens"] = {
            layer: value.to(dtype=torch.bfloat16)
            for layer, value in extraction["tokens"].items()
        }
        extraction["camera_register_tokens"] = extraction[
            "camera_register_tokens"
        ].to(dtype=torch.bfloat16)
    prediction = model.forward_head(
        extraction,
        include_merged=True,
        include_scale=True,
        assemble_runtime_outputs=False,
    )
    targets = regrid_merged_metric_targets_to_vggt_units(
        batch["merged_fov_complete_target"],
        batch["merged_visible_target"],
        batch["merged_fov_support_target"],
        batch["merged_gt_valid_mask"],
        scale_target["lambda_gt"],
        scale_target["target_valid"],
        source_extent_m=float(config["data"]["merged_source_extent_m"]),
        target_extent_vggt=float(config["model"]["merged_bev_extent_vggt"]),
        target_size=int(config["model"]["merged_bev_output_size"]),
    )
    geometric_source_coverage = targets["source_coverage_fraction"].mean()
    scale_valid = scale_target["target_valid"].bool()
    valid_scale_source_coverage = (
        targets["source_coverage_fraction"]
        * scale_valid.to(targets["source_coverage_fraction"].dtype)
    ).sum() / scale_valid.sum().clamp_min(1).to(
        targets["source_coverage_fraction"].dtype
    )
    effective_supervision = targets["gt_valid_mask"].float().mean()
    minimum_coverage = float(training.get("minimum_mean_source_coverage", 0.0))
    if (
        minimum_coverage > 0.0
        and float(valid_scale_source_coverage.detach()) < minimum_coverage
    ):
        raise RuntimeError(
            "M04 10 m source coverage fell below the configured fail-closed "
            "threshold on valid-scale samples: "
            f"{float(valid_scale_source_coverage.detach()):.6f} "
            f"< {minimum_coverage:.6f}"
        )
    bev = m04_bev_loss(
        prediction["merged_bev"],
        targets["complete_target"],
        targets["visible_target"],
        targets["support_target"],
        gt_valid_mask=targets["gt_valid_mask"],
        evidence_kl_weight=float(training["evidence_kl_weight"]),
    )
    scale = m04_scale_loss(
        prediction["scale"],
        scale_target,
        degrees_of_freedom=float(
            training["scale_student_t_degrees_of_freedom"]
        ),
        minimum_sigma_log=float(training["scale_minimum_sigma_log"]),
        maximum_sigma_log=float(training["scale_maximum_sigma_log"]),
    )
    # Each term is one normalized conditional likelihood; no task-specific
    # hand weight or navigation signal is introduced.
    total = bev["loss"] + scale["loss"]
    values = {
        **{f"merged_{key}": value for key, value in bev.items()},
        **{f"scale_{key}": value for key, value in scale.items()},
        "target_source_coverage_fraction": geometric_source_coverage,
        "target_valid_scale_source_coverage_fraction": (
            valid_scale_source_coverage
        ),
        "target_effective_supervision_fraction": effective_supervision,
        "target_effective_metric_extent_gt": targets[
            "effective_metric_extent_gt"
        ].mean(),
        "frame_reliability_mean": prediction["frame_reliability"].float().mean(),
        "frame_reliability_std": prediction["frame_reliability"].float().std(
            unbiased=False
        ),
        "scale_frame_reliability_mean": prediction[
            "scale_frame_reliability"
        ].float().mean(),
        "scale_frame_reliability_std": prediction[
            "scale_frame_reliability"
        ].float().std(unbiased=False),
        "loss": total,
    }
    return total, values


def _contract(config: dict, manifest_sha256: str, vggt_sha256: str) -> dict:
    model = config["model"]
    data = config["data"]
    return {
        "pipeline_id": PIPELINE_ID,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "vggt_checkpoint_sha256": vggt_sha256,
        "single_bev_present": False,
        "relative_pose_head_present": False,
        "merged_coordinate_mode": "vggt_native_units",
        "merged_extent_vggt": float(model["merged_bev_extent_vggt"]),
        "merged_output_size": int(model["merged_bev_output_size"]),
        "merged_source_extent_m": float(data["merged_source_extent_m"]),
        "source_outside_loss_policy": "hard_ignore",
        "scale_unit": "meter_per_vggt_runtime_unit",
        "scale_is_merged_input": False,
        "geometry_conditioning": "parallel_latest_anchor_and_implicit_history",
        "temporal_execution": "one_window_non_recurrent",
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
    model: M04System,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
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
            "format_version": 1,
            "epoch": epoch,
            "global_step": global_step,
            "batch_in_epoch": batch_in_epoch,
            "config": config,
            "head": model.unwrapped_head().state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "trained_outputs": [
                "merged_bev_vggt_units",
                "merged_confidence",
                "fov_support",
                "observed_gate",
                "scale_m_per_vggt",
                "scale_uncertainty",
                "frame_reliability",
            ],
            "runtime_contract": {
                "input": ["ordered RGB window"],
                "outputs": [
                    "512x512 Merged evidential BEV in 6.5 VGGT units",
                    "FOV support and observed-free gate",
                    "lambda_hat in meter/VGGT-unit and uncertainty",
                    "frame reliability",
                ],
                "metric_restoration": "x_m = lambda_hat * x_vggt",
                "camera_height_input": False,
                "path_head": False,
                "training_gt_extent_m": 10.0,
                "outside_training_extent": "ignored, never relabelled unknown/free",
            },
        },
        temporary,
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    model: M04System,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    contract: dict,
) -> tuple[int, int, int]:
    state = torch.load(
        path.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    for key, expected in contract.items():
        if state.get(key) != expected:
            raise ValueError(f"M04 resume contract mismatch for {key}")
    model.unwrapped_head().load_state_dict(state["head"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return (
        int(state["epoch"]),
        int(state["global_step"]),
        int(state.get("batch_in_epoch", 0)),
    )


@torch.no_grad()
def _validate(
    model: M04System,
    loader: DataLoader,
    config: dict,
    device: torch.device,
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
            _, values = _forward_losses(model, batch, config)
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
        totals = {
            key: float(payload[index].item())
            for index, key in enumerate(keys)
        }
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    arguments = parse_args()
    config = load_m04_config(arguments.config)
    training = config["training"]
    seed_everything(int(training["seed"]))
    train_dataset, validation_dataset = build_m04_datasets(config)
    if arguments.data_only:
        print(
            json.dumps(
                {
                    "pipeline": PIPELINE_ID,
                    "train_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                    "single_prediction_targets_loaded": False,
                    "source_gt": "existing 10x10 m metric Merged PNGs",
                    "outside_source_loss_policy": "hard_ignore",
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
    model = build_model(config, device)
    _set_trainable_head(model)
    compilation = _configure_head_compilation(model, training)
    if distributed:
        model.head = DistributedDataParallel(
            model.head,
            device_ids=[device.index],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=bool(
                training["ddp_find_unused_parameters"]
            ),
            static_graph=bool(training["ddp_static_graph"]),
        )
    trainable = [
        parameter for parameter in model.head.parameters() if parameter.requires_grad
    ]
    trainable_names = [
        name
        for name, parameter in model.unwrapped_head().named_parameters()
        if parameter.requires_grad
    ]
    forbidden_prefixes = ("single_", "relative_pose", "path_", "planner_")
    if any(name.startswith(forbidden_prefixes) for name in trainable_names):
        raise RuntimeError("M04 unexpectedly contains forbidden trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.02)),
        fused=device.type == "cuda" and bool(training.get("fused_optimizer", True)),
    )
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
        collate_fn=m04_collate,
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
        collate_fn=m04_collate,
        **worker_options,
    )
    total_steps = max(1, len(loader) * int(training["epochs"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_factor(step, total_steps, training),
    )
    manifest_sha256 = _base_dataset(train_dataset).split_manifest_sha256
    vggt_sha256 = _sha256(config["model"]["checkpoint"])
    contract = _contract(config, manifest_sha256, vggt_sha256)
    start_epoch = global_step = batch_offset = 0
    if arguments.resume is not None:
        start_epoch, global_step, batch_offset = _load_checkpoint(
            arguments.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
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
                    "nccl": preflight,
                    "train_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                    "total_steps": total_steps,
                    "trainable_parameter_tensors": len(trainable_names),
                    "trainable_prefixes": sorted(
                        {name.split(".", 1)[0] for name in trainable_names}
                    ),
                    "vggt_runs_per_batch": 1,
                    "runtime_forward_passes": 1,
                    "ddp_static_graph": bool(training["ddp_static_graph"]),
                    "ddp_find_unused_parameters": bool(
                        training["ddp_find_unused_parameters"]
                    ),
                    "fresh_m04_head": arguments.resume is None,
                    "head_compilation": compilation,
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
            optimizer.zero_grad(set_to_none=True)
            use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=device.type == "cuda",
            ):
                loss, values = _forward_losses(model, batch, config)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("M04 produced a non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable,
                float(training["gradient_clip_norm"]),
                foreach=True,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            if primary and global_step % int(training["log_every_steps"]) == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "batch_seconds": time.monotonic() - batch_started,
                            "elapsed_seconds": time.monotonic() - started,
                            "learning_rate": scheduler.get_last_lr()[0],
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
                    output_dir / f"m04_step_{global_step:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    contract=contract,
                    epoch=epoch,
                    global_step=global_step,
                    batch_in_epoch=batch_index + 1,
                )
            if (
                arguments.max_train_steps is not None
                and global_step >= arguments.max_train_steps
            ):
                stop = True
                break
        batch_offset = 0
        if not arguments.skip_validation:
            metrics = _validate(model, validation_loader, config, device)
            if primary:
                print(
                    json.dumps({"epoch": epoch + 1, "validation": metrics}),
                    flush=True,
                )
        if primary:
            _save_checkpoint(
                output_dir / "m04_latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
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


if __name__ == "__main__":
    main()
