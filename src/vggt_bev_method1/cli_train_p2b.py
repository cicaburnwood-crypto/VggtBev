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
    _checkpoint_sha256,
    _configure_head_compilation,
    _enabled_bev_branches,
    _sampler,
    _teacher_inputs,
    learning_rate_factor,
    scale_fit_config,
)
from vggt_bev_method1.data import method1_collate
from vggt_bev_method1.models import (
    LiveVGGTOmegaAdapter,
    P2BSystem,
    metric_scale_losses,
    metric_scale_metrics,
)
from vggt_bev_method1.p2b_config import load_p2b_config
from vggt_bev_method1.p2b_losses import (
    P2BLossWeights,
    p2b_bev_loss,
    wrong_evidence_kl_weight,
)
from vggt_bev_method1.p2b_metrics import (
    finalize_p2b_metrics,
    p2b_metric_totals,
)
from vggt_bev_method1.teacher_cache import TeacherCache
from vggt_bev_method1.train_utils import (
    build_datasets,
    distributed_runtime,
    move_batch,
    seed_everything,
    smoke_subset,
)
from vggt_bev_method1.training_state import (
    EpochOffsetSampler,
    StratifiedValidationSampler,
)

FORMAT_VERSION = 20
SCHEMAS = {
    "evidential": "p2b-three-region-evidential-v3",
    "bce": "p2b-three-region-bce-v3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P2B Two-Experts BEV head")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--smoke-first-sample", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def build_model(config: dict, device: torch.device) -> P2BSystem:
    values = config["model"]
    layers = tuple(int(value) for value in values["cached_layers"])
    adapter = LiveVGGTOmegaAdapter(
        values["vggt_source"],
        values["checkpoint"],
        device=device,
        patch_size=int(values["patch_size"]),
        cached_layers=layers,
    )
    return P2BSystem(
        adapter,
        probability_model=str(values["probability_model"]),
        cached_layers=layers,
        spatial_scales=tuple(float(value) for value in values["spatial_scales"]),
        vggt_token_dim=int(values["vggt_token_dim"]),
        hidden_dim=int(values["hidden_dim"]),
        heads=int(values["attention_heads"]),
        decoder_layers=int(values["decoder_layers"]),
        scale_decoder_layers=int(values["scale_decoder_layers"]),
        self_attention_mode=str(values["self_attention_mode"]),
        cross_attention_mode=str(values["cross_attention_mode"]),
        deformable_samples=int(values["deformable_samples"]),
        cross_query_chunk_size=int(values["cross_query_chunk_size"]),
        single_latent_bev_size=int(values["single_latent_bev_size"]),
        merged_latent_bev_size=int(values["merged_latent_bev_size"]),
        single_output_size=int(values["single_bev_output_size"]),
        merged_output_size=int(values["merged_bev_output_size"]),
        single_bev_extent_m=float(values["single_bev_extent_m"]),
        merged_bev_extent_m=float(values["merged_bev_extent_m"]),
        predict_scale_uncertainty=bool(values.get("predict_scale_uncertainty", True)),
    ).to(device)


def _set_stage(model: P2BSystem, stage: str, enabled: tuple[str, ...]) -> None:
    head = model.unwrapped_head()
    for parameter in head.parameters():
        parameter.requires_grad = False
    if stage in ("bev_only", "joint"):
        modules = [
            head.guessed_token_projector,
            head.routing_token_projector,
        ] + [getattr(head, f"{branch}_bev_decoder") for branch in enabled]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
    if stage in ("scale_only", "joint"):
        for module in (head.scale_token_projector, head.scale_decoder):
            for parameter in module.parameters():
                parameter.requires_grad = True


def _loss_weights(training: dict) -> P2BLossWeights:
    return P2BLossWeights(
        observed_gate_pixel=float(training.get("observed_gate_pixel_weight", 1.0)),
        guessed_pixel=float(training.get("guessed_pixel_weight", 1.0)),
        wrong_evidence_kl=float(training.get("wrong_evidence_kl_weight", 0.0)),
        support_bce=float(training.get("support_bce_weight", 0.5)),
        support_dice=float(training.get("support_dice_weight", 0.5)),
    )


def step_losses(
    prediction: dict,
    batch: dict,
    geometry: dict | None,
    config: dict,
    *,
    global_step: int,
    total_steps: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict | None]:
    training = config["training"]
    stage = str(training["stage"])
    branches = _enabled_bev_branches(training)
    probability_model = str(config["model"]["probability_model"])
    if branches and f"{branches[0]}_bev" in prediction:
        zero = (
            prediction[f"{branches[0]}_bev"]["observed_gate_logit"].sum()
            * 0.0
        )
    else:
        zero = prediction["scale"]["log_lambda_m_per_vggt"].sum() * 0.0
    values: dict[str, torch.Tensor] = {}
    bev_total = zero
    if stage in ("bev_only", "joint"):
        kl_scale = wrong_evidence_kl_weight(
            global_step,
            total_steps,
            maximum=1.0,
        )
        for branch in branches:
            branch_loss = p2b_bev_loss(
                prediction[f"{branch}_bev"],
                batch[f"{branch}_fov_complete_target"],
                batch[f"{branch}_visible_target"],
                batch[f"{branch}_fov_support_target"],
                probability_model=probability_model,
                weights=_loss_weights(training),
                wrong_evidence_scale=kl_scale,
            )
            task_weight = float(training[f"{branch}_task_weight"])
            bev_total = bev_total + task_weight * branch_loss["loss"]
            values.update(
                {f"{branch}_bev_{key}": value for key, value in branch_loss.items()}
            )
        values["bev_loss"] = bev_total

    scale_target = None
    if stage in ("scale_only", "joint"):
        if geometry is None:
            raise RuntimeError("scale supervision requires training-only geometry")
        scale_target = _build_scale_target(batch, geometry, scale_fit_config(config))
        scale = metric_scale_losses(prediction["scale"], scale_target)
        values.update({f"scale_{key}": value for key, value in scale.items()})
    else:
        scale = {"scale": zero, "depth_scale": zero, "uncertainty": zero}
    total = (
        float(training["bev_loss_weight"]) * bev_total
        + float(training["scale_loss_weight"]) * scale["scale"]
        + float(training["depth_scale_loss_weight"]) * scale["depth_scale"]
        + float(training.get("uncertainty_loss_weight", 0.0)) * scale["uncertainty"]
    )
    values["loss"] = total
    return total, values, scale_target


def checkpoint_contract(config: dict, manifest_sha256: str) -> dict:
    probability_model = str(config["model"]["probability_model"])
    return {
        "format_version": FORMAT_VERSION,
        "checkpoint_schema": SCHEMAS[probability_model],
        "pipeline_id": config["training"]["pipeline"],
        "probability_model": probability_model,
        "manifest_sha256": manifest_sha256,
        "runtime_inputs": ["rgb_window"],
        "bev_architecture": "observed-free-gate-plus-guessed-binary-completion",
        "routing_classes": [
            "observed_free",
            "guessed_free",
            "guessed_occupied",
        ],
        "single_output": [512, 512, 6.5],
        "merged_output": [800, 800, 10.0],
    }


def save_checkpoint(
    path: Path,
    *,
    model: P2BSystem,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    contract: dict,
    epoch: int,
    global_step: int,
    batch_in_epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **contract,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "batch_in_epoch": int(batch_in_epoch),
        "head": model.unwrapped_head().state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: P2BSystem,
    optimizer: torch.optim.Optimizer,
    scheduler,
    contract: dict,
) -> tuple[int, int, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in contract.items():
        if state.get(key) != expected:
            raise ValueError(f"checkpoint contract mismatch for {key}")
    model.unwrapped_head().load_state_dict(state["head"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return (
        int(state["epoch"]),
        int(state["global_step"]),
        int(state["batch_in_epoch"]),
    )


def _reduce_dict(values: dict[str, float], device: torch.device) -> dict[str, float]:
    if not dist.is_available() or not dist.is_initialized():
        return values
    keys = sorted(values)
    tensor = torch.tensor([values[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return dict(zip(keys, tensor.cpu().tolist(), strict=True))


@torch.no_grad()
def validate(
    model: P2BSystem,
    loader: DataLoader,
    *,
    config: dict,
    device: torch.device,
    checkpoint_hash: str,
) -> dict[str, float]:
    model.eval()
    training = config["training"]
    stage = str(training["stage"])
    branches = _enabled_bev_branches(training)
    metric_totals: dict[str, float] = {}
    scalar_totals: dict[str, float] = {}
    batches = 0
    cache_values = config.get("teacher_cache", {"mode": "live"})
    cache_mode = str(cache_values.get("mode", "live"))
    cache = (
        TeacherCache(
            cache_values["root"],
            checkpoint_sha256=checkpoint_hash,
            preprocessing_version="rgb-depth-resize-pad-v2",
        )
        if cache_mode != "live"
        else None
    )
    for batch_index, batch in enumerate(loader):
        if batch_index >= int(training["validation_batches"]):
            break
        batch = move_batch(batch, device)
        extraction, geometry = _teacher_inputs(
            model,
            batch,
            cache=cache,
            cache_mode=cache_mode,
            need_geometry=stage in ("scale_only", "joint"),
        )
        prediction = model.forward_head(
            extraction,
            enabled_bev_branches=branches if stage in ("bev_only", "joint") else (),
            include_scale=stage in ("scale_only", "joint"),
        )
        _, losses, scale_target = step_losses(
            prediction,
            batch,
            geometry,
            config,
            global_step=1,
            total_steps=1,
        )
        for key, value in losses.items():
            scalar_totals[key] = scalar_totals.get(key, 0.0) + float(value.cpu())
        if stage in ("bev_only", "joint"):
            for branch in branches:
                totals = p2b_metric_totals(
                    prediction[f"{branch}_bev"],
                    batch[f"{branch}_fov_complete_target"],
                    batch[f"{branch}_visible_target"],
                    batch[f"{branch}_fov_support_target"],
                )
                for key, value in totals.items():
                    name = f"{branch}_{key}"
                    metric_totals[name] = metric_totals.get(name, 0.0) + float(value.cpu())
        if scale_target is not None:
            for key, value in metric_scale_metrics(
                prediction["scale"], scale_target
            ).items():
                name = f"scale_{key}"
                scalar_totals[name] = scalar_totals.get(name, 0.0) + float(value.cpu())
        batches += 1
    reduced_metrics = _reduce_dict(metric_totals, device)
    reduced_scalars = _reduce_dict({**scalar_totals, "__batches": float(batches)}, device)
    count = max(reduced_scalars.pop("__batches", 0.0), 1.0)
    output = {key: value / count for key, value in reduced_scalars.items()}
    if stage in ("bev_only", "joint"):
        for branch in branches:
            prefix = f"{branch}_"
            raw = {
                key.removeprefix(prefix): value
                for key, value in reduced_metrics.items()
                if key.startswith(prefix)
            }
            output.update(
                {
                    f"{branch}_{key}": value
                    for key, value in finalize_p2b_metrics(raw).items()
                }
            )
    model.train()
    return output


def main() -> None:
    args = parse_args()
    config = load_p2b_config(args.config)
    training = config["training"]
    seed_everything(int(training["seed"]))
    train_dataset, validation_dataset = build_datasets(
        config, verify_manifest=not args.smoke_first_sample
    )
    if args.smoke_first_sample:
        train_dataset = smoke_subset(train_dataset, 1)
        validation_dataset = smoke_subset(validation_dataset, 1)
    if args.data_only:
        print(
            json.dumps(
                {
                    "pipeline": training["pipeline"],
                    "probability_model": config["model"]["probability_model"],
                    "train_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                    "runtime_inputs": ["rgb_window"],
                },
                indent=2,
            )
        )
        return

    branches = _enabled_bev_branches(training)
    distributed, rank, world_size, _, device, preflight = distributed_runtime(training)
    primary = rank == 0
    model = build_model(config, device)
    _set_stage(model, str(training["stage"]), branches)
    compile_settings = _configure_head_compilation(model, training)
    if distributed:
        model.head = DistributedDataParallel(
            model.head,
            device_ids=[device.index],
            bucket_cap_mb=float(training.get("ddp_bucket_cap_mb", 25.0)),
        )
    trainable = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.02)),
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
    loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        sampler=sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=method1_collate,
    )
    validation_sampler = StratifiedValidationSampler(
        validation_dataset,
        maximum_samples=int(training["validation_batches"]),
        seed=int(training["seed"]),
        num_replicas=world_size if distributed else 1,
        rank=rank if distributed else 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        sampler=validation_sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=method1_collate,
    )
    total_steps = max(1, len(loader) * int(training["epochs"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_factor(step, total_steps, training),
    )
    manifest_hash = _base_dataset(train_dataset).split_manifest_sha256
    contract = checkpoint_contract(config, manifest_hash)
    start_epoch = 0
    global_step = 0
    batch_offset = 0
    if args.resume is not None:
        start_epoch, global_step, batch_offset = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            contract=contract,
        )
    checkpoint_hash = _checkpoint_sha256(config["model"]["checkpoint"])
    output_dir = Path(training["output_dir"]).expanduser().resolve()
    if primary:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "pipeline": training["pipeline"],
                    "schema": contract["checkpoint_schema"],
                    "world_size": world_size,
                    "nccl": preflight,
                    "train_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                    "total_steps": total_steps,
                    "head_compile": compile_settings,
                    "vggt_runs_per_batch": 1,
                }
            ),
            flush=True,
        )
    cache_values = config.get("teacher_cache", {"mode": "live"})
    cache_mode = str(cache_values.get("mode", "live"))
    cache = (
        TeacherCache(
            cache_values["root"],
            checkpoint_sha256=checkpoint_hash,
            preprocessing_version="rgb-depth-resize-pad-v2",
        )
        if cache_mode != "live"
        else None
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
            extraction, geometry = _teacher_inputs(
                model,
                batch,
                cache=cache,
                cache_mode=cache_mode,
                need_geometry=training["stage"] in ("scale_only", "joint"),
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=device.type == "cuda",
            ):
                prediction = model.forward_head(
                    extraction,
                    enabled_bev_branches=(
                        branches if training["stage"] in ("bev_only", "joint") else ()
                    ),
                    include_scale=training["stage"] in ("scale_only", "joint"),
                )
                loss, values, _ = step_losses(
                    prediction,
                    batch,
                    geometry,
                    config,
                    global_step=global_step,
                    total_steps=total_steps,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip_norm"]))
            optimizer.step()
            scheduler.step()
            global_step += 1
            if primary and global_step % int(training["log_every_steps"]) == 0:
                memory = {}
                if device.type == "cuda":
                    gib = float(1024**3)
                    memory = {
                        "cuda_allocated_gib": torch.cuda.memory_allocated(device) / gib,
                        "cuda_reserved_gib": torch.cuda.memory_reserved(device) / gib,
                        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                        / gib,
                        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device)
                        / gib,
                    }
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "batch_seconds": time.monotonic() - batch_started,
                            "elapsed_seconds": time.monotonic() - started,
                            "learning_rate": scheduler.get_last_lr()[0],
                            **memory,
                            **{key: float(value.detach().cpu()) for key, value in values.items()},
                        }
                    ),
                    flush=True,
                )
            if primary and global_step % int(training["checkpoint_every_steps"]) == 0:
                save_checkpoint(
                    output_dir / f"p2b_step_{global_step:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    contract=contract,
                    epoch=epoch,
                    global_step=global_step,
                    batch_in_epoch=batch_index + 1,
                )
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                stop = True
                break
        batch_offset = 0
        if not args.skip_validation:
            metrics = validate(
                model,
                validation_loader,
                config=config,
                device=device,
                checkpoint_hash=checkpoint_hash,
            )
            if primary:
                print(json.dumps({"epoch": epoch, "validation": metrics}), flush=True)
        if primary:
            save_checkpoint(
                output_dir / "p2b_latest.pt",
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


if __name__ == "__main__":
    main()
