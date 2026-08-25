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
    _configure_head_compilation,
    _base_dataset,
    _build_scale_target,
    _sampler,
    learning_rate_factor,
    scale_fit_config,
)
from vggt_bev_method1.data.vggt_unit_targets import (
    regrid_merged_metric_targets_to_vggt_units,
)
from vggt_bev_method1.models import (
    LiveVGGTOmegaAdapter,
    P1DSystem,
    metric_scale_losses,
)
from vggt_bev_method1.p1d_losses import (
    P1DAdditionalLossWeights,
    p1d_bev_loss,
)
from vggt_bev_method1.p1d_metrics import p1d_validation_metrics
from vggt_bev_method1.p1b_losses import (
    P1BLossWeights,
    hidden_occupied_supervision_weight,
    wrong_evidence_kl_weight,
)
from vggt_bev_method1.train_utils import (
    distributed_runtime,
    move_batch,
    seed_everything,
)
from vggt_bev_method1.training_state import (
    EpochOffsetSampler,
    StratifiedValidationSampler,
)
from vggt_bev_method1.p1d_config import (
    CHECKPOINT_SCHEMA,
    PIPELINE_ID,
    load_p1d_config,
)
from vggt_bev_method1.p1d_train_utils import (
    build_p1d_datasets,
    p1d_collate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the direct P1D Merged + evidence + Scale pipeline"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume only a checkpoint created by this exact P1D schema.",
    )
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(config: dict, device: torch.device) -> P1DSystem:
    values = config["model"]
    layers = tuple(int(value) for value in values["cached_layers"])
    adapter = LiveVGGTOmegaAdapter(
        values["vggt_source"],
        values["checkpoint"],
        device=device,
        patch_size=int(values["patch_size"]),
        cached_layers=layers,
    )
    return P1DSystem(
        adapter,
        probability_model="evidential",
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
        merged_latent_bev_size=int(values["merged_latent_bev_size"]),
        merged_output_size=int(values["merged_bev_output_size"]),
        merged_extent_vggt=float(values["merged_bev_extent_vggt"]),
        predict_scale_uncertainty=bool(
            values.get("predict_scale_uncertainty", True)
        ),
        implicit_geometry_hidden_dim=int(
            values["implicit_geometry_hidden_dim"]
        ),
        implicit_geometry_heads=int(values["implicit_geometry_heads"]),
        implicit_geometry_layers=int(values["implicit_geometry_layers"]),
        maximum_history=int(values["maximum_history"]),
        maximum_prefix_tokens=int(values["maximum_prefix_tokens"]),
        frame_reliability_hidden_dim=int(
            values["frame_reliability_hidden_dim"]
        ),
        frame_reliability_minimum=float(
            values["frame_reliability_minimum"]
        ),
        frame_reliability_maximum=float(
            values["frame_reliability_maximum"]
        ),
        training_frame_dropout_probability=float(
            values["training_frame_dropout_probability"]
        ),
    ).to(device)


def _set_stage(model: P1DSystem, stage: str) -> None:
    if stage != "joint":
        raise ValueError("P1D trains every parallel output jointly")
    head = model.unwrapped_head()
    for parameter in head.parameters():
        parameter.requires_grad_(True)


def _bev_weights(training: dict) -> P1BLossWeights:
    return P1BLossWeights(
        observed_gate_pixel=float(training["observed_gate_pixel_weight"]),
        observed_gate_region=float(
            training.get("observed_gate_region_weight", 0.0)
        ),
        observed_gate_boundary_emphasis=float(
            training.get("observed_gate_boundary_emphasis", 0.0)
        ),
        guessed_pixel=float(training["guessed_pixel_weight"]),
        guessed_surface=float(training["guessed_surface_weight"]),
        guessed_free=float(training["guessed_free_weight"]),
        guessed_visible_surface=float(
            training["guessed_visible_surface_weight"]
        ),
        guessed_hidden_occupied=float(
            training["guessed_hidden_occupied_weight"]
        ),
        wrong_evidence_kl=float(training["wrong_evidence_kl_weight"]),
        support_bce=float(training["support_bce_weight"]),
        support_dice=float(training["support_dice_weight"]),
        support_boundary_emphasis=float(
            training.get("support_boundary_emphasis", 0.0)
        ),
        boundary_sigma=float(training.get("boundary_sigma", 3.0)),
    )


def _additional_bev_weights(training: dict) -> P1DAdditionalLossWeights:
    return P1DAdditionalLossWeights(
        history_observed_gate=float(training["history_observed_gate_weight"]),
        history_support=float(training["history_support_weight"]),
        history_guessed=float(training["history_guessed_weight"]),
        guessed_hard_pixel=float(training["guessed_hard_pixel_weight"]),
        guessed_hard_fraction=float(training["guessed_hard_fraction"]),
        guessed_hard_minimum=int(training["guessed_hard_minimum"]),
        guessed_hard_maximum_per_group=int(
            training["guessed_hard_maximum_per_group"]
        ),
    )


def _forward_losses(
    model: P1DSystem,
    batch: dict,
    config: dict,
    *,
    global_step: int,
    total_steps: int,
    include_validation_metrics: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    stage = str(config["training"]["stage"])
    training = config["training"]
    extraction = model.extract(batch["images"])
    # A separate no-grad teacher branch exists only to build lambda* labels
    # for target-coordinate conversion and Scale Token supervision. Running
    # it first lowers peak VRAM; it is not an input dependency of the head.
    scale_teacher_geometry = model.decode_scale_teacher(extraction)
    scale_target = _build_scale_target(
        batch,
        scale_teacher_geometry,
        scale_fit_config(config),
    )
    # The dense teacher has finished its only job once lambda* has been built.
    # Do not keep the frozen aggregator stack alive while the trainable native
    # 800x800 decoder runs. Public cached tokens are independent detached
    # copies, so releasing these private values changes neither head inputs nor
    # gradients.
    for private_key in ("_aggregated", "_patch_start", "_images"):
        extraction.pop(private_key, None)
    del scale_teacher_geometry

    # VGGT produced these cached values under BF16 autocast. The adapter widens
    # them to FP32 for its generic API, but this trainer immediately feeds them
    # to a BF16-autocast head. Restoring their source dtype before storage is
    # lossless and avoids retaining an unnecessarily wide frozen-token bank.
    if batch["images"].device.type == "cuda" and torch.cuda.is_bf16_supported():
        extraction["tokens"] = {
            layer: value.to(dtype=torch.bfloat16)
            for layer, value in extraction["tokens"].items()
        }
        extraction["camera_register_tokens"] = extraction[
            "camera_register_tokens"
        ].to(dtype=torch.bfloat16)
    # The deployable task path branches directly from frozen aggregator tokens.
    prediction = model.forward_head(
        extraction,
        include_merged=True,
        include_scale=True,
        assemble_runtime_outputs=False,
    )
    zero = batch["images"].new_zeros((), dtype=torch.float32)
    values: dict[str, torch.Tensor] = {}

    if stage == "joint":
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
            latest_observed_free_metric=batch[
                "latest_observed_free_target"
            ],
            latest_support_metric=batch["latest_fov_support_target"],
        )
        wrong_scale = wrong_evidence_kl_weight(
            global_step,
            total_steps,
            maximum=1.0,
            zero_fraction=float(training["wrong_evidence_zero_fraction"]),
            ramp_fraction=float(training["wrong_evidence_ramp_fraction"]),
        )
        hidden_scale = hidden_occupied_supervision_weight(
            global_step,
            total_steps,
            zero_fraction=float(training["hidden_occupied_zero_fraction"]),
            ramp_fraction=float(training["hidden_occupied_ramp_fraction"]),
        )
        bev = p1d_bev_loss(
            prediction["merged_bev"],
            targets["complete_target"],
            targets["visible_target"],
            targets["support_target"],
            latest_observed_free_target=targets[
                "latest_observed_free_target"
            ],
            latest_support_target=targets["latest_support_target"],
            gt_valid_mask=targets["gt_valid_mask"],
            probability_model="evidential",
            base_weights=_bev_weights(training),
            additional_weights=_additional_bev_weights(training),
            wrong_evidence_scale=wrong_scale,
            hidden_occupied_scale=hidden_scale,
        )
        values.update({f"merged_{key}": value for key, value in bev.items()})
        values["target_source_coverage_fraction"] = targets[
            "source_coverage_fraction"
        ].mean()
        values["target_effective_metric_extent_gt"] = targets[
            "effective_metric_extent_gt"
        ].mean()
        if include_validation_metrics:
            boundary_metrics = p1d_validation_metrics(
                prediction["merged_bev"],
                targets["complete_target"],
                targets["visible_target"],
                targets["support_target"],
                latest_observed_free_target=targets[
                    "latest_observed_free_target"
                ],
                latest_support_target=targets["latest_support_target"],
                gt_valid_mask=targets["gt_valid_mask"],
            )
            values.update(
                {f"merged_{key}": value for key, value in boundary_metrics.items()}
            )
        bev_loss = bev["loss"]
    else:
        bev_loss = zero

    if stage == "joint":
        scale = metric_scale_losses(prediction["scale"], scale_target)
        values.update({f"scale_{key}": value for key, value in scale.items()})
        scale_loss = scale["scale"]
        depth_scale_loss = scale["depth_scale"]
        uncertainty_loss = scale["uncertainty"]
    else:
        scale_loss = depth_scale_loss = uncertainty_loss = zero
        values["scale_valid_scale_fraction"] = scale_target[
            "target_valid"
        ].float().mean()

    total = (
        float(training["bev_loss_weight"]) * bev_loss
        + float(training["scale_loss_weight"]) * scale_loss
        + float(training["depth_scale_loss_weight"]) * depth_scale_loss
        + float(training.get("uncertainty_loss_weight", 0.0))
        * uncertainty_loss
    )
    values["loss"] = total
    values["frame_reliability_mean"] = prediction[
        "frame_reliability"
    ].float().mean()
    values["frame_reliability_std"] = prediction[
        "frame_reliability"
    ].float().std(unbiased=False)
    values["frame_reliability_min"] = prediction[
        "frame_reliability"
    ].float().amin()
    values["frame_reliability_max"] = prediction[
        "frame_reliability"
    ].float().amax()
    values["training_frame_keep_fraction"] = prediction[
        "frame_keep_mask"
    ].float().mean()
    return total, values


def _contract(config: dict, manifest_sha256: str, vggt_sha256: str) -> dict:
    model = config["model"]
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
        "scale_unit": "meter_per_vggt_runtime_unit",
        "scale_is_merged_input": False,
        "geometry_conditioning": (
            "implicit_temporal_cross_attention_with_learned_reliability"
        ),
        "extrinsic_input_present": False,
        "bev_waits_for_geometry_heads": False,
        "learned_frame_reliability": True,
        "runtime_postprocessing_present": False,
        "runtime_passes": 1,
        "runtime_external_inputs": ["rgb_window"],
    }


def _save_checkpoint(
    path: Path,
    *,
    model: P1DSystem,
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
            ],
            "runtime_contract": {
                "input": ["RGB frames"],
                "internal_frozen_predictions": [
                    "VGGT patch tokens",
                    "VGGT camera/register tokens",
                ],
                "training_only_teacher_predictions": [
                    "VGGT depth/confidence for lambda target construction",
                ],
                "outputs": [
                    "Merged evidential BEV in VGGT units",
                    "FOV support and observed gate",
                    "pixelwise navigation confidence",
                    "lambda_hat in meter/VGGT-unit",
                ],
                "metric_restoration": "x_m = lambda_hat * x_vggt",
                "camera_height_input": False,
                "path_head": False,
            },
        },
        temporary,
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    model: P1DSystem,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    contract: dict,
) -> tuple[int, int, int]:
    state = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    for key, expected in contract.items():
        if state.get(key) != expected:
            raise ValueError(f"resume contract mismatch for {key}")
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
    model: P1DSystem,
    loader: DataLoader,
    config: dict,
    device: torch.device,
    *,
    total_steps: int,
    global_step: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        _, values = _forward_losses(
            model,
            batch,
            config,
            global_step=global_step,
            total_steps=total_steps,
            include_validation_metrics=True,
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


def main() -> None:
    arguments = parse_args()
    config = load_p1d_config(arguments.config)
    training = config["training"]
    seed_everything(int(training["seed"]))
    train_dataset, validation_dataset = build_p1d_datasets(config)
    if arguments.data_only:
        print(
            json.dumps(
                {
                    "pipeline": PIPELINE_ID,
                    "train_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                    "single_prediction_targets_loaded": False,
                    "latest_masked_gt_loaded_for_history_loss": True,
                    "source_gt": "10m metric Merged only",
                    "runtime_external_inputs": ["rgb_window"],
                },
                indent=2,
            )
        )
        return

    distributed, rank, world_size, _, device, preflight = distributed_runtime(training)
    primary = rank == 0
    model = build_model(config, device)
    _set_stage(model, str(training["stage"]))
    compilation = _configure_head_compilation(model, training)
    if distributed:
        model.head = DistributedDataParallel(
            model.head,
            device_ids=[device.index],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=bool(training.get("ddp_static_graph", True)),
        )
    trainable = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    trainable_names = [
        name
        for name, parameter in model.unwrapped_head().named_parameters()
        if parameter.requires_grad
    ]
    if any(name.startswith("single_") for name in trainable_names):
        raise RuntimeError("P1D unexpectedly contains Single parameters")
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
        collate_fn=p1d_collate,
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
        collate_fn=p1d_collate,
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
                    "training_frame_dropout": float(
                        config["model"]["training_frame_dropout_probability"]
                    ),
                    "fresh_heads": arguments.resume is None,
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
                loss, values = _forward_losses(
                    model,
                    batch,
                    config,
                    global_step=global_step,
                    total_steps=total_steps,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable, float(training["gradient_clip_norm"]), foreach=True
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
                    output_dir / f"p1d_step_{global_step:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
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
                total_steps=total_steps,
                global_step=global_step,
            )
            if primary:
                print(json.dumps({"epoch": epoch + 1, "validation": metrics}), flush=True)
        if primary:
            _save_checkpoint(
                output_dir / "p1d_latest.pt",
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
