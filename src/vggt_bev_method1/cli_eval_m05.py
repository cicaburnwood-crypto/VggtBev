"""Unseen-scene evaluation for M05 Merged BEV + Scale."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from vggt_bev_method1.cli_eval_p1d import (
    CALIBRATION_BINS,
    FinalPrefixBatchSampler,
    _sample_metrics,
    _summarize,
)
from vggt_bev_method1.cli_train_m05 import _contract, _sha256, build_model
from vggt_bev_method1.cli_train_metric import (
    _base_dataset,
    _build_scale_target,
    scale_fit_config,
)
from vggt_bev_method1.m04_losses import m04_scale_loss
from vggt_bev_method1.m05_config import load_m05_config
from vggt_bev_method1.m05_losses import m05_loss_weights
from vggt_bev_method1.m05_train_utils import (
    build_m05_datasets,
    fixed_metric_m05_target,
    m05_collate,
)
from vggt_bev_method1.p1b_losses import p1b_bev_loss
from vggt_bev_method1.train_utils import (
    distributed_runtime,
    move_batch,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate M05 on final RGB prefixes from unseen scenes"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--maximum-sessions", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _load_head(path: Path, model, contract: dict[str, Any]) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in contract.items():
        if state.get(key) != expected:
            raise ValueError(f"M05 evaluation contract mismatch for {key}")
    model.unwrapped_head().load_state_dict(state["head"], strict=True)
    return state


@torch.no_grad()
def main() -> None:
    arguments = parse_args()
    config = load_m05_config(arguments.config)
    training = config["training"]
    seed_everything(int(training["seed"]))
    distributed, rank, world_size, _, device, preflight = distributed_runtime(
        training
    )
    _, validation = build_m05_datasets(config, verify_manifest=False)
    sampler = FinalPrefixBatchSampler(
        validation,
        rank=rank,
        world_size=world_size,
        batch_size=arguments.batch_size,
        maximum_sessions=arguments.maximum_sessions,
    )
    worker_options = {}
    if arguments.num_workers > 0:
        worker_options = {"persistent_workers": True, "prefetch_factor": 2}
    loader = DataLoader(
        validation,
        batch_sampler=sampler,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=m05_collate,
        **worker_options,
    )
    model = build_model(config, device)
    manifest_sha = _base_dataset(validation).split_manifest_sha256
    contract = _contract(
        config,
        manifest_sha,
        _sha256(config["model"]["checkpoint"]),
    )
    state = _load_head(
        arguments.checkpoint.expanduser().resolve(), model, contract
    )
    model.eval()
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"records.rank{rank:02d}.jsonl"
    calibration = torch.zeros(
        CALIBRATION_BINS, 3, dtype=torch.float64, device=device
    )
    processed = 0
    started = time.monotonic()
    with shard_path.open("w", encoding="utf-8") as stream:
        for batch in loader:
            metadata = batch["metadata"]
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
                for private_key in ("_aggregated", "_patch_start", "_images"):
                    extraction.pop(private_key, None)
                del teacher
                if device.type == "cuda" and torch.cuda.is_bf16_supported():
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
                    include_latest_auxiliary=False,
                    assemble_runtime_outputs=True,
                )
                targets = fixed_metric_m05_target(
                    batch["merged_fov_complete_target"],
                    batch["merged_visible_target"],
                    batch["merged_fov_support_target"],
                    batch["merged_gt_valid_mask"],
                    extent_m=float(config["model"]["merged_bev_extent_m"]),
                    latest_observed_free=batch[
                        "latest_observed_free_target"
                    ],
                    latest_support=batch["latest_fov_support_target"],
                )
                bev_loss = p1b_bev_loss(
                    prediction["merged_bev"],
                    targets["complete_target"],
                    targets["visible_target"],
                    targets["support_target"],
                    gt_valid_mask=targets["gt_valid_mask"],
                    probability_model="evidential",
                    weights=m05_loss_weights(training),
                    wrong_evidence_scale=1.0,
                    hidden_occupied_scale=1.0,
                )
                scale_loss = m04_scale_loss(
                    prediction["scale"],
                    scale_target,
                    degrees_of_freedom=float(
                        training["scale_student_t_degrees_of_freedom"]
                    ),
                    minimum_sigma_log=float(training["scale_minimum_sigma_log"]),
                    maximum_sigma_log=float(training["scale_maximum_sigma_log"]),
                )
            sample_values, aggregate = _sample_metrics(
                prediction["merged_bev"],
                targets,
                prediction["scale"],
                scale_target,
            )
            calibration += aggregate["calibration"]
            losses = {
                f"merged_{key}": float(value.item())
                for key, value in bev_loss.items()
                if torch.is_tensor(value) and value.numel() == 1
            }
            losses.update(
                {
                    f"scale_{key}": float(value.item())
                    for key, value in scale_loss.items()
                }
            )
            for item, meta, valid_mask in zip(
                sample_values,
                metadata,
                targets["gt_valid_mask"],
                strict=True,
            ):
                item.update(
                    {
                        "sample_id": meta["sample_id"],
                        "session_key": meta["session_key"],
                        "dataset": meta["dataset"],
                        "scene_key": meta["scene_key"],
                        "history_frame_count": int(meta["history_frame_count"]),
                        "coordinate_coverage_fraction": 1.0,
                        "effective_supervision_fraction": float(
                            valid_mask.float().mean().item()
                        ),
                        "batch_loss": losses,
                    }
                )
                stream.write(json.dumps(item, sort_keys=True) + "\n")
            processed += len(metadata)
            if processed % arguments.log_every < len(metadata) and rank == 0:
                elapsed = time.monotonic() - started
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "processed": processed,
                            "sessions_per_s": processed / max(elapsed, 1e-6),
                        }
                    ),
                    flush=True,
                )
    torch.save(calibration.cpu(), output_dir / f"calibration.rank{rank:02d}.pt")
    if distributed:
        dist.barrier()
    if rank == 0:
        records: list[dict[str, Any]] = []
        merged_calibration = torch.zeros(CALIBRATION_BINS, 3, dtype=torch.float64)
        for shard_rank in range(world_size):
            with (output_dir / f"records.rank{shard_rank:02d}.jsonl").open(
                encoding="utf-8"
            ) as stream:
                records.extend(json.loads(line) for line in stream)
            merged_calibration += torch.load(
                output_dir / f"calibration.rank{shard_rank:02d}.pt",
                weights_only=True,
            )
        records.sort(key=lambda record: record["session_key"])
        with (output_dir / "records.all.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        summary = _summarize(records, merged_calibration.tolist())
        summary.update(
            {
                "evaluation_contract": (
                    "one final/max-context RGB prefix per unseen validation session"
                ),
                "checkpoint": str(arguments.checkpoint.expanduser().resolve()),
                "checkpoint_epoch": int(state["epoch"]),
                "checkpoint_global_step": int(state["global_step"]),
                "manifest_sha256": manifest_sha,
                "world_size": world_size,
                "rank0_preflight": preflight,
                "elapsed_seconds": time.monotonic() - started,
                "source_gt_extent_m": 10.0,
                "source_outside_loss_policy": "hard_ignore",
            }
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
