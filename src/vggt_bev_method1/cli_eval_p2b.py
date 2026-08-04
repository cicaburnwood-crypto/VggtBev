"""Distributed all-session runtime audit for a frozen P2B checkpoint.

The model path is strictly RGB-only: every record first runs the same frozen
VGGT aggregation and P2B head used at deployment.  GT BEV/depth is consumed
only afterwards to compute and record the training objective for that window.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from vggt_bev_method1.cli_train_metric import (
    _build_scale_target,
    _checkpoint_sha256,
    _enabled_bev_branches,
    scale_fit_config,
)
from vggt_bev_method1.cli_train_p2b import (
    build_model,
    checkpoint_contract,
    step_losses,
)
from vggt_bev_method1.data import (
    RGBResizePad,
    VGGNAVMethod1Dataset,
    load_split_manifest,
    manifest_session_keys,
    method1_collate,
)
from vggt_bev_method1.models import metric_scale_metrics
from vggt_bev_method1.p2b_config import load_p2b_config
from vggt_bev_method1.train_utils import distributed_runtime, move_batch, seed_everything


class RankGroupedSessionBatchSampler(Sampler[list[int]]):
    """Shard one exact final-prefix sample per session without duplication.

    Batches have one history length, which is required because VGGT has no
    temporal padding mask.  Ranks deliberately do not need equally many
    batches: this evaluator contains no per-step collectives or DDP gradient
    synchronization.  A final barrier is used only after every shard is saved.
    """

    def __init__(
        self,
        dataset: VGGNAVMethod1Dataset,
        *,
        rank: int,
        world_size: int,
        batch_size: int,
        completed_session_keys: set[str],
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError("invalid rank/world size")
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        buckets: dict[int, list[int]] = defaultdict(list)
        assigned = 0
        for index, sample in enumerate(dataset.samples):
            if index % world_size == rank:
                assigned += 1
                session_key = dataset.sessions[sample.session_index].key
                if session_key not in completed_session_keys:
                    buckets[int(sample.target_frame)].append(index)
        self.assigned_sessions = assigned
        self.remaining_sessions = sum(len(bucket) for bucket in buckets.values())
        self.batches = tuple(
            tuple(bucket[offset : offset + batch_size])
            for target in sorted(buckets)
            for bucket in (buckets[target],)
            for offset in range(0, len(bucket), batch_size)
        )

    def __iter__(self) -> Iterator[list[int]]:
        return iter([list(batch) for batch in self.batches])

    def __len__(self) -> int:
        return len(self.batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RGB-only P2B runtime and record GT-evaluated loss per session"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every-sessions", type=int, default=100)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append only missing sessions to an interrupted audit output directory",
    )
    return parser.parse_args()


def _all_session_final_prefix_dataset(config: dict) -> tuple[VGGNAVMethod1Dataset, dict]:
    data = config["data"]
    manifest = load_split_manifest(
        data["split_manifest"],
        dataset_root=data["root"],
        validation_fraction=float(data["validation_fraction"]),
        seed=int(data["split_seed"]),
        maximum_sessions=int(data["maximum_sessions"]),
        # The immutable manifest checksum is verified.  The dataset constructor
        # below still validates each loaded session's metadata and targets.
        verify_metadata=False,
        verify_artifacts=False,
    )
    train_keys, validation_keys = manifest_session_keys(manifest)
    session_keys = tuple(train_keys) + tuple(validation_keys)
    expected = int(data["maximum_sessions"])
    if len(session_keys) != expected or len(set(session_keys)) != expected:
        raise ValueError(
            f"full audit requires exactly {expected} unique manifest sessions, got "
            f"{len(session_keys)}"
        )
    dataset = VGGNAVMethod1Dataset(
        root=data["root"],
        supervision=data["supervision"],
        preprocess=RGBResizePad(
            int(data["image_height"]), int(data["image_width"])
        ),
        session_keys=session_keys,
        sample_stride=int(data["sample_stride"]),
        minimum_history=int(data["minimum_history"]),
        maximum_history=int(data["maximum_history"]),
    )
    # There are normally 1..10 prefixes per session.  Runtime audit chooses
    # the final available prefix, hence the maximum RGB context, exactly once.
    final_by_session = {
        session_index: min(
            int(session.frame_count), int(data["maximum_history"])
        )
        - 1
        for session_index, session in enumerate(dataset.sessions)
    }
    dataset.samples = [
        sample
        for sample in dataset.samples
        if sample.target_frame == final_by_session[sample.session_index]
    ]
    if len(dataset.samples) != expected:
        raise RuntimeError(
            f"expected one final prefix for each of {expected} sessions, got "
            f"{len(dataset.samples)}"
        )
    return dataset, manifest


def _slice_batch(value: Any, index: int, batch_size: int) -> Any:
    if torch.is_tensor(value):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[index : index + 1]
        return value
    if isinstance(value, dict):
        return {key: _slice_batch(item, index, batch_size) for key, item in value.items()}
    if isinstance(value, list) and len(value) == batch_size:
        return [value[index]]
    if isinstance(value, tuple) and len(value) == batch_size:
        return (value[index],)
    return value


def _float_values(values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().float().cpu())
        for key, value in values.items()
        if torch.is_tensor(value) and value.numel() == 1
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _existing_shard_totals(path: Path, *, rank: int) -> tuple[set[str], int, dict[str, float]]:
    """Validate and recover a shard after an interrupted audit."""

    session_keys: set[str] = set()
    sums: dict[str, float] = defaultdict(float)
    if not path.exists():
        return session_keys, 0, sums
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            if int(record.get("rank", -1)) != rank:
                raise ValueError(f"{path}:{line_number} belongs to another rank")
            session_key = str(record["session_key"])
            if session_key in session_keys:
                raise ValueError(f"{path}:{line_number} duplicates {session_key}")
            session_keys.add(session_key)
            for name, value in dict(record["loss"]).items():
                sums[name] += float(value)
    return session_keys, len(session_keys), sums


def _load_head_checkpoint(
    path: Path,
    *,
    model,
    contract: dict[str, Any],
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in contract.items():
        if state.get(key) != expected:
            raise ValueError(
                f"checkpoint contract mismatch for {key}: {state.get(key)!r} != {expected!r}"
            )
    model.unwrapped_head().load_state_dict(state["head"], strict=True)
    return state


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.log_every_sessions <= 0:
        raise ValueError("batch size/log interval must be positive and workers non-negative")
    config = load_p2b_config(args.config)
    training = config["training"]
    seed_everything(int(training["seed"]))
    distributed, rank, world_size, _, device, preflight = distributed_runtime(training)
    if not distributed:
        raise RuntimeError("full audit must run under torchrun on the configured GPUs")

    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        if args.resume:
            if not output_dir.is_dir():
                raise FileNotFoundError(f"resume output directory is missing: {output_dir}")
        else:
            output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    dataset, manifest = _all_session_final_prefix_dataset(config)
    contract = checkpoint_contract(config, str(manifest["content_sha256"]))
    model = build_model(config, device)
    checkpoint_state = _load_head_checkpoint(
        args.checkpoint.expanduser().resolve(), model=model, contract=contract
    )
    model.eval()
    shard_path = output_dir / f"session_losses_rank{rank:02d}.jsonl"
    completed_keys, previously_completed, existing_summaries = _existing_shard_totals(
        shard_path,
        rank=rank,
    )
    if previously_completed and not args.resume:
        raise FileExistsError(
            f"audit shard already has {previously_completed} records; use --resume"
        )
    sampler = RankGroupedSessionBatchSampler(
        dataset,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        completed_session_keys=completed_keys,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=method1_collate,
    )
    branches = _enabled_bev_branches(training)
    if branches != ("single",):
        raise ValueError(f"this audit is for the saved single-only model, got {branches}")
    checkpoint_step = int(checkpoint_state["global_step"])
    # Epoch 8 is post-warmup, therefore the trained objective's KL annealing
    # factor is 1.0.  Passing a complete-progress context reproduces it.
    total_training_steps = max(checkpoint_step, 1)
    use_bf16 = torch.cuda.is_bf16_supported()
    progress_path = output_dir / f"progress_rank{rank:02d}.json"
    summaries: dict[str, float] = defaultdict(float, existing_summaries)
    completed = previously_completed
    started = time.monotonic()

    with shard_path.open("a" if args.resume else "x", encoding="utf-8") as stream:
        for batch in loader:
            batch = move_batch(batch, device)
            batch_size = int(batch["images"].shape[0])
            batch_started = time.monotonic()
            # Deployment-equivalent RGB-only path.  Geometry is decoded only
            # after output creation because scale loss has GT supervision.
            extraction = model.extract(batch["images"])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                prediction = model.forward_head(
                    extraction,
                    enabled_bev_branches=branches,
                    include_scale=True,
                )
            geometry = model.decode_teacher_geometry(extraction)
            scale_target = _build_scale_target(batch, geometry, scale_fit_config(config))
            batch_seconds = time.monotonic() - batch_started

            for item_index, metadata in enumerate(batch["metadata"]):
                item_batch = _slice_batch(batch, item_index, batch_size)
                item_prediction = _slice_batch(prediction, item_index, batch_size)
                item_target = _slice_batch(scale_target, item_index, batch_size)
                branch_loss = step_losses(
                    item_prediction,
                    item_batch,
                    _slice_batch(geometry, item_index, batch_size),
                    config,
                    global_step=checkpoint_step,
                    total_steps=total_training_steps,
                )[1]
                scale_metrics = metric_scale_metrics(item_prediction["scale"], item_target)
                losses = _float_values(branch_loss)
                scale = _float_values(scale_metrics)
                scale.update(
                    {
                        "lambda_pred_m_per_vggt": float(
                            item_prediction["scale"]["lambda_m_per_vggt"].float().cpu()[0]
                        ),
                        "lambda_gt_m_per_vggt": float(item_target["lambda_gt"].float().cpu()[0]),
                        "quality_weight": float(item_target["quality_weight"].float().cpu()[0]),
                        "target_valid": bool(item_target["target_valid"].cpu()[0]),
                        "depth_alignment_residual": float(
                            item_target["depth_alignment_residual"].float().cpu()[0]
                        ),
                        "inlier_ratio": float(item_target["inlier_ratio"].float().cpu()[0]),
                    }
                )
                record = {
                    "schema": "p2b-runtime-session-loss-audit-v1",
                    "checkpoint": str(args.checkpoint.expanduser().resolve()),
                    "checkpoint_epoch": int(checkpoint_state["epoch"]),
                    "checkpoint_global_step": checkpoint_step,
                    "rank": rank,
                    "sample_id": metadata["sample_id"],
                    "session_key": metadata["session_key"],
                    "scene_key": metadata["scene_key"],
                    "reference_frame_id": int(metadata["reference_frame_id"]),
                    "history_frame_count": int(metadata["history_frame_count"]),
                    "runtime_inputs": ["rgb_window"],
                    "runtime_seconds_share": batch_seconds / batch_size,
                    "loss": losses,
                    "scale": scale,
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                for name, value in losses.items():
                    summaries[name] += value
                completed += 1
            if completed % args.log_every_sessions == 0:
                stream.flush()
                elapsed = time.monotonic() - started
                _atomic_json(
                    progress_path,
                    {
                        "rank": rank,
                        "completed_sessions": completed,
                        "assigned_sessions": sampler.assigned_sessions,
                        "elapsed_seconds": elapsed,
                        "sessions_per_second": completed / max(elapsed, 1e-6),
                        "mean_losses": {key: value / completed for key, value in sorted(summaries.items())},
                    },
                )
        stream.flush()

    elapsed = time.monotonic() - started
    _atomic_json(
        progress_path,
        {
            "rank": rank,
            "completed_sessions": completed,
            "assigned_sessions": sampler.assigned_sessions,
            "elapsed_seconds": elapsed,
            "sessions_per_second": completed / max(elapsed, 1e-6),
            "complete": True,
            "mean_losses": {key: value / max(completed, 1) for key, value in sorted(summaries.items())},
        },
    )

    count_tensor = torch.tensor([completed], device=device, dtype=torch.float64)
    loss_names = sorted(summaries)
    loss_tensor = torch.tensor(
        [summaries[name] for name in loss_names], device=device, dtype=torch.float64
    )
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.barrier()
    if rank == 0:
        total = int(count_tensor.item())
        expected = int(config["data"]["maximum_sessions"])
        if total != expected:
            raise RuntimeError(f"audit expected {expected} records, found {total}")
        merged_path = output_dir / "session_losses_all.jsonl"
        seen_sessions: set[str] = set()
        with merged_path.open("x", encoding="utf-8") as destination:
            for shard_rank in range(world_size):
                source = output_dir / f"session_losses_rank{shard_rank:02d}.jsonl"
                with source.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        record = json.loads(line)
                        key = str(record["session_key"])
                        if key in seen_sessions:
                            raise RuntimeError(f"duplicate audited session: {key}")
                        seen_sessions.add(key)
                        destination.write(line)
        if len(seen_sessions) != expected:
            raise RuntimeError(f"merged audit has {len(seen_sessions)} unique sessions, expected {expected}")
        _atomic_json(
            output_dir / "summary.json",
            {
                "schema": "p2b-runtime-session-loss-audit-v1",
                "complete": True,
                "session_count": total,
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
                "checkpoint_epoch": int(checkpoint_state["epoch"]),
                "checkpoint_global_step": checkpoint_step,
                "manifest_sha256": manifest["content_sha256"],
                "world_size": world_size,
                "physical_gpus": preflight["physical_devices"],
                "runtime_model_inputs": ["rgb_window"],
                "evaluation_labels_used_after_runtime": [
                    "single_fov_complete_bev",
                    "single_masked_bev",
                    "single_fov_support",
                    "gt_depth_for_scale_label",
                ],
                "window_selection": "one final prefix per manifest session; maximum usable history up to 10 RGB frames",
                "mean_losses": {
                    name: float(loss_tensor[index].item()) / total
                    for index, name in enumerate(loss_names)
                },
                "records": "session_losses_all.jsonl",
                "shards": [f"session_losses_rank{index:02d}.jsonl" for index in range(world_size)],
            },
        )
        print(json.dumps({"complete": True, "sessions": total, "output_dir": str(output_dir)}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
