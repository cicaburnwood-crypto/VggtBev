"""Paper-level evaluation for the direct P1D Merged+Scale head.

The primary unit is one final (maximum-context) RGB prefix per validation
session.  No GT quantity is an inference input: GT depth/BEV is read only
after the frozen RGB forward pass to construct evaluation targets.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from vggt_bev_method1.cli_train_metric import (
    _base_dataset,
    _build_scale_target,
    scale_fit_config,
)
from vggt_bev_method1.cli_train_p1d import (
    _additional_bev_weights,
    _bev_weights,
    _contract,
    _sha256,
    build_model,
)
from vggt_bev_method1.data.p1b_targets import p1b_region_masks
from vggt_bev_method1.data.vggt_unit_targets import (
    regrid_merged_metric_targets_to_vggt_units,
)
from vggt_bev_method1.models import metric_scale_losses
from vggt_bev_method1.p1d_losses import p1d_bev_loss
from vggt_bev_method1.train_utils import distributed_runtime, move_batch, seed_everything
from vggt_bev_method1.p1d_config import load_p1d_config
from vggt_bev_method1.p1d_train_utils import build_p1d_datasets, p1d_collate


COUNT_PREFIXES = (
    "support",
    "gate",
    "guessed_occupied",
    "guessed_free",
    "fused_occupied",
    "visible_surface",
)
BOUNDARY_NAMES = ("support", "gate")
BOUNDARY_RADII = (1, 2, 4, 8)
CALIBRATION_BINS = 15


class FinalPrefixBatchSampler(Sampler[list[int]]):
    """Exactly one final prefix/session, sharded without rank duplication."""

    def __init__(
        self,
        dataset,
        *,
        rank: int,
        world_size: int,
        batch_size: int,
        maximum_sessions: int | None = None,
    ) -> None:
        final: dict[int, tuple[int, int]] = {}
        for index, sample in enumerate(dataset.samples):
            previous = final.get(sample.session_index)
            candidate = (int(sample.target_frame), index)
            if previous is None or candidate[0] > previous[0]:
                final[sample.session_index] = candidate
        indices = [value[1] for _, value in sorted(final.items())]
        if maximum_sessions is not None:
            indices = indices[:maximum_sessions]
        buckets: dict[int, list[int]] = defaultdict(list)
        for ordinal, index in enumerate(indices):
            if ordinal % world_size == rank:
                target_frame = int(dataset.samples[index].target_frame)
                buckets[target_frame].append(index)
        self.total_sessions = len(indices)
        self.rank_sessions = sum(map(len, buckets.values()))
        self.batches = tuple(
            tuple(values[offset : offset + batch_size])
            for target_frame in sorted(buckets)
            for values in (buckets[target_frame],)
            for offset in range(0, len(values), batch_size)
        )

    def __iter__(self) -> Iterator[list[int]]:
        return iter([list(batch) for batch in self.batches])

    def __len__(self) -> int:
        return len(self.batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate P1D on the complete unseen split")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--maximum-sessions", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius == 0:
        return mask.bool()
    return F.max_pool2d(
        mask.float().unsqueeze(1), 2 * radius + 1, stride=1, padding=radius
    ).squeeze(1) > 0.5


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return ~_dilate(~mask.bool(), radius)


def _boundary(mask: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
    # Restrict the contour to pixels whose local neighbourhood is reliable.
    stable = _erode(domain.bool(), 1)
    return (_dilate(mask, 1) ^ _erode(mask, 1)) & stable


def _binary_counts(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    domain: torch.Tensor,
) -> dict[str, torch.Tensor]:
    predicted = predicted.bool() & domain.bool()
    truth = truth.bool() & domain.bool()
    return {
        "tp": (predicted & truth).flatten(1).sum(1),
        "fp": (predicted & ~truth & domain).flatten(1).sum(1),
        "fn": (~predicted & truth & domain).flatten(1).sum(1),
        "tn": (~predicted & ~truth & domain).flatten(1).sum(1),
    }


def _boundary_counts(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    domain: torch.Tensor,
) -> dict[str, torch.Tensor]:
    predicted_boundary = _boundary(predicted, domain)
    truth_boundary = _boundary(truth, domain)
    output: dict[str, torch.Tensor] = {}
    for radius in BOUNDARY_RADII:
        output[f"r{radius}_pred"] = predicted_boundary.flatten(1).sum(1)
        output[f"r{radius}_truth"] = truth_boundary.flatten(1).sum(1)
        output[f"r{radius}_precision_hit"] = (
            predicted_boundary & _dilate(truth_boundary, radius)
        ).flatten(1).sum(1)
        output[f"r{radius}_recall_hit"] = (
            truth_boundary & _dilate(predicted_boundary, radius)
        ).flatten(1).sum(1)
    return output


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else float("nan")


def _prf(counts: dict[str, float]) -> dict[str, float]:
    precision = _ratio(counts["tp"], counts["tp"] + counts["fp"])
    recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])
    iou = _ratio(counts["tp"], counts["tp"] + counts["fp"] + counts["fn"])
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if math.isfinite(precision) and math.isfinite(recall) and precision + recall > 0
        else float("nan")
    )
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def _sample_metrics(
    prediction: dict,
    targets: dict,
    scale_prediction: dict,
    scale_target: dict,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    masks = p1b_region_masks(
        targets["complete_target"], targets["visible_target"], targets["support_target"]
    )
    gt_valid = targets["gt_valid_mask"].bool()
    support_truth = masks.valid
    support_pred = prediction["fov_support_probability"] >= 0.5
    gate_truth = masks.observed_free
    gate_pred = prediction["observed_gate_probability"] >= 0.5
    guessed_domain = masks.guessed & gt_valid
    guessed_occ_truth = masks.occupied
    guessed_occ_pred = prediction["guessed"]["occupancy_probability"] >= 0.5
    fused_domain = support_truth & gt_valid
    fused_occ_pred = prediction["occupancy_probability"] >= 0.5
    history_support = support_truth & ~targets["latest_support_target"] & gt_valid
    history_observed = gate_truth & ~targets["latest_observed_free_target"] & gt_valid

    count_sets = {
        "support": _binary_counts(support_pred, support_truth, gt_valid),
        "gate": _binary_counts(gate_pred, gate_truth, support_truth & gt_valid),
        "guessed_occupied": _binary_counts(guessed_occ_pred, guessed_occ_truth, guessed_domain),
        "guessed_free": _binary_counts(~guessed_occ_pred, ~guessed_occ_truth, guessed_domain),
        "fused_occupied": _binary_counts(fused_occ_pred, masks.occupied, fused_domain),
        "visible_surface": _binary_counts(guessed_occ_pred, masks.visible_surface, masks.visible_surface & gt_valid),
    }
    boundary_sets = {
        "support": _boundary_counts(support_pred, support_truth, gt_valid),
        "gate": _boundary_counts(gate_pred, gate_truth, support_truth & gt_valid),
    }

    guessed_probability = prediction["guessed"]["occupancy_probability"].float()
    guessed_truth_float = masks.occupied.float()
    alpha = prediction["guessed"]["alpha_occupied"].float()
    beta = prediction["guessed"]["beta_free"].float()
    strength = alpha + beta
    nll_map = torch.where(
        masks.occupied,
        torch.digamma(strength) - torch.digamma(alpha),
        torch.digamma(strength) - torch.digamma(beta),
    )
    brier_map = (guessed_probability - guessed_truth_float).square()
    evidence_confidence = (1.0 - 2.0 / strength).clamp(0.0, 1.0)
    domain_float = guessed_domain.float()
    domain_count = domain_float.flatten(1).sum(1)

    bin_index = torch.clamp(
        (guessed_probability * CALIBRATION_BINS).long(), 0, CALIBRATION_BINS - 1
    )
    records: list[dict[str, Any]] = []
    calibration = torch.zeros(
        CALIBRATION_BINS, 3, device=guessed_probability.device, dtype=torch.float64
    )
    batch = guessed_probability.shape[0]
    for sample in range(batch):
        record: dict[str, Any] = {"counts": {}, "boundary": {}}
        for name, values in count_sets.items():
            record["counts"][name] = {
                key: int(value[sample].item()) for key, value in values.items()
            }
        for name, values in boundary_sets.items():
            record["boundary"][name] = {
                key: int(value[sample].item()) for key, value in values.items()
            }
        count = float(domain_count[sample].item())
        record["guessed_evidential_nll"] = float(
            (nll_map[sample] * domain_float[sample]).sum().item() / max(count, 1.0)
        )
        record["guessed_brier"] = float(
            (brier_map[sample] * domain_float[sample]).sum().item() / max(count, 1.0)
        )
        record["guessed_evidence_confidence"] = float(
            (evidence_confidence[sample] * domain_float[sample]).sum().item()
            / max(count, 1.0)
        )
        record["guessed_pixels"] = int(count)
        record["temporal_gain"] = {
            "history_support_pixels": int(history_support[sample].sum().item()),
            "history_support_recalled": int(
                (support_pred[sample] & history_support[sample]).sum().item()
            ),
            "history_observed_pixels": int(history_observed[sample].sum().item()),
            "history_observed_recalled": int(
                (gate_pred[sample] & history_observed[sample]).sum().item()
            ),
        }
        sample_calibration = torch.zeros(
            CALIBRATION_BINS,
            3,
            device=guessed_probability.device,
            dtype=torch.float64,
        )
        if bool(scale_target["target_valid"][sample]):
            expected = float(scale_target["lambda_gt"][sample].item())
            predicted = float(scale_prediction["lambda_m_per_vggt"][sample].item())
            record["scale_valid"] = True
            record["lambda_gt"] = expected
            record["lambda_pred"] = predicted
            record["scale_relative_error"] = abs(predicted - expected) / expected
            record["scale_log_error"] = abs(math.log(predicted) - math.log(expected))
            record["cell_size_m_gt"] = float(targets["cell_size_m_gt"][sample].item())
        else:
            record["scale_valid"] = False
        records.append(record)

        active_bins = bin_index[sample][guessed_domain[sample]]
        active_probability = guessed_probability[sample][guessed_domain[sample]].double()
        active_truth = guessed_truth_float[sample][guessed_domain[sample]].double()
        if active_bins.numel():
            sample_calibration[:, 0].scatter_add_(
                0, active_bins, torch.ones_like(active_probability)
            )
            sample_calibration[:, 1].scatter_add_(0, active_bins, active_probability)
            sample_calibration[:, 2].scatter_add_(0, active_bins, active_truth)
        calibration += sample_calibration
        record["calibration_bins"] = sample_calibration.cpu().tolist()
    return records, {"calibration": calibration}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _summarize(
    records: list[dict[str, Any]],
    calibration: list[list[float]] | None = None,
    *,
    include_groups: bool = True,
) -> dict[str, Any]:
    if calibration is None:
        calibration = [
            [
                sum(record["calibration_bins"][bin_index][column] for record in records)
                for column in range(3)
            ]
            for bin_index in range(CALIBRATION_BINS)
        ]
    summary: dict[str, Any] = {"sessions": len(records), "pixel_metrics": {}, "boundary_metrics": {}}
    for name in COUNT_PREFIXES:
        counts = {
            key: float(sum(record["counts"][name][key] for record in records))
            for key in ("tp", "fp", "fn", "tn")
        }
        if name == "visible_surface":
            summary["pixel_metrics"][name] = {
                **counts,
                # Precision is intentionally undefined: valid hidden occupied
                # predictions are not false positives for visible-surface recall.
                "recall": _ratio(counts["tp"], counts["tp"] + counts["fn"]),
            }
        else:
            summary["pixel_metrics"][name] = {**counts, **_prf(counts)}
    for name in BOUNDARY_NAMES:
        summary["boundary_metrics"][name] = {}
        for radius in BOUNDARY_RADII:
            pred = sum(record["boundary"][name][f"r{radius}_pred"] for record in records)
            truth = sum(record["boundary"][name][f"r{radius}_truth"] for record in records)
            precision_hit = sum(
                record["boundary"][name][f"r{radius}_precision_hit"] for record in records
            )
            recall_hit = sum(
                record["boundary"][name][f"r{radius}_recall_hit"] for record in records
            )
            precision = _ratio(precision_hit, pred)
            recall = _ratio(recall_hit, truth)
            f1 = (
                2 * precision * recall / (precision + recall)
                if math.isfinite(precision) and math.isfinite(recall) and precision + recall > 0
                else float("nan")
            )
            summary["boundary_metrics"][name][f"radius_{radius}_px"] = {
                "precision": precision, "recall": recall, "f1": f1,
            }

    guessed_pixels = sum(record["guessed_pixels"] for record in records)
    summary["guessed_calibration"] = {
        key: _ratio(
            sum(record[key] * record["guessed_pixels"] for record in records), guessed_pixels
        )
        for key in (
            "guessed_evidential_nll", "guessed_brier", "guessed_evidence_confidence"
        )
    }
    total = sum(row[0] for row in calibration)
    ece = sum(abs(row[1] / row[0] - row[2] / row[0]) * row[0] for row in calibration if row[0] > 0)
    summary["guessed_calibration"]["ece_15_bin"] = _ratio(ece, total)
    summary["guessed_calibration"]["bins"] = calibration

    history_support_pixels = sum(
        record["temporal_gain"]["history_support_pixels"] for record in records
    )
    history_observed_pixels = sum(
        record["temporal_gain"]["history_observed_pixels"] for record in records
    )
    summary["temporal_gain"] = {
        "history_support_pixels": history_support_pixels,
        "history_support_recall": _ratio(
            sum(
                record["temporal_gain"]["history_support_recalled"]
                for record in records
            ),
            history_support_pixels,
        ),
        "history_observed_pixels": history_observed_pixels,
        "history_observed_recall": _ratio(
            sum(
                record["temporal_gain"]["history_observed_recalled"]
                for record in records
            ),
            history_observed_pixels,
        ),
    }

    valid_scale = [record for record in records if record["scale_valid"]]
    relative = [record["scale_relative_error"] for record in valid_scale]
    log_error = [record["scale_log_error"] for record in valid_scale]
    summary["scale"] = {
        "valid_sessions": len(valid_scale),
        "valid_fraction": _ratio(len(valid_scale), len(records)),
        "relative_error_mean": _ratio(sum(relative), len(relative)),
        "relative_error_p50": _percentile(relative, 0.50),
        "relative_error_p90": _percentile(relative, 0.90),
        "relative_error_p95": _percentile(relative, 0.95),
        "log_error_mean": _ratio(sum(log_error), len(log_error)),
        "log_error_p50": _percentile(log_error, 0.50),
        "log_error_p90": _percentile(log_error, 0.90),
        "log_error_p95": _percentile(log_error, 0.95),
    }
    cell_sizes = [record["cell_size_m_gt"] for record in valid_scale]
    summary["mean_gt_cell_size_m"] = _ratio(sum(cell_sizes), len(cell_sizes))
    loss_names = sorted(
        {name for record in records for name in record.get("batch_loss", {})}
    )
    summary["objective_means"] = {
        name: _ratio(
            sum(record["batch_loss"][name] for record in records if name in record["batch_loss"]),
            sum(name in record.get("batch_loss", {}) for record in records),
        )
        for name in loss_names
    }
    if include_groups:
        datasets = sorted({str(record["dataset"]) for record in records})
        histories = sorted({int(record["history_frame_count"]) for record in records})
        summary["by_dataset"] = {
            name: _summarize(
                [record for record in records if str(record["dataset"]) == name],
                include_groups=False,
            )
            for name in datasets
        }
        summary["by_history_frames"] = {
            str(history): _summarize(
                [
                    record
                    for record in records
                    if int(record["history_frame_count"]) == history
                ],
                include_groups=False,
            )
            for history in histories
        }
    return summary


def _load_checkpoint(path: Path, model, contract: dict[str, Any]) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in contract.items():
        if state.get(key) != expected:
            raise ValueError(f"checkpoint contract mismatch for {key}")
    model.unwrapped_head().load_state_dict(state["head"], strict=True)
    return state


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_p1d_config(args.config)
    training = config["training"]
    seed_everything(int(training["seed"]))
    distributed, rank, world_size, _, device, preflight = distributed_runtime(training)
    if not distributed:
        raise RuntimeError("P1D full evaluation requires torchrun")
    _, validation = build_p1d_datasets(config, verify_manifest=False)
    sampler = FinalPrefixBatchSampler(
        validation,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        maximum_sessions=args.maximum_sessions,
    )
    loader = DataLoader(
        validation,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=p1d_collate,
    )
    model = build_model(config, device)
    manifest_sha = _base_dataset(validation).split_manifest_sha256
    contract = _contract(config, manifest_sha, _sha256(config["model"]["checkpoint"]))
    state = _load_checkpoint(args.checkpoint.expanduser().resolve(), model, contract)
    model.eval()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"records.rank{rank:02d}.jsonl"
    calibration = torch.zeros(CALIBRATION_BINS, 3, dtype=torch.float64, device=device)
    processed = 0
    started = time.monotonic()
    with shard_path.open("w", encoding="utf-8") as stream:
        for batch in loader:
            metadata = batch["metadata"]
            batch = move_batch(batch, device)
            extraction = model.extract(batch["images"])
            teacher = model.decode_scale_teacher(extraction)
            scale_target = _build_scale_target(batch, teacher, scale_fit_config(config))
            prediction = model.forward_head(
                extraction, include_merged=True, include_scale=True, assemble_runtime_outputs=True
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
                latest_observed_free_metric=batch[
                    "latest_observed_free_target"
                ],
                latest_support_metric=batch["latest_fov_support_target"],
            )
            bev_loss = p1d_bev_loss(
                prediction["merged_bev"],
                targets["complete_target"], targets["visible_target"], targets["support_target"],
                latest_observed_free_target=targets[
                    "latest_observed_free_target"
                ],
                latest_support_target=targets["latest_support_target"],
                gt_valid_mask=targets["gt_valid_mask"], probability_model="evidential",
                base_weights=_bev_weights(training),
                additional_weights=_additional_bev_weights(training),
                wrong_evidence_scale=1.0,
                hidden_occupied_scale=1.0,
            )
            scale_loss = metric_scale_losses(prediction["scale"], scale_target)
            sample_values, batch_aggregate = _sample_metrics(
                prediction["merged_bev"], targets, prediction["scale"], scale_target
            )
            calibration += batch_aggregate["calibration"]
            scalar_losses = {
                f"merged_{key}": float(value.item())
                for key, value in bev_loss.items() if torch.is_tensor(value) and value.numel() == 1
            }
            scalar_losses.update(
                {f"scale_{key}": float(value.item()) for key, value in scale_loss.items()}
            )
            for item, meta in zip(sample_values, metadata, strict=True):
                item.update(
                    {
                        "sample_id": meta["sample_id"],
                        "session_key": meta["session_key"],
                        "dataset": meta["dataset"],
                        "scene_key": meta["scene_key"],
                        "history_frame_count": int(meta["history_frame_count"]),
                        "batch_loss": scalar_losses,
                    }
                )
                stream.write(json.dumps(item, sort_keys=True) + "\n")
            processed += len(metadata)
            if processed % args.log_every < len(metadata) and rank == 0:
                elapsed = time.monotonic() - started
                print(
                    json.dumps({"rank": rank, "processed": processed, "sessions_per_s": processed / elapsed}),
                    flush=True,
                )
    torch.save(calibration.cpu(), output_dir / f"calibration.rank{rank:02d}.pt")
    dist.barrier()
    if rank == 0:
        records: list[dict[str, Any]] = []
        merged_calibration = torch.zeros(CALIBRATION_BINS, 3, dtype=torch.float64)
        for shard_rank in range(world_size):
            with (output_dir / f"records.rank{shard_rank:02d}.jsonl").open(encoding="utf-8") as stream:
                records.extend(json.loads(line) for line in stream)
            merged_calibration += torch.load(
                output_dir / f"calibration.rank{shard_rank:02d}.pt", weights_only=True
            )
        records.sort(key=lambda record: record["session_key"])
        with (output_dir / "records.all.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        summary = _summarize(records, merged_calibration.tolist())
        summary.update(
            {
                "evaluation_contract": "one final/max-context RGB prefix per unseen validation session",
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "checkpoint_epoch": int(state["epoch"]),
                "checkpoint_global_step": int(state["global_step"]),
                "manifest_sha256": manifest_sha,
                "world_size": world_size,
                "rank0_preflight": preflight,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
