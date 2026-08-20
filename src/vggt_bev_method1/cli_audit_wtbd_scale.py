from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vggt_bev_method1.cli_train_metric import _build_scale_target, scale_fit_config
from vggt_bev_method1.cli_train_wtbd import build_model
from vggt_bev_method1.train_utils import move_batch, seed_everything
from vggt_bev_method1.wtbd_config import load_wtbd_config
from vggt_bev_method1.wtbd_train_utils import (
    build_wtbd_datasets,
    wtbd_collate,
)


def summarize_scale_audit(
    lambda_values: torch.Tensor,
    quality_values: torch.Tensor,
    residual_values: torch.Tensor,
    *,
    source_extent_m: float,
    configured_extent_vggt: float,
) -> dict:
    if lambda_values.ndim != 1 or lambda_values.numel() == 0:
        raise ValueError("scale audit requires valid one-dimensional labels")
    available_extent = float(source_extent_m) / lambda_values

    def quantiles(values: torch.Tensor) -> dict[str, float]:
        levels = torch.tensor(
            [0.01, 0.05, 0.50, 0.95, 0.99],
            device=values.device,
            dtype=values.dtype,
        )
        result = torch.quantile(values, levels)
        return {
            name: float(value)
            for name, value in zip(
                ("p01", "p05", "p50", "p95", "p99"), result, strict=True
            )
        }

    source_coverage_fraction = torch.minimum(
        torch.ones_like(available_extent),
        (available_extent / float(configured_extent_vggt)).square(),
    )
    return {
        "valid_sample_count": int(lambda_values.numel()),
        "lambda_m_per_vggt": quantiles(lambda_values),
        "quality_weight": quantiles(quality_values),
        "median_log_depth_residual": quantiles(residual_values),
        "available_extent_vggt_from_10m_source": quantiles(available_extent),
        "recommended_max_extent_vggt": {
            "for_90_percent_full_source_coverage": float(
                torch.quantile(available_extent, 0.10)
            ),
            "for_95_percent_full_source_coverage": float(
                torch.quantile(available_extent, 0.05)
            ),
            "for_99_percent_full_source_coverage": float(
                torch.quantile(available_extent, 0.01)
            ),
        },
        "configured_extent_vggt": float(configured_extent_vggt),
        "mean_source_area_coverage_at_configured_extent": float(
            source_coverage_fraction.mean()
        ),
        "contract": "extent recommendations are source-GT coverage limits, not metres",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit WTBD metric/VGGT scale labels")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/wtbd_scale_audit.json")
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    arguments = parse_args()
    if arguments.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    config = load_wtbd_config(arguments.config)
    seed_everything(int(config["training"]["seed"]))
    train_dataset, _ = build_wtbd_datasets(config)
    loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=True,
        collate_fn=wtbd_collate,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("WTBD scale audit requires CUDA for live frozen VGGT")
    device = torch.device("cuda", 0)
    model = build_model(config, device).eval()
    lambdas = []
    qualities = []
    residuals = []
    attempted = 0
    for batch in loader:
        attempted += 1
        batch = move_batch(batch, device)
        extraction = model.extract(batch["images"])
        geometry = model.decode_teacher_geometry(extraction)
        target = _build_scale_target(batch, geometry, scale_fit_config(config))
        valid = target["target_valid"]
        if bool(valid.any()):
            lambdas.append(target["lambda_gt"][valid].cpu())
            qualities.append(target["quality_weight"][valid].cpu())
            residuals.append(target["depth_alignment_residual"][valid].cpu())
        if attempted >= arguments.max_samples:
            break
    if not lambdas:
        raise RuntimeError("scale audit found no valid VGGT/GT alignment labels")
    report = summarize_scale_audit(
        torch.cat(lambdas),
        torch.cat(qualities),
        torch.cat(residuals),
        source_extent_m=float(config["data"]["merged_source_extent_m"]),
        configured_extent_vggt=float(config["model"]["merged_bev_extent_vggt"]),
    )
    report.update(
        {
            "attempted_sample_count": attempted,
            "valid_fraction": report["valid_sample_count"] / attempted,
            "vggt_checkpoint": str(config["model"]["checkpoint"]),
            "manifest": str(config["data"]["split_manifest"]),
        }
    )
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
