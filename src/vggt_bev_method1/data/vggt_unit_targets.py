from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from vggt_bev_method1.config import LabelValues


@dataclass(frozen=True)
class VGGTUnitTargetContract:
    source_extent_m: float
    target_extent_vggt: float
    target_size: int


def _query_grid(
    lambda_m_per_vggt: torch.Tensor,
    contract: VGGTUnitTargetContract,
) -> tuple[torch.Tensor, torch.Tensor]:
    if lambda_m_per_vggt.ndim != 1:
        raise ValueError("lambda_m_per_vggt must have shape [B]")
    if (
        contract.source_extent_m <= 0.0
        or contract.target_extent_vggt <= 0.0
        or contract.target_size <= 0
    ):
        raise ValueError("target-grid contract must be positive")
    size = contract.target_size
    pixel = torch.arange(
        size,
        device=lambda_m_per_vggt.device,
        dtype=lambda_m_per_vggt.dtype,
    )
    cell_vggt = contract.target_extent_vggt / size
    x_vggt = -contract.target_extent_vggt / 2.0 + (pixel + 0.5) * cell_vggt
    z_vggt = contract.target_extent_vggt / 2.0 - (pixel + 0.5) * cell_vggt
    z_grid, x_grid = torch.meshgrid(z_vggt, x_vggt, indexing="ij")
    scale = lambda_m_per_vggt[:, None, None]
    grid_x = scale * x_grid[None] / (contract.source_extent_m / 2.0)
    grid_y = -scale * z_grid[None] / (contract.source_extent_m / 2.0)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    coverage = (grid.abs() <= 1.0).all(dim=-1)
    return grid, coverage


def _sample_nearest(value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(
        value[:, None].float(),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0]


@torch.no_grad()
def regrid_merged_metric_targets_to_vggt_units(
    complete_metric: torch.Tensor,
    visible_metric: torch.Tensor,
    support_metric: torch.Tensor,
    gt_valid_metric: torch.Tensor,
    lambda_m_per_vggt: torch.Tensor,
    scale_target_valid: torch.Tensor,
    *,
    source_extent_m: float,
    target_extent_vggt: float,
    target_size: int,
    latest_observed_free_metric: torch.Tensor | None = None,
    latest_support_metric: torch.Tensor | None = None,
    labels: LabelValues = LabelValues(),
) -> dict[str, torch.Tensor]:
    """Re-express existing metric Merged labels in current VGGT units.

    This is an inverse lookup, not an image resize.  Every target cell at
    ``(x_V,z_V)`` samples the metric source at
    ``(lambda*x_V, lambda*z_V)``.  Source-exterior and unreliable-scale cells
    are hard ignored by all BEV losses.
    """

    tensors = (complete_metric, visible_metric, support_metric, gt_valid_metric)
    if any(value.ndim != 3 for value in tensors):
        raise ValueError("merged metric targets must have shape [B,H,W]")
    if any(value.shape != complete_metric.shape for value in tensors[1:]):
        raise ValueError("merged metric targets must align")
    if scale_target_valid.shape != lambda_m_per_vggt.shape:
        raise ValueError("scale target validity must have shape [B]")
    if (latest_observed_free_metric is None) != (latest_support_metric is None):
        raise ValueError(
            "latest observed-free and support targets must be supplied together"
        )
    if latest_observed_free_metric is not None and (
        latest_observed_free_metric.shape != complete_metric.shape
        or latest_support_metric.shape != complete_metric.shape
    ):
        raise ValueError("latest temporal targets must align with Merged GT")
    contract = VGGTUnitTargetContract(
        source_extent_m=float(source_extent_m),
        target_extent_vggt=float(target_extent_vggt),
        target_size=int(target_size),
    )
    grid, source_coverage = _query_grid(lambda_m_per_vggt.detach(), contract)
    complete = _sample_nearest(complete_metric, grid).round().to(torch.uint8)
    visible = _sample_nearest(visible_metric, grid).round().to(torch.uint8)
    support = _sample_nearest(support_metric, grid) > 0.5
    sampled_valid = _sample_nearest(gt_valid_metric, grid) > 0.5
    target_valid = (
        sampled_valid
        & source_coverage
        & scale_target_valid[:, None, None].bool()
    )

    unknown = int(labels.unknown)
    complete = torch.where(
        support,
        complete,
        torch.full_like(complete, unknown),
    )
    visible_known = support & (visible != unknown)
    visible = torch.where(
        visible_known,
        visible,
        torch.full_like(visible, unknown),
    )
    if not torch.equal(complete != unknown, support):
        raise ValueError("regridded support and complete labels disagree")
    disagreement = visible_known & (visible != complete)
    if bool(disagreement.any()):
        raise ValueError("regridded visible labels disagree with complete GT")
    metric_extent = lambda_m_per_vggt * float(target_extent_vggt)
    source_coverage_fraction = source_coverage.float().mean(dim=(1, 2))
    output = {
        "complete_target": complete,
        "visible_target": visible,
        "support_target": support,
        "gt_valid_mask": target_valid,
        "source_coverage_mask": source_coverage,
        "source_coverage_fraction": source_coverage_fraction,
        "effective_metric_extent_gt": metric_extent,
        "cell_size_vggt": complete.new_full(
            (complete.shape[0],),
            float(target_extent_vggt) / int(target_size),
            dtype=torch.float32,
        ),
        "cell_size_m_gt": metric_extent / int(target_size),
    }
    if latest_observed_free_metric is not None:
        latest_observed = _sample_nearest(
            latest_observed_free_metric,
            grid,
        ) > 0.5
        latest_support = _sample_nearest(latest_support_metric, grid) > 0.5
        latest_support = latest_support & target_valid
        latest_observed = latest_observed & latest_support
        output.update(
            {
                "latest_observed_free_target": latest_observed,
                "latest_support_target": latest_support,
                "history_observed_free_region": (
                    (visible != unknown)
                    & (visible == int(labels.free))
                    & ~latest_observed
                    & target_valid
                ),
                "history_support_region": (
                    support & ~latest_support & target_valid
                ),
            }
        )
    return output


def restore_metric_grid_contract(
    lambda_m_per_vggt: torch.Tensor,
    *,
    extent_vggt: float,
    output_size: int,
) -> dict[str, torch.Tensor]:
    """Attach metric coordinates externally without resampling the BEV."""

    if lambda_m_per_vggt.ndim != 1:
        raise ValueError("runtime scale must have shape [B]")
    extent_m = lambda_m_per_vggt * float(extent_vggt)
    return {
        "extent_m": extent_m,
        "cell_size_m": extent_m / int(output_size),
        "bounds_m": torch.stack(
            (-extent_m / 2.0, extent_m / 2.0, -extent_m / 2.0, extent_m / 2.0),
            dim=-1,
        ),
    }
