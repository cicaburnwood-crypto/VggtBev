from __future__ import annotations

import torch

from vggt_bev_method1.config import LabelValues
from vggt_bev_method1.m04_train_utils import build_m04_datasets, m04_collate

M05_REQUIRED_LATEST_TARGETS = (
    "latest_fov_complete_target",
    "latest_visible_target",
    "latest_fov_support_target",
    "latest_gt_valid_mask",
)


def m05_collate(samples: list[dict]) -> dict:
    output = m04_collate(samples)
    missing = [key for key in M05_REQUIRED_LATEST_TARGETS if key not in output]
    if missing:
        raise KeyError(f"M05 batch is missing latest-frame targets: {missing}")
    return output


def build_m05_datasets(config: dict, *, verify_manifest: bool = True):
    datasets = build_m04_datasets(
        config,
        verify_manifest=verify_manifest,
        include_latest_temporal_targets=True,
    )
    return datasets


def fixed_metric_m05_target(
    complete: torch.Tensor,
    visible: torch.Tensor,
    support: torch.Tensor,
    gt_valid: torch.Tensor,
    *,
    extent_m: float,
    latest_observed_free: torch.Tensor | None = None,
    latest_support: torch.Tensor | None = None,
    labels: LabelValues | None = None,
) -> dict[str, torch.Tensor]:
    """Return native 10 m M05 supervision without scale-dependent regridding.

    M05 predicts a fixed metric raster.  Consequently BEV supervision is a
    direct pixel-aligned view of the existing 10 m GT and remains valid even
    when the independent Scale Token has no usable label for this window.
    """

    if labels is None:
        labels = LabelValues()

    tensors = (complete, visible, support, gt_valid)
    if any(value.ndim != 3 for value in tensors):
        raise ValueError("fixed-metric M05 targets must have shape [B,H,W]")
    if any(value.shape != complete.shape for value in tensors[1:]):
        raise ValueError("fixed-metric M05 targets must align exactly")
    if complete.shape[-1] != complete.shape[-2]:
        raise ValueError("fixed-metric M05 targets must be square")
    if extent_m <= 0.0:
        raise ValueError("fixed-metric M05 extent must be positive")
    if (latest_observed_free is None) != (latest_support is None):
        raise ValueError(
            "latest observed-free and support targets must be supplied together"
        )
    if latest_observed_free is not None and (
        latest_observed_free.shape != complete.shape
        or latest_support.shape != complete.shape
    ):
        raise ValueError("latest temporal targets must align with the 10 m grid")

    unknown = int(labels.unknown)
    support = support.bool()
    gt_valid = gt_valid.bool()
    complete = complete.to(torch.uint8)
    visible = visible.to(torch.uint8)
    complete = torch.where(
        support, complete, torch.full_like(complete, unknown)
    )
    visible_known = support & (visible != unknown)
    visible = torch.where(
        visible_known, visible, torch.full_like(visible, unknown)
    )
    if bool((visible_known & (visible != complete)).any()):
        raise ValueError("fixed-metric visible labels disagree with complete GT")

    batch = complete.shape[0]
    size = complete.shape[-1]
    output = {
        "complete_target": complete,
        "visible_target": visible,
        "support_target": support,
        "gt_valid_mask": gt_valid,
        "coordinate_coverage_mask": torch.ones_like(support),
        "coordinate_coverage_fraction": torch.ones(
            batch, device=complete.device, dtype=torch.float32
        ),
        "metric_extent_m": torch.full(
            (batch,), float(extent_m), device=complete.device, dtype=torch.float32
        ),
        "cell_size_m_gt": torch.full(
            (batch,), float(extent_m) / size,
            device=complete.device,
            dtype=torch.float32,
        ),
    }
    if latest_observed_free is not None:
        latest_support = latest_support.bool() & gt_valid
        latest_observed_free = latest_observed_free.bool() & latest_support
        output.update(
            {
                "latest_observed_free_target": latest_observed_free,
                "latest_support_target": latest_support,
                "history_observed_free_region": (
                    visible_known
                    & (visible == int(labels.free))
                    & ~latest_observed_free
                    & gt_valid
                ),
                "history_support_region": support & ~latest_support & gt_valid,
            }
        )
    return output
