from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LabelValues:
    """Lossless grayscale encoding used by the VGGNAV simulator export."""

    occupied: int = 0
    unknown: int = 112
    free: int = 255


@dataclass(frozen=True)
class BEVGridSpec:
    """Square ego BEV with right increasing in columns and forward toward row zero."""

    extent_m: float
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.extent_m <= 0:
            raise ValueError("extent_m must be positive")
        if self.height <= 1 or self.width <= 1:
            raise ValueError("BEV dimensions must be greater than one")
        if self.height != self.width:
            raise ValueError("Method II currently expects a square BEV")

    @property
    def meters_per_pixel(self) -> float:
        return self.extent_m / self.width

    @property
    def cell_size(self) -> float:
        return self.meters_per_pixel

    @property
    def half_extent(self) -> float:
        return self.extent_m / 2.0

    @property
    def center_x(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.height - 1) / 2.0

    def metric_to_pixel(self, right_forward: torch.Tensor) -> torch.Tensor:
        """Convert [..., (right, forward)] metres to [..., (column, row)]."""

        right = right_forward[..., 0]
        forward = right_forward[..., 1]
        column = right / self.meters_per_pixel + self.center_x
        row = self.center_y - forward / self.meters_per_pixel
        return torch.stack((column, row), dim=-1)

    def spatial_to_pixel(self, right_forward: torch.Tensor) -> torch.Tensor:
        return self.metric_to_pixel(right_forward)

    def pixel_to_metric(self, column_row: torch.Tensor) -> torch.Tensor:
        """Convert [..., (column, row)] to [..., (right, forward)] metres."""

        column = column_row[..., 0]
        row = column_row[..., 1]
        right = (column - self.center_x) * self.meters_per_pixel
        forward = (self.center_y - row) * self.meters_per_pixel
        return torch.stack((right, forward), dim=-1)


@dataclass(frozen=True)
class BatchBEVGridSpec:
    """Per-sample square grid for learned normalized VGGT coordinates."""

    extent: torch.Tensor
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.extent.ndim != 1 or self.extent.numel() < 1:
            raise ValueError("extent must have shape [B]")
        if not torch.isfinite(self.extent).all() or (self.extent <= 0).any():
            raise ValueError("every learned extent must be finite and positive")
        if self.height <= 1 or self.width <= 1:
            raise ValueError("BEV dimensions must be greater than one")
        if self.height != self.width:
            raise ValueError("Method II currently expects a square BEV")

    @property
    def cell_size(self) -> torch.Tensor:
        return self.extent / self.width

    @property
    def half_extent(self) -> torch.Tensor:
        return self.extent / 2.0

    @property
    def center_x(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.height - 1) / 2.0

    def _expand(self, values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape[0] != values.shape[0]:
            raise ValueError("grid batch size does not match spatial coordinates")
        return values.reshape(values.shape[0], *((1,) * (target.ndim - 2)))

    def spatial_to_pixel(self, right_forward: torch.Tensor) -> torch.Tensor:
        """Map normalized VGGT coordinates to per-sample raster coordinates."""

        cell_size = self._expand(self.cell_size, right_forward)
        right = right_forward[..., 0]
        forward = right_forward[..., 1]
        column = right / cell_size + self.center_x
        row = self.center_y - forward / cell_size
        return torch.stack((column, row), dim=-1)
