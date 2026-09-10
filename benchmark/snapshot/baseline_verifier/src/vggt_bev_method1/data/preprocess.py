from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


class RGBResizePad:
    """Aspect-preserving RGB resize with white padding and [0, 1] output."""

    def __init__(self, height: int = 384, width: int = 512) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("output dimensions must be positive")
        self.height = height
        self.width = width
        self.version = "rgb-depth-resize-pad-v2"

    def parameters(
        self,
        source_height: int,
        source_width: int,
    ) -> tuple[int, int, int, int]:
        scale = min(
            self.width / source_width,
            self.height / source_height,
        )
        resized_width = max(1, round(source_width * scale))
        resized_height = max(1, round(source_height * scale))
        left = (self.width - resized_width) // 2
        top = (self.height - resized_height) // 2
        return resized_height, resized_width, top, left

    def __call__(self, image: Image.Image) -> torch.Tensor:
        rgb = image.convert("RGB")
        resized_height, resized_width, top, left = self.parameters(
            rgb.height,
            rgb.width,
        )
        resized = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.width, self.height), color=(255, 255, 255))
        canvas.paste(resized, (left, top))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def depth(
        self,
        depth_m: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the exact RGB geometry to metric z-depth and its valid mask."""

        if depth_m.ndim != 2:
            raise ValueError("metric depth must have shape [H, W]")
        source_height, source_width = depth_m.shape
        resized_height, resized_width, top, left = self.parameters(
            source_height,
            source_width,
        )
        depth = torch.from_numpy(
            np.asarray(depth_m, dtype=np.float32).copy()
        )[None, None]
        resized = F.interpolate(
            depth,
            size=(resized_height, resized_width),
            mode="nearest",
        )[0, 0]
        output = torch.zeros(self.height, self.width, dtype=torch.float32)
        valid = torch.zeros(self.height, self.width, dtype=torch.bool)
        output[
            top : top + resized_height,
            left : left + resized_width,
        ] = resized
        valid_region = torch.isfinite(resized) & (resized > 0)
        valid[
            top : top + resized_height,
            left : left + resized_width,
        ] = valid_region
        output[~valid] = 0.0
        return output, valid

    def intrinsics(
        self,
        matrix: np.ndarray,
        *,
        source_height: int,
        source_width: int,
    ) -> torch.Tensor:
        """Transform K with the same resize and padding used for RGB/depth."""

        if matrix.shape != (3, 3):
            raise ValueError("camera intrinsics must be a 3x3 matrix")
        resized_height, resized_width, top, left = self.parameters(
            source_height,
            source_width,
        )
        scale_x = resized_width / source_width
        scale_y = resized_height / source_height
        output = torch.from_numpy(
            np.asarray(matrix, dtype=np.float32).copy()
        )
        output[0, 0] *= scale_x
        output[1, 1] *= scale_y
        output[0, 2] = (output[0, 2] + 0.5) * scale_x - 0.5 + left
        output[1, 2] = (output[1, 2] + 0.5) * scale_y - 0.5 + top
        return output
