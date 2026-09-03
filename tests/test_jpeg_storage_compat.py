from pathlib import Path

import numpy as np
from PIL import Image

from vggt_bev_method1.data.dataset import (
    VGGNAVMethod1Dataset,
    _resolve_raster_path,
)


def test_resolve_raster_path_falls_back_to_jpeg(tmp_path: Path) -> None:
    jpeg = tmp_path / "frame_000000.jpeg"
    Image.new("RGB", (8, 8), (20, 40, 60)).save(
        jpeg,
        "JPEG",
        quality=95,
        subsampling=0,
    )
    assert _resolve_raster_path(tmp_path / "frame_000000.png") == jpeg


def test_jpeg_bev_is_restored_to_exact_gt_palette(tmp_path: Path) -> None:
    labels = np.full((512, 512), 255, dtype=np.uint8)
    labels[20:180, 40:200] = 0
    labels[210:430, 100:420] = 112
    labels[100:400:7, 12:500] = 0
    path = tmp_path / "frame_000000.jpeg"
    Image.fromarray(labels).save(path, "JPEG", quality=95, optimize=True)

    restored = VGGNAVMethod1Dataset._load_bev(
        tmp_path / "frame_000000.png",
        source_extent_m=6.5,
        output_extent_m=6.5,
        output_size=512,
    )
    np.testing.assert_array_equal(restored.numpy(), labels)
