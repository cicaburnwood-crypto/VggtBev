from pathlib import Path

import numpy as np

from vggt_bev_method1.data.dataset import VGGNAVMethod1Dataset


def test_compact_uint16_depth_is_restored_in_metres(tmp_path: Path) -> None:
    path = tmp_path / "frame_000000.npz"
    quantized = np.asarray([[0, 1250], [65535, 9995]], dtype=np.uint16)
    np.savez_compressed(
        path,
        depth_q=quantized,
        scale_m=np.asarray(0.002, dtype=np.float32),
        invalid_q=np.asarray(65535, dtype=np.uint16),
        format_version=np.asarray(1, dtype=np.uint8),
    )
    depth = VGGNAVMethod1Dataset._load_depth(path)
    np.testing.assert_allclose(
        depth,
        np.asarray([[0.0, 2.5], [0.0, 19.99]], dtype=np.float32),
        rtol=0.0,
        atol=3e-6,
    )


def test_legacy_float_depth_remains_supported(tmp_path: Path) -> None:
    path = tmp_path / "frame_000000.npz"
    expected = np.asarray([[0.5, 1.25], [2.0, 3.0]], dtype=np.float32)
    np.savez_compressed(path, depth=expected)
    np.testing.assert_array_equal(
        VGGNAVMethod1Dataset._load_depth(path), expected
    )
