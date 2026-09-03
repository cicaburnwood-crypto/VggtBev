from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2", reason="TartanGround conversion requires OpenCV")
from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_tartanground_p1d_assets import validate_session  # noqa: E402


def _session(root: Path, *, known_pixels: int) -> Path:
    session = root / ".session_test.partial"
    for directory in (
        "camera",
        "depth",
        "bev_6p5m/masked",
        "bev_6p5m/complete",
        "bev_6p5m/merged_masked_10m",
        "bev_6p5m/merged_complete_10m",
    ):
        (session / directory).mkdir(parents=True, exist_ok=True)
    intrinsics = {
        "K": [[320.0, 0.0, 320.0], [0.0, 320.0, 320.0], [0.0, 0.0, 1.0]]
    }
    (session / "metadata.json").write_text(
        json.dumps({"frame_count": 1, "camera_intrinsics": intrinsics}),
        encoding="utf-8",
    )
    (session / "camera_intrinsics.json").write_text(
        json.dumps(intrinsics), encoding="utf-8"
    )
    (session / "camera_extrinsics.jsonl").write_text(
        json.dumps({"frame_id": 0}) + "\n", encoding="utf-8"
    )
    Image.new("RGB", (640, 640)).save(session / "camera/frame_000000.png")
    np.savez_compressed(
        session / "depth/frame_000000.npz",
        depth=np.ones((640, 640), dtype=np.float32),
    )
    complete = np.full((512, 512), 112, dtype=np.uint8)
    observed = np.full_like(complete, 112)
    side = int(np.ceil(np.sqrt(known_pixels)))
    complete[:side, :side] = 255
    observed[:side, :side] = 255
    center = complete.shape[0] // 2
    complete[center, center] = 255
    observed[center, center] = 255
    for complete_directory, observed_directory in (
        ("complete", "masked"),
        ("merged_complete_10m", "merged_masked_10m"),
    ):
        Image.fromarray(complete).save(
            session / "bev_6p5m" / complete_directory / "frame_000000.png"
        )
        Image.fromarray(observed).save(
            session / "bev_6p5m" / observed_directory / "frame_000000.png"
        )
    return session


def test_observed_support_below_threshold_is_rejected(tmp_path: Path) -> None:
    session = _session(tmp_path, known_pixels=9)
    with pytest.raises(ValueError, match="insufficient known support"):
        validate_session(
            session,
            expected_frames=1,
            require_complete=False,
            minimum_observed_known_pixels=1024,
        )


def test_observed_support_at_threshold_is_accepted(tmp_path: Path) -> None:
    session = _session(tmp_path, known_pixels=1024)
    result = validate_session(
        session,
        expected_frames=1,
        require_complete=False,
        minimum_observed_known_pixels=1024,
    )
    assert int(result["minimum_observed_known_pixels"]) >= 1024
