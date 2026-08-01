import base64
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import torch
from PIL import Image

from vggt_bev.runtime import RuntimePrediction, RuntimeSinglePrediction
from vggt_bev.runtime_server import (
    HistoryFullError,
    RuntimeHistory,
    make_runtime_handler,
)


class FakeRuntime:
    image_height = 32
    image_width = 32
    extent_key = "bev_5m"
    single_extent_m = 5.0
    merged_extent_m = 8.0
    output_size = 4
    checkpoint_epoch = 10
    checkpoint_global_step = 172820
    coordinate_mode = "metric"

    def __init__(self) -> None:
        self.sequences = []

    def predict(self, sequence, *, threshold):
        self.sequences.append(sequence)
        labels = torch.tensor(
            [
                [112, 112, 112, 112],
                [112, 255, 0, 112],
                [112, 255, 255, 112],
                [112, 112, 112, 112],
            ],
            dtype=torch.uint8,
        )
        probability = torch.full((4, 4), threshold)
        return RuntimePrediction(
            single_labels=labels,
            merged_labels=labels.flip(0),
            geometry_single_labels=labels.flip(1),
            geometry_merged_labels=labels.rot90(),
            single_occupancy_probability=probability,
            single_observed_probability=probability,
            merged_occupancy_probability=probability,
            merged_observed_probability=probability,
            depth_scale=1.25,
            single_extent_m=self.single_extent_m,
            merged_extent_m=self.merged_extent_m,
            output_size=self.output_size,
            checkpoint_epoch=self.checkpoint_epoch,
            checkpoint_global_step=self.checkpoint_global_step,
            geometry_estimated_intrinsic=torch.eye(3),
        )


class FakeMultiRuntime:
    image_height = 32
    image_width = 32

    def predict(self, sequence, *, threshold):
        labels = torch.tensor(
            [[112, 112], [255, 0]],
            dtype=torch.uint8,
        )
        probability = torch.full((2, 2), threshold)
        return {
            model_key: RuntimeSinglePrediction(
                labels=labels,
                occupancy_probability=probability,
                observed_probability=probability,
                depth_scale=depth_scale,
                extent_m=extent_m,
                output_size=2,
                checkpoint_epoch=10,
                checkpoint_global_step=172820,
            )
            for model_key, extent_m, depth_scale in (
                ("3p5m", 3.5, 2.60),
                ("5m", 5.0, 2.59),
                ("6p5m", 6.5, 2.68),
            )
        }


def image_payload(frame_seq: int, segment_id: str = "segment-a") -> dict:
    image = Image.fromarray(np.full((24, 40, 3), 127, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {
        "segment_id": segment_id,
        "frame_seq": frame_seq,
        "image_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "threshold": 0.5,
    }


def test_live_payload_contains_no_gt_camera_geometry() -> None:
    assert set(image_payload(1)) == {
        "segment_id",
        "frame_seq",
        "image_png_base64",
        "threshold",
    }


def request_json(url: str, method: str = "GET", payload: dict | None = None):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def test_runtime_history_is_cumulative_and_bounded() -> None:
    runtime = FakeRuntime()
    history = RuntimeHistory(runtime, max_history=2)

    first = history.predict(image_payload(1))
    second = history.predict(image_payload(2))

    assert first["history_frame_count"] == 1
    assert second["history_frame_count"] == 2
    assert runtime.sequences[0].images.shape == (1, 3, 32, 32)
    assert runtime.sequences[1].images.shape == (2, 3, 32, 32)
    assert runtime.sequences[1].frame_ids == (1, 2)
    assert runtime.sequences[1].camera_height_m is None
    with pytest.raises(HistoryFullError):
        history.predict(image_payload(3))

    history.reset("segment-b")
    reset = history.predict(image_payload(10, "segment-b"))
    assert reset["history_frame_count"] == 1
    assert runtime.sequences[-1].frame_ids == (10,)


def test_multi_runtime_history_returns_three_single_predictions() -> None:
    history = RuntimeHistory(FakeMultiRuntime(), max_history=2)

    result = history.predict(image_payload(1))

    assert result["shared_vggt_extraction"] is True
    assert set(result["models"]) == {"3p5m", "5m", "6p5m"}
    assert result["models"]["3p5m"]["single_extent_m"] == 3.5
    assert result["models"]["5m"]["depth_scale"] == 2.59
    assert result["models"]["6p5m"]["output_size"] == 2
    assert result["runtime_input_contract"]["external_intrinsics"] is False
    with Image.open(
        io.BytesIO(
            base64.b64decode(
                result["models"]["5m"]["model_single_png_base64"]
            )
        )
    ) as image:
        assert image.size == (2, 2)
        assert set(image.getdata()) == {0, 112, 255}


def test_runtime_http_contract() -> None:
    runtime = FakeRuntime()
    history = RuntimeHistory(runtime, max_history=2)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_runtime_handler(history))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        status, health = request_json(f"{base_url}/health")
        assert status == 200
        assert health["single_extent_m"] == 5.0
        assert health["max_history"] == 2
        assert health["geometry_source"] == "VGGT-estimated intrinsics and poses"

        status, reset = request_json(
            f"{base_url}/reset",
            "POST",
            {"segment_id": "live-1"},
        )
        assert status == 200
        assert reset["history_frame_count"] == 0

        status, result = request_json(
            f"{base_url}/predict",
            "POST",
            image_payload(7, "live-1"),
        )
        assert status == 200
        assert result["history_frame_count"] == 1
        with Image.open(
            io.BytesIO(base64.b64decode(result["model_single_png_base64"]))
        ) as image:
            assert image.size == (4, 4)
            assert set(image.getdata()) == {0, 112, 255}
        assert result["geometry_projection"]["uses_bev_head"] is False
        assert result["runtime_input_contract"]["external_intrinsics"] is False
        assert result["runtime_input_contract"]["external_camera_poses"] is False
        with Image.open(
            io.BytesIO(base64.b64decode(result["geometry_single_png_base64"]))
        ) as image:
            assert image.size == (4, 4)

        with pytest.raises(urllib.error.HTTPError) as captured:
            request_json(
                f"{base_url}/predict",
                "POST",
                image_payload(7, "live-1"),
            )
        assert captured.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
