"""Python 3.9 smoke tests for the Habitat-side comparison support."""

import base64
import io
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from bev_accumulator import BEVAccumulator, UNKNOWN_VALUE
from model_comparison import (
    ComparisonFrame,
    ModelComparisonWorker,
    runtime_request_payload,
)


def identity_extrinsic():
    return {
        "agent_position_world_m": [0.0, 0.35, 0.0],
        "bev_forward_xz": [0.0, -1.0],
        "bev_right_xz": [1.0, 0.0],
        "camera_to_world_matrix": np.eye(4).tolist(),
        "world_from_bev_planar": np.eye(3).tolist(),
    }


class FakeRuntimeHandler(BaseHTTPRequestHandler):
    history_count = 0

    def _send(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(
            {
                "models": {
                    "3p5m": {"single_extent_m": 3.5},
                    "5m": {"single_extent_m": 5.0},
                    "6p5m": {"single_extent_m": 6.5},
                },
                "shared_vggt_backbone": True,
                "max_history": 34,
            }
        )

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        if self.path == "/reset":
            type(self).history_count = 0
            self._send({"reset": True, "segment_id": payload["segment_id"]})
            return
        type(self).history_count += 1
        image = Image.fromarray(np.full((32, 32), 112, dtype=np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        models = {}
        for model_key, extent_m in (
            ("3p5m", 3.5),
            ("5m", 5.0),
            ("6p5m", 6.5),
        ):
            models[model_key] = {
                "single_extent_m": extent_m,
                "depth_scale": 1.25,
                "checkpoint_epoch": 10,
                "checkpoint_global_step": 172820,
                "model_single_png_base64": encoded,
            }
        self._send(
            {
                "history_frame_count": type(self).history_count,
                "inference_seconds": 0.01,
                "shared_vggt_extraction": True,
                "models": models,
                "runtime_input_contract": {
                    "camera_rgb": True,
                    "external_intrinsics": False,
                    "external_camera_poses": False,
                    "ground_truth_trajectory": False,
                    "geometry_source": (
                        "VGGT-estimated intrinsics and relative poses"
                    ),
                },
            }
        )

    def log_message(self, format_string, *args):
        return


class ComparisonSupportTest(unittest.TestCase):
    def test_gt_accumulator_outputs_fixed_merged_grid(self):
        size = 32
        complete = np.full((size, size), 255, dtype=np.uint8)
        complete[5:8, 15:18] = 0
        masked = np.full((size, size), UNKNOWN_VALUE, dtype=np.uint8)
        masked[4:20, 8:24] = 255
        accumulator = BEVAccumulator(5.0, size)
        extrinsic = identity_extrinsic()

        accumulator.update(complete, masked, extrinsic)
        merged = accumulator.render_masked(extrinsic, 8.0)

        self.assertEqual(merged.shape, (size, size))
        self.assertTrue(np.any(merged == 0))
        self.assertTrue(np.any(merged == 255))
        self.assertTrue(np.any(merged == UNKNOWN_VALUE))

    def test_worker_deduplicates_static_motion_step(self):
        worker = ModelComparisonWorker(
            server_url="http://127.0.0.1:1",
            model_extents_m={"3p5m": 3.5, "5m": 5.0, "6p5m": 6.5},
            bev_size=32,
            sample_hz=1000.0,
            max_history=34,
        )
        frame = ComparisonFrame(
            frame_seq=1,
            motion_step=0,
            camera_rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            gt_masked_by_model={
                key: np.zeros((32, 32), dtype=np.uint8)
                for key in ("3p5m", "5m", "6p5m")
            },
        )
        self.assertTrue(worker.submit(frame))
        self.assertFalse(worker.submit(frame))

    def test_runtime_request_contains_rgb_but_no_gt_geometry(self):
        frame = ComparisonFrame(
            frame_seq=3,
            motion_step=2,
            camera_rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            gt_masked_by_model={
                key: np.zeros((32, 32), dtype=np.uint8)
                for key in ("3p5m", "5m", "6p5m")
            },
        )

        payload = runtime_request_payload(
            frame,
            segment_id="segment-test",
            threshold=0.5,
        )

        self.assertEqual(
            set(payload),
            {
                "segment_id",
                "frame_seq",
                "image_png_base64",
                "threshold",
            },
        )

    def test_worker_publishes_three_extent_gt_prediction_contract(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeRuntimeHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        worker = ModelComparisonWorker(
            server_url=f"http://127.0.0.1:{server.server_port}",
            model_extents_m={"3p5m": 3.5, "5m": 5.0, "6p5m": 6.5},
            bev_size=32,
            sample_hz=1000.0,
            max_history=34,
        )
        frame = ComparisonFrame(
            frame_seq=1,
            motion_step=1,
            camera_rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            gt_masked_by_model={
                key: np.full((32, 32), 112, dtype=np.uint8)
                for key in ("3p5m", "5m", "6p5m")
            },
        )
        try:
            worker.start()
            self.assertTrue(worker.submit(frame))
            deadline = time.monotonic() + 3.0
            snapshot = worker.snapshot()
            while (
                "predicted_png_base64"
                not in snapshot.get("models", {}).get("5m", {})
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
                snapshot = worker.snapshot()
            self.assertEqual(
                snapshot["status"],
                "Three-model comparison synchronized",
            )
            self.assertEqual(snapshot["history_frame_count"], 1)
            for model_key in ("3p5m", "5m", "6p5m"):
                model = snapshot["models"][model_key]
                self.assertTrue(model["gt_png_base64"])
                self.assertTrue(model["predicted_png_base64"])
        finally:
            worker.stop()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
