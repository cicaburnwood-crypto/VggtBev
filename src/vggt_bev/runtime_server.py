from __future__ import annotations

import base64
import io
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
from PIL import Image

from vggt_bev.data import CalibrationAwareResize
from vggt_bev.runtime import (
    Method2Runtime,
    MultiMethod2Runtime,
    RuntimePrediction,
    RuntimeSequence,
    RuntimeSinglePrediction,
)


class HistoryFullError(RuntimeError):
    """Raised when a cumulative segment reaches the configured training limit."""


def _tensor_png_base64(labels: torch.Tensor) -> str:
    buffer = io.BytesIO()
    Image.fromarray(labels.numpy(), mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class RuntimeHistory:
    """Stateful cumulative frame history owned by the inference service."""

    def __init__(
        self,
        runtime: Method2Runtime | MultiMethod2Runtime,
        max_history: int = 34,
        camera_height_m: float | None = None,
    ) -> None:
        if max_history < 1:
            raise ValueError("max_history must be positive")
        if camera_height_m is not None and not 0.0 < camera_height_m < 5.0:
            raise ValueError("camera_height_m must be positive when supplied")
        self.runtime = runtime
        self.max_history = int(max_history)
        self.camera_height_m = (
            float(camera_height_m)
            if camera_height_m is not None
            else None
        )
        self.preprocess = CalibrationAwareResize(
            runtime.image_height,
            runtime.image_width,
        )
        self.lock = threading.Lock()
        self.segment_id: str | None = None
        self.frame_ids: list[int] = []
        self.images: list[torch.Tensor] = []
        self.image_valid: list[torch.Tensor] = []

    def reset(self, segment_id: str) -> None:
        if not segment_id:
            raise ValueError("segment_id must not be empty")
        with self.lock:
            self._reset_unlocked(segment_id)

    def _reset_unlocked(self, segment_id: str) -> None:
        self.segment_id = segment_id
        self.frame_ids.clear()
        self.images.clear()
        self.image_valid.clear()

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "segment_id": self.segment_id,
                "history_frame_count": len(self.frame_ids),
                "max_history": self.max_history,
                "camera_height_m": self.camera_height_m,
                "geometry_source": "VGGT-estimated intrinsics and poses",
            }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        segment_id = str(payload["segment_id"])
        if not segment_id:
            raise ValueError("segment_id must not be empty")
        frame_seq = int(payload["frame_seq"])
        if frame_seq < 0:
            raise ValueError("frame_seq must be non-negative")
        image_data = base64.b64decode(payload["image_png_base64"], validate=True)
        try:
            image = Image.open(io.BytesIO(image_data))
            image.load()
        except Exception as error:
            raise ValueError("image_png_base64 is not a decodable image") from error
        image_tensor, _, valid = self.preprocess(
            image,
            torch.eye(3, dtype=torch.float32),
        )

        with self.lock:
            if self.segment_id != segment_id:
                self._reset_unlocked(segment_id)
            if self.frame_ids and frame_seq <= self.frame_ids[-1]:
                raise ValueError("frame_seq must increase within a segment")
            if len(self.frame_ids) >= self.max_history:
                raise HistoryFullError(
                    "cumulative history is full; start a new segment_id before predicting"
                )
            self.frame_ids.append(frame_seq)
            self.images.append(image_tensor)
            self.image_valid.append(valid)
            sequence = RuntimeSequence(
                images=torch.stack(self.images),
                image_valid=torch.stack(self.image_valid),
                camera_height_m=(
                    torch.tensor(
                        self.camera_height_m,
                        dtype=torch.float32,
                    )
                    if self.camera_height_m is not None
                    else None
                ),
                frame_ids=tuple(self.frame_ids),
                session_path=Path("live-simulator"),
                target_frame_id=frame_seq,
            )
            started = time.perf_counter()
            try:
                prediction = self.runtime.predict(
                    sequence,
                    threshold=float(payload.get("threshold", 0.5)),
                )
            except Exception:
                self.frame_ids.pop()
                self.images.pop()
                self.image_valid.pop()
                raise
            inference_seconds = time.perf_counter() - started
            response_arguments = {
                "segment_id": segment_id,
                "frame_seq": frame_seq,
                "history_frame_count": len(self.frame_ids),
                "inference_seconds": inference_seconds,
            }
            if isinstance(prediction, dict):
                return multi_prediction_response(
                    prediction,
                    **response_arguments,
                )
            return prediction_response(prediction, **response_arguments)


def prediction_response(
    prediction: RuntimePrediction,
    *,
    segment_id: str,
    frame_seq: int,
    history_frame_count: int,
    inference_seconds: float,
) -> dict[str, Any]:
    if prediction.coordinate_mode == "vggt_normalized":
        extent_fields = {
            "single_span_normalized_vggt": prediction.single_extent_m,
            "merged_span_normalized_vggt": prediction.merged_extent_m,
            "normalized_units_per_output_pixel": (
                prediction.normalized_units_per_output_pixel
            ),
            "reference_scale_vggt": prediction.reference_scale_vggt,
            "range_source": "learned_per_sequence",
        }
    else:
        extent_fields = (
        {
            "single_extent_vggt_units": prediction.single_extent_m,
            "merged_extent_vggt_units": prediction.merged_extent_m,
        }
        if prediction.coordinate_mode == "vggt_raw"
        else {
            "single_extent_m": prediction.single_extent_m,
            "merged_extent_m": prediction.merged_extent_m,
        }
        )
    return {
        "segment_id": segment_id,
        "frame_seq": frame_seq,
        "history_frame_count": history_frame_count,
        "inference_seconds": inference_seconds,
        "checkpoint_epoch": prediction.checkpoint_epoch,
        "checkpoint_global_step": prediction.checkpoint_global_step,
        "depth_scale": prediction.depth_scale,
        "coordinate_mode": prediction.coordinate_mode,
        **extent_fields,
        "output_size": prediction.output_size,
        "merged_output_size": (
            prediction.merged_output_size or prediction.output_size
        ),
        "runtime_input_contract": {
            "camera_rgb": True,
            "external_intrinsics": False,
            "external_camera_poses": False,
            "ground_truth_trajectory": False,
            "geometry_source": "VGGT-estimated intrinsics and relative poses",
        },
        "model_single_png_base64": _tensor_png_base64(prediction.single_labels),
        "model_merged_png_base64": _tensor_png_base64(prediction.merged_labels),
        "geometry_single_png_base64": _tensor_png_base64(
            prediction.geometry_single_labels
        ),
        "geometry_merged_png_base64": _tensor_png_base64(
            prediction.geometry_merged_labels
        ),
        "geometry_estimated_intrinsic": (
            prediction.geometry_estimated_intrinsic.tolist()
        ),
        "geometry_projection": {
            "uses_bev_head": False,
            "intrinsic_source": "VGGT camera head",
            "depth_source": "VGGT dense head",
            "pose_source": "VGGT camera head",
            "spatial_scale_source": (
                "raw VGGT reconstruction coordinates"
                if prediction.metric_scale_mode == "vggt_raw"
                else "legacy calibrated scale"
            ),
        },
    }


def multi_prediction_response(
    predictions: dict[str, RuntimeSinglePrediction],
    *,
    segment_id: str,
    frame_seq: int,
    history_frame_count: int,
    inference_seconds: float,
) -> dict[str, Any]:
    return {
        "segment_id": segment_id,
        "frame_seq": frame_seq,
        "history_frame_count": history_frame_count,
        "inference_seconds": inference_seconds,
        "shared_vggt_extraction": True,
        "runtime_input_contract": {
            "camera_rgb": True,
            "external_intrinsics": False,
            "external_camera_poses": False,
            "ground_truth_trajectory": False,
            "geometry_source": "VGGT-estimated intrinsics and relative poses",
        },
        "models": {
            model_key: {
                (
                    "single_span_normalized_vggt"
                    if prediction.coordinate_mode == "vggt_normalized"
                    else (
                        "single_extent_vggt_units"
                        if prediction.coordinate_mode == "vggt_raw"
                        else "single_extent_m"
                    )
                ): prediction.extent_m,
                "coordinate_mode": prediction.coordinate_mode,
                "output_size": prediction.output_size,
                "checkpoint_epoch": prediction.checkpoint_epoch,
                "checkpoint_global_step": prediction.checkpoint_global_step,
                "depth_scale": prediction.depth_scale,
                "normalized_units_per_output_pixel": (
                    prediction.normalized_units_per_output_pixel
                ),
                "reference_scale_vggt": (
                    prediction.reference_scale_vggt
                ),
                "model_single_png_base64": _tensor_png_base64(
                    prediction.labels
                ),
            }
            for model_key, prediction in predictions.items()
        },
    }


def make_runtime_handler(history: RuntimeHistory) -> type[BaseHTTPRequestHandler]:
    class RuntimeRequestHandler(BaseHTTPRequestHandler):
        server_version = "VGGTBEVRuntime/0.4"

        def _send_json(
            self,
            payload: dict[str, Any],
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if urlparse(self.path).path != "/health":
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            if isinstance(history.runtime, MultiMethod2Runtime):
                runtime_status = {
                    "models": history.runtime.metadata(),
                    "shared_vggt_backbone": True,
                }
            else:
                if history.runtime.coordinate_mode == "vggt_normalized":
                    extent_status = {
                        "range_source": "learned_per_sequence",
                        "single_output_size": history.runtime.output_size,
                        "merged_output_size": (
                            history.runtime.merged_output_size
                        ),
                    }
                else:
                    extent_status = (
                    {
                        "single_extent_vggt_units": (
                            history.runtime.single_extent_m
                        ),
                        "merged_extent_vggt_units": (
                            history.runtime.merged_extent_m
                        ),
                    }
                    if history.runtime.coordinate_mode == "vggt_raw"
                    else {
                        "single_extent_m": history.runtime.single_extent_m,
                        "merged_extent_m": history.runtime.merged_extent_m,
                    }
                    )
                runtime_status = {
                    "extent_key": history.runtime.extent_key,
                    "coordinate_mode": history.runtime.coordinate_mode,
                    **extent_status,
                    "checkpoint_epoch": history.runtime.checkpoint_epoch,
                    "checkpoint_global_step": (
                        history.runtime.checkpoint_global_step
                    ),
                }
            self._send_json(
                {
                    "ready": True,
                    **runtime_status,
                    **history.status(),
                }
            )

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ValueError("request size must be between 1 byte and 32 MiB")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                if path == "/predict":
                    self._send_json(history.predict(payload))
                elif path == "/reset":
                    history.reset(str(payload["segment_id"]))
                    self._send_json({"reset": True, **history.status()})
                else:
                    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except HistoryFullError as error:
                self._send_json({"error": str(error)}, HTTPStatus.CONFLICT)
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:
                self._send_json(
                    {"error": f"runtime inference failed: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def log_message(self, format_string: str, *args: Any) -> None:
            if "/health" not in str(args[0]):
                super().log_message(format_string, *args)

    return RuntimeRequestHandler


def serve_runtime(
    history: RuntimeHistory,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    server = ThreadingHTTPServer((host, port), make_runtime_handler(history))
    server.daemon_threads = True
    if isinstance(history.runtime, MultiMethod2Runtime):
        model_summary = ", ".join(
            (
                f"{key}={metadata['single_extent_vggt_units']:g} raw-VGGT-units"
                if metadata["coordinate_mode"] == "vggt_raw"
                else f"{key}={metadata['single_extent_m']:g}m"
            )
            for key, metadata in history.runtime.metadata().items()
        )
        description = f"models: {model_summary}; one shared VGGT backbone"
    else:
        if history.runtime.coordinate_mode == "vggt_normalized":
            description = (
                f"learned normalized range; "
                f"single={history.runtime.output_size}px "
                f"merged={history.runtime.merged_output_size}px"
            )
        else:
            suffix = (
            " raw-VGGT-units"
            if history.runtime.coordinate_mode == "vggt_raw"
            else "m"
            )
            description = (
                f"single={history.runtime.single_extent_m:g}{suffix} "
                f"merged={history.runtime.merged_extent_m:g}{suffix}"
            )
    print(
        f"VGGTBEV runtime ready: http://{host}:{port} {description}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping VGGTBEV runtime…", flush=True)
    finally:
        server.server_close()
