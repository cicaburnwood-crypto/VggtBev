"""Asynchronous client joining Habitat ground truth with VGGTBEV predictions."""

from __future__ import annotations

import base64
import io
import json
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from PIL import Image


def encode_png_base64(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass(frozen=True)
class ComparisonFrame:
    frame_seq: int
    motion_step: int
    camera_rgb: np.ndarray
    gt_complete_by_model: dict[str, np.ndarray]
    gt_observed_by_model: dict[str, np.ndarray]
    gt_guessed_by_model: dict[str, np.ndarray]
    gt_merged_observed_by_model: dict[str, np.ndarray]


def runtime_request_payload(
    frame: ComparisonFrame,
    *,
    segment_id: str,
    threshold: float,
) -> dict[str, Any]:
    """Build the RGB-only model request; simulator geometry stays client-side."""

    return {
        "segment_id": segment_id,
        "frame_seq": frame.frame_seq,
        "image_png_base64": encode_png_base64(frame.camera_rgb[..., :3]),
        "threshold": threshold,
    }


class ModelComparisonWorker:
    def __init__(
        self,
        *,
        server_url: str,
        model_extents_m: dict[str, float],
        bev_size: int,
        sample_hz: float,
        max_history: int,
        threshold: float = 0.5,
    ) -> None:
        if sample_hz <= 0 or max_history < 1:
            raise ValueError("sample_hz and max_history must be positive")
        if not model_extents_m:
            raise ValueError("at least one model extent is required")
        self.server_url = server_url.rstrip("/")
        self.model_extents_m = {
            key: float(extent) for key, extent in model_extents_m.items()
        }
        self.bev_size = int(bev_size)
        self.sample_interval = 1.0 / sample_hz
        self.max_history = int(max_history)
        self.threshold = float(threshold)
        self.frames: queue.Queue[Optional[ComparisonFrame]] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="vggtbev-comparison",
            daemon=True,
        )
        self.lock = threading.Lock()
        self.next_sample_time = 0.0
        self.last_submitted_motion_step: Optional[int] = None
        self.segment_counter = 0
        self.segment_id = ""
        self.history_count = 0
        self.runtime_mode = "unknown"
        self.latest: Optional[dict[str, Any]] = None
        self.state: dict[str, Any] = {
            "ready": False,
            "status": "Waiting for model runtime",
            "server_url": self.server_url,
            "model_extents_m": dict(self.model_extents_m),
            "max_history": self.max_history,
            "history_frame_count": 0,
            "inference_seconds": None,
            "last_frame_seq": None,
            "segment_id": None,
        }

    def start(self) -> None:
        health = self._request_json("GET", "/health")
        health_models = health.get("models")
        if isinstance(health_models, dict):
            self.runtime_mode = "multi"
        else:
            if len(self.model_extents_m) != 1:
                raise RuntimeError(
                    "single-model runtime cannot serve multiple comparison rows"
                )
            model_key = next(iter(self.model_extents_m))
            health_models = {
                model_key: {
                    "single_extent_m": health["single_extent_m"],
                    "checkpoint_epoch": health["checkpoint_epoch"],
                    "checkpoint_global_step": health[
                        "checkpoint_global_step"
                    ],
                }
            }
            self.runtime_mode = "single"
        for model_key, expected_extent in self.model_extents_m.items():
            if model_key not in health_models:
                raise RuntimeError(f"model server is missing {model_key}")
            actual = float(health_models[model_key]["single_extent_m"])
            if abs(actual - expected_extent) > 1e-6:
                raise RuntimeError(
                    f"{model_key} server extent is {actual:g} m, "
                    f"UI expects {expected_extent:g} m"
                )
        if int(health["max_history"]) != self.max_history:
            raise RuntimeError(
                "model server and UI must use the same maximum history length"
            )
        with self.lock:
            self.state.update(
                {
                    "ready": True,
                    "status": (
                        f"{len(self.model_extents_m)} model(s) ready; "
                        "waiting for first synchronized frame"
                    ),
                    "models": health_models,
                    "runtime_mode": self.runtime_mode,
                    "shared_vggt_backbone": bool(
                        health.get("shared_vggt_backbone")
                    ),
                }
            )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.frames.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=5.0)

    def submit(self, frame: ComparisonFrame) -> bool:
        now = time.monotonic()
        if (
            self.last_submitted_motion_step == frame.motion_step
            or now < self.next_sample_time
        ):
            return False
        self.last_submitted_motion_step = frame.motion_step
        self.next_sample_time = now + self.sample_interval
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(frame)
        return True

    def should_sample(self, motion_step: int) -> bool:
        """Cheap preflight before copying the synchronized RGB and single GT."""

        now = time.monotonic()
        return (
            self.last_submitted_motion_step != motion_step
            and now >= self.next_sample_time
        )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            payload = dict(self.state)
            if self.latest is not None:
                payload.update(self.latest)
            return json.loads(json.dumps(payload))

    def status(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.state))

    def _new_segment(self) -> None:
        self.segment_counter += 1
        self.segment_id = f"segment-{self.segment_counter:06d}"
        self.history_count = 0
        self._request_json(
            "POST",
            "/reset",
            {"segment_id": self.segment_id},
        )

    def _run(self) -> None:
        while not self.stop_event.is_set():
            if not self.segment_id:
                try:
                    self._new_segment()
                except Exception as error:
                    with self.lock:
                        self.state.update(
                            {
                                "ready": False,
                                "status": f"Model reset error: {error}",
                            }
                        )
                    self.stop_event.wait(1.0)
                    continue
            try:
                frame = self.frames.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                return
            try:
                response = self._request_json(
                    "POST",
                    "/predict",
                    runtime_request_payload(
                        frame,
                        segment_id=self.segment_id,
                        threshold=self.threshold,
                    ),
                    timeout=180.0,
                )
                self.history_count = int(response["history_frame_count"])
                response_models = response.get("models")
                if not isinstance(response_models, dict):
                    if len(self.model_extents_m) != 1:
                        raise RuntimeError(
                            "runtime response has no multi-model outputs"
                        )
                    response_models = {
                        next(iter(self.model_extents_m)): response
                    }
                latest_models = {}
                for model_key, extent_m in self.model_extents_m.items():
                    prediction = response_models[model_key]
                    actual_extent = float(prediction["single_extent_m"])
                    if abs(actual_extent - extent_m) > 1e-6:
                        raise RuntimeError(
                            f"{model_key} prediction extent changed to "
                            f"{actual_extent:g} m"
                        )
                    latest_models[model_key] = {
                        "extent_m": extent_m,
                        "gt_png_base64": encode_png_base64(
                            frame.gt_complete_by_model[model_key]
                        ),
                        "gt_complete_png_base64": encode_png_base64(
                            frame.gt_complete_by_model[model_key]
                        ),
                        "gt_observed_png_base64": encode_png_base64(
                            frame.gt_observed_by_model[model_key]
                        ),
                        "gt_guessed_png_base64": encode_png_base64(
                            frame.gt_guessed_by_model[model_key]
                        ),
                        "gt_merged_observed_png_base64": encode_png_base64(
                            frame.gt_merged_observed_by_model[model_key]
                        ),
                        "predicted_png_base64": prediction[
                            "model_single_png_base64"
                        ],
                        "predicted_merged_png_base64": prediction.get(
                            "model_merged_png_base64"
                        ),
                        "merged_score_png_base64": prediction.get(
                            "merged_score_png_base64"
                        ),
                        "confidence_single_png_base64": prediction.get(
                            "confidence_single_png_base64"
                        ),
                        "observed_gate_single_png_base64": prediction.get(
                            "observed_gate_single_png_base64"
                        ),
                        "observed_gate_confidence_png_base64": prediction.get(
                            "observed_gate_confidence_png_base64"
                        ),
                        "guessed_occupancy_confidence_png_base64": prediction.get(
                            "guessed_occupancy_confidence_png_base64",
                            prediction.get("guessed_confidence_png_base64"),
                        ),
                        "depth_scale": float(prediction["depth_scale"]),
                        "scale_std_m_per_vggt": float(
                            prediction["scale_std_m_per_vggt"]
                        ),
                        "single_output_size": int(
                            prediction["single_output_size"]
                        ),
                        "single_latent_size": int(
                            prediction["single_latent_size"]
                        ),
                        "merged_enabled": bool(
                            prediction["merged_enabled"]
                        ),
                        "merged_extent_m": float(
                            prediction.get("merged_extent_m", 0.0)
                        ),
                        "merged_output_size": int(
                            prediction.get("merged_output_size", 0)
                        ),
                        "merged_latent_size": int(
                            prediction.get("merged_latent_size", 0)
                        ),
                        "visualization_mode": prediction.get(
                            "visualization_mode", "p2b"
                        ),
                        "checkpoint_epoch": int(
                            prediction["checkpoint_epoch"]
                        ),
                        "checkpoint_global_step": int(
                            prediction["checkpoint_global_step"]
                        ),
                    }
                latest = {"models": latest_models}
                with self.lock:
                    self.latest = latest
                    self.state.update(
                        {
                            "ready": True,
                            "status": (
                                "Three-model comparison synchronized"
                                if len(latest_models) == 3
                                else "Single-model comparison synchronized"
                            ),
                            "history_frame_count": self.history_count,
                            "inference_seconds": float(
                                response["inference_seconds"]
                            ),
                            "last_frame_seq": frame.frame_seq,
                            "segment_id": self.segment_id,
                            "shared_vggt_extraction": bool(
                                response.get("shared_vggt_extraction")
                            ),
                            "runtime_input_contract": response[
                                "runtime_input_contract"
                            ],
                        }
                    )
            except Exception as error:
                with self.lock:
                    self.state.update(
                        {
                            "ready": False,
                            "status": f"Model comparison error: {error}",
                        }
                    )
                try:
                    self._new_segment()
                except Exception:
                    self.segment_id = ""
                    self.stop_event.wait(1.0)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.server_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"runtime HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"runtime is unavailable: {error.reason}") from error
        if not isinstance(result, dict):
            raise RuntimeError("runtime response is not a JSON object")
        return result
