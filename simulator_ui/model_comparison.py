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


def decode_semantic_png_base64(encoded: str) -> np.ndarray:
    """Decode a runtime semantic PNG without changing its class values."""

    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        semantic = np.asarray(image.convert("L"), dtype=np.uint8)
    if semantic.ndim != 2 or semantic.shape[0] != semantic.shape[1]:
        raise ValueError("runtime semantic output must be a square grayscale PNG")
    return semantic.copy()


def render_gt_fov_support(complete: np.ndarray) -> np.ndarray:
    """Gray outside support and green inside, independent of occupancy."""

    support = np.asarray(complete) != 112
    rendered = np.full((*support.shape, 3), (112, 112, 112), dtype=np.uint8)
    rendered[support] = (73, 206, 122)
    return rendered


def render_gt_observed_gate(
    complete: np.ndarray,
    visible: np.ndarray,
) -> np.ndarray:
    """Render the exact Policy-A Gate target: observed-free vs guessed."""

    complete = np.asarray(complete)
    visible = np.asarray(visible)
    support = complete != 112
    observed_free = support & (visible != 112) & (visible != 0)
    rendered = np.full((*support.shape, 3), (112, 112, 112), dtype=np.uint8)
    rendered[support] = (25, 54, 104)
    rendered[observed_free] = (255, 197, 61)
    return rendered


@dataclass(frozen=True)
class ComparisonFrame:
    frame_seq: int
    motion_step: int
    camera_rgb: np.ndarray
    gt_complete_by_model: dict[str, np.ndarray]
    gt_observed_by_model: dict[str, np.ndarray]
    gt_guessed_by_model: dict[str, np.ndarray]
    gt_merged_observed_by_model: dict[str, np.ndarray]
    gt_merged_complete_by_model: dict[str, np.ndarray]
    gt_merged_visible_by_model: dict[str, np.ndarray]
    extrinsic: dict[str, Any]


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
        self.merged_extents_m = dict(self.model_extents_m)
        self.merged_output_sizes = {
            key: int(bev_size) for key in self.model_extents_m
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
        self.visualization_mode = "p1b"
        self.latest: Optional[dict[str, Any]] = None
        # Raw synchronized rasters are intentionally kept outside ``latest``;
        # the latter must remain JSON-serializable for the browser endpoint.
        self.latest_planning: Optional[dict[str, Any]] = None
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
        self.visualization_mode = str(health.get("visualization_mode", "p1b"))
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
                    "merged_extent_m": health.get(
                        "merged_extent_m", health["single_extent_m"]
                    ),
                    "merged_output_size": health.get(
                        "merged_output_size", self.bev_size
                    ),
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
            self.merged_extents_m[model_key] = float(
                health_models[model_key].get(
                    "merged_extent_m", health.get("merged_extent_m", actual)
                )
            )
            self.merged_output_sizes[model_key] = int(
                health_models[model_key].get(
                    "merged_output_size",
                    health.get("merged_output_size", self.bev_size),
                )
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
                    "visualization_mode": self.visualization_mode,
                    "merged_extents_m": dict(self.merged_extents_m),
                    "merged_output_sizes": dict(self.merged_output_sizes),
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

    def planning_snapshot(
        self,
        model_key: str,
        expected_frame_seq: Optional[int] = None,
    ) -> dict[str, Any]:
        """Return one atomic prediction-only runtime tuple for local planning.

        GT renderings remain inside ``presentation_images`` for the browser,
        but no GT array or simulator extrinsic is exposed as a numeric planner
        input.
        """

        with self.lock:
            if self.latest_planning is None:
                raise RuntimeError("no synchronized model frame is ready")
            frame_seq = int(self.latest_planning["frame_seq"])
            if expected_frame_seq is not None and frame_seq != expected_frame_seq:
                raise RuntimeError(
                    f"requested frame {expected_frame_seq} is stale; latest is {frame_seq}"
                )
            models = self.latest_planning["models"]
            if model_key not in models:
                raise KeyError(f"model {model_key!r} is unavailable")
            source = models[model_key]
            return {
                "frame_seq": frame_seq,
                "model_key": model_key,
                "predicted_semantic": source["predicted_semantic"].copy(),
                "predicted_extent_m": float(source["predicted_extent_m"]),
                "lambda_m_per_vggt": float(source["lambda_m_per_vggt"]),
                "scale_std_m_per_vggt": float(
                    source["scale_std_m_per_vggt"]
                ),
                "vggt_pose_frame_seqs": list(
                    source["vggt_pose_frame_seqs"]
                ),
                "vggt_predicted_camera_from_world": json.loads(
                    json.dumps(source["vggt_predicted_camera_from_world"])
                ),
                "presentation_images": json.loads(
                    json.dumps(source["presentation_images"])
                ),
            }

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
                latest_planning_models = {}
                for model_key, extent_m in self.model_extents_m.items():
                    prediction = response_models[model_key]
                    actual_extent = float(prediction["single_extent_m"])
                    if abs(actual_extent - extent_m) > 1e-6:
                        raise RuntimeError(
                            f"{model_key} prediction extent changed to "
                            f"{actual_extent:g} m"
                        )
                    if self.visualization_mode == "merged_routing_geometry":
                        merged_extent = float(prediction["merged_extent_m"])
                        expected_merged_extent = self.merged_extents_m[model_key]
                        if abs(merged_extent - expected_merged_extent) > 1e-6:
                            raise RuntimeError(
                                f"routing runtime merged extent changed from "
                                f"{expected_merged_extent:g} to {merged_extent:g} m"
                            )
                        complete = frame.gt_merged_complete_by_model[model_key]
                        visible = frame.gt_merged_visible_by_model[model_key]
                        latest_models[model_key] = {
                            "extent_m": merged_extent,
                            "gt_fov_support_png_base64": encode_png_base64(
                                render_gt_fov_support(complete)
                            ),
                            "gt_observed_gate_png_base64": encode_png_base64(
                                render_gt_observed_gate(complete, visible)
                            ),
                            "gt_merged_complete_png_base64": encode_png_base64(
                                complete
                            ),
                            "gt_merged_visible_png_base64": encode_png_base64(
                                visible
                            ),
                            "predicted_fov_support_probability_png_base64": prediction[
                                "predicted_fov_support_probability_png_base64"
                            ],
                            "predicted_fov_support_binary_png_base64": prediction[
                                "predicted_fov_support_binary_png_base64"
                            ],
                            "predicted_observed_gate_probability_png_base64": prediction[
                                "predicted_observed_gate_probability_png_base64"
                            ],
                            "predicted_observed_gate_binary_png_base64": prediction[
                                "predicted_observed_gate_binary_png_base64"
                            ],
                            "merged_output_size": int(prediction["merged_output_size"]),
                            "merged_latent_size": int(prediction["merged_latent_size"]),
                            "checkpoint_epoch": int(prediction["checkpoint_epoch"]),
                            "checkpoint_global_step": int(
                                prediction["checkpoint_global_step"]
                            ),
                            "visualization_mode": self.visualization_mode,
                        }
                        continue
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
                        "planning_semantic_png_base64": prediction.get(
                            "model_single_semantic_png_base64",
                            prediction["model_single_png_base64"],
                        ),
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
                            "visualization_mode", "p1b"
                        ),
                        "checkpoint_epoch": int(
                            prediction["checkpoint_epoch"]
                        ),
                        "checkpoint_global_step": int(
                            prediction["checkpoint_global_step"]
                        ),
                    }
                    predicted_semantic = decode_semantic_png_base64(
                        prediction.get(
                            "model_single_semantic_png_base64",
                            prediction["model_single_png_base64"],
                        )
                    )
                    lambda_m_per_vggt = float(
                        prediction.get(
                            "lambda_m_per_vggt", prediction["depth_scale"]
                        )
                    )
                    pose_frame_seqs = prediction.get(
                        "vggt_pose_frame_seqs",
                        response.get("vggt_pose_frame_seqs", []),
                    )
                    predicted_camera_from_world = prediction.get(
                        "vggt_predicted_camera_from_world",
                        response.get(
                            "vggt_predicted_camera_from_world", []
                        ),
                    )
                    if len(pose_frame_seqs) != len(
                        predicted_camera_from_world
                    ):
                        raise RuntimeError(
                            "VGGT pose IDs and predicted extrinsics differ in length"
                        )
                    latest_models[model_key]["vggt_pose_available"] = bool(
                        pose_frame_seqs
                    )
                    latest_models[model_key]["vggt_pose_frame_count"] = len(
                        pose_frame_seqs
                    )
                    latest_planning_models[model_key] = {
                        "predicted_semantic": predicted_semantic,
                        "predicted_extent_m": actual_extent,
                        "lambda_m_per_vggt": lambda_m_per_vggt,
                        "scale_std_m_per_vggt": float(
                            prediction["scale_std_m_per_vggt"]
                        ),
                        "vggt_pose_frame_seqs": [
                            int(value) for value in pose_frame_seqs
                        ],
                        "vggt_predicted_camera_from_world": (
                            json.loads(
                                json.dumps(predicted_camera_from_world)
                            )
                        ),
                        "presentation_images": json.loads(
                            json.dumps(latest_models[model_key])
                        ),
                    }
                latest = {"models": latest_models}
                with self.lock:
                    self.latest = latest
                    self.latest_planning = {
                        "frame_seq": int(frame.frame_seq),
                        "models": latest_planning_models,
                    }
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
