#!/usr/bin/env python3
"""RGB-only HTTP runtime for the P1B-NLL interactive visualizer.

This server runs the frozen VGGT aggregator once per submitted RGB window and
then executes the trained P1B single-BEV, gate and scale branches.  It never
accepts simulator geometry, BEV targets, camera parameters, depth or poses.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vggt_bev_method1.data.preprocess import RGBResizePad
from vggt_bev_method1.models import LiveVGGTOmegaAdapter, P1BSystem


SCHEMA = "p1b-three-region-evidential-v6"
LEGACY_SCHEMAS: set[str] = set()
FREE_COLOR = np.asarray((73, 206, 122), dtype=np.float32)
OCCUPIED_COLOR = np.asarray((242, 78, 78), dtype=np.float32)
UNKNOWN_COLOR = np.asarray((128, 128, 128), dtype=np.uint8)
GATE_GUESSED_COLOR = np.asarray((25, 54, 104), dtype=np.float32)
GATE_OBSERVED_COLOR = np.asarray((255, 197, 61), dtype=np.float32)
CONFIDENCE_LOW_COLOR = np.asarray((18, 23, 31), dtype=np.float32)
OBSERVED_CONFIDENCE_COLOR = np.asarray((52, 233, 138), dtype=np.float32)
GATE_CONTOUR_COLOR = np.asarray((255, 0, 0), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-source", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--max-history", type=int, default=10)
    return parser.parse_args()


def build_system(
    state: dict,
    *,
    backbone_source: Path,
    backbone_checkpoint: Path,
    device: torch.device,
) -> P1BSystem:
    if state.get("checkpoint_schema") not in {SCHEMA, *LEGACY_SCHEMAS}:
        raise ValueError(
            f"expected {SCHEMA} or a declared legacy alias, "
            f"got {state.get('checkpoint_schema')}"
        )
    values = state["config"]["model"]
    layers = tuple(int(value) for value in values["cached_layers"])
    # Keep unused frozen camera/depth heads and the disabled merged decoder on
    # CPU.  The executed computation remains precisely the Single-BEV graph.
    adapter = LiveVGGTOmegaAdapter(
        backbone_source,
        backbone_checkpoint,
        device=torch.device("cpu"),
        patch_size=int(values["patch_size"]),
        cached_layers=layers,
    )
    system = P1BSystem(
        adapter,
        probability_model=str(values["probability_model"]),
        cached_layers=layers,
        spatial_scales=tuple(float(value) for value in values["spatial_scales"]),
        vggt_token_dim=int(values["vggt_token_dim"]),
        hidden_dim=int(values["hidden_dim"]),
        heads=int(values["attention_heads"]),
        decoder_layers=int(values["decoder_layers"]),
        scale_decoder_layers=int(values["scale_decoder_layers"]),
        self_attention_mode=str(values["self_attention_mode"]),
        cross_attention_mode=str(values["cross_attention_mode"]),
        deformable_samples=int(values["deformable_samples"]),
        cross_query_chunk_size=int(values["cross_query_chunk_size"]),
        single_latent_bev_size=int(values["single_latent_bev_size"]),
        merged_latent_bev_size=int(values["merged_latent_bev_size"]),
        single_output_size=int(values["single_bev_output_size"]),
        merged_output_size=int(values["merged_bev_output_size"]),
        single_bev_extent_m=float(values["single_bev_extent_m"]),
        merged_bev_extent_m=float(values["merged_bev_extent_m"]),
        predict_scale_uncertainty=bool(values.get("predict_scale_uncertainty", True)),
    )
    system.head.load_state_dict(state["head"], strict=True)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    system.adapter.backbone.aggregator.to(device=device, dtype=dtype)
    system.adapter.backbone.camera_head.to(device=device, dtype=dtype)
    head = system.unwrapped_head()
    for module in (
        head.single_guessed_token_projector,
        head.single_routing_token_projector,
        head.scale_token_projector,
        head.single_bev_decoder,
        head.scale_decoder,
    ):
        module.to(device=device, dtype=dtype)
    return system.eval()


def _encode_png(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def render_semantic(branch: dict, threshold: float) -> np.ndarray:
    occupied = branch["occupancy_probability"][0].detach().float().cpu().numpy()
    support = branch["fov_support_probability"][0].detach().float().cpu().numpy()
    semantic = np.full(occupied.shape, 112, dtype=np.uint8)
    inside = support >= 0.5
    semantic[inside & (occupied < threshold)] = 255
    semantic[inside & (occupied >= threshold)] = 0
    return semantic


def decode_vggt_camera_extrinsics(
    system: P1BSystem,
    extraction: dict,
    images: torch.Tensor,
) -> torch.Tensor:
    """Decode VGGT world-to-camera poses without executing its depth head."""

    with torch.no_grad(), torch.autocast(
        device_type=images.device.type,
        dtype=torch.bfloat16,
        enabled=images.device.type == "cuda",
    ):
        pose_encoding = system.adapter.backbone.camera_head(
            extraction["_aggregated"],
            patch_token_start=extraction["_patch_start"],
        )
    from vggt_omega.utils.pose_enc import encoding_to_camera

    camera_from_world, _ = encoding_to_camera(
        pose_encoding.float(), images.shape[-2:]
    )
    return camera_from_world


def render_confidence(branch: dict, threshold: float) -> np.ndarray:
    occupied = branch["occupancy_probability"][0].detach().float().cpu().numpy()
    support = branch["fov_support_probability"][0].detach().float().cpu().numpy()
    confidence = branch["navigation_confidence"][0].detach().float().cpu().numpy()
    colors = np.where(
        (occupied >= threshold)[..., None], OCCUPIED_COLOR, FREE_COLOR
    )
    rendered = 28.0 * (1.0 - confidence[..., None]) + colors * confidence[..., None]
    rendered[support < 0.5] = UNKNOWN_COLOR
    return rendered.clip(0, 255).astype(np.uint8)


def render_observed_gate(branch: dict) -> np.ndarray:
    """Render P(observed-free): yellow=observed, blue=guessed, gray=outside FOV."""

    gate = (
        branch["observed_gate_probability"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .clip(0.0, 1.0)
    )
    support = branch["fov_support_probability"][0].detach().float().cpu().numpy()
    rendered = GATE_GUESSED_COLOR + gate[..., None] * (
        GATE_OBSERVED_COLOR - GATE_GUESSED_COLOR
    )
    rendered[support < 0.5] = UNKNOWN_COLOR
    return rendered.clip(0, 255).astype(np.uint8)


def _render_role_confidence(
    confidence: np.ndarray,
    support: np.ndarray,
    high_color: np.ndarray,
) -> np.ndarray:
    confidence = np.asarray(confidence, dtype=np.float32).clip(0.0, 1.0)
    rendered = CONFIDENCE_LOW_COLOR + confidence[..., None] * (
        high_color - CONFIDENCE_LOW_COLOR
    )
    rendered[np.asarray(support) < 0.5] = UNKNOWN_COLOR
    return rendered.clip(0, 255).astype(np.uint8)


def _shift_boolean(mask: np.ndarray, delta_y: int, delta_x: int) -> np.ndarray:
    """Shift a boolean raster without wrapping values across image edges."""

    height, width = mask.shape
    shifted = np.zeros_like(mask, dtype=bool)
    source_y = slice(max(0, -delta_y), min(height, height - delta_y))
    source_x = slice(max(0, -delta_x), min(width, width - delta_x))
    target_y = slice(max(0, delta_y), min(height, height + delta_y))
    target_x = slice(max(0, delta_x), min(width, width + delta_x))
    shifted[target_y, target_x] = mask[source_y, source_x]
    return shifted


def predicted_gate_contour(branch: dict) -> np.ndarray:
    """Return the one-pixel observed/guessed threshold boundary inside FOV."""

    support = branch["fov_support_probability"][0].detach().float().cpu().numpy()
    gate = branch["observed_gate_probability"][0].detach().float().cpu().numpy()
    valid = support >= 0.5
    observed = valid & (gate >= 0.5)
    guessed = valid & ~observed
    boundary = np.zeros_like(valid, dtype=bool)
    for delta_y, delta_x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        boundary |= observed & _shift_boolean(guessed, delta_y, delta_x)
    # No dilation: in a raster image one pixel is the minimum representation
    # of the zero-width continuous threshold curve.
    return boundary


def overlay_gate_contour(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    rendered = np.asarray(image).copy()
    if rendered.ndim == 2:
        rendered = np.repeat(rendered[..., None], 3, axis=2)
    if rendered.ndim != 3 or rendered.shape[2] != 3:
        raise ValueError("gate contour overlay expects grayscale or RGB image")
    rendered[contour] = GATE_CONTOUR_COLOR
    return rendered


def render_observed_gate_confidence(branch: dict) -> np.ndarray:
    """Confidence mass assigned specifically to the Observed-Free route."""

    support = branch["fov_support_probability"][0].detach().float().cpu().numpy()
    gate = branch["observed_gate_probability"][0].detach().float().cpu().numpy()
    routing_confidence = (
        branch["fused"]["routing_confidence"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    score = support * gate * routing_confidence
    return _render_role_confidence(score, support, OBSERVED_CONFIDENCE_COLOR)


def render_guessed_occupancy_confidence(branch: dict) -> np.ndarray:
    """Render the Guessed Expert's own free/occupied class confidence.

    The Gate is used only as a hard display-domain mask. Within that domain,
    green/red comes from the Guessed Expert's occupancy prediction and
    brightness is evidential confidence times classification confidence. This
    intentionally does not visualize ``1 - P(observed)`` as confidence.
    """

    support = branch["fov_support_probability"][0].detach().float().cpu().numpy()
    guessed_region = (
        branch["guessed_region_probability"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    occupied = (
        branch["guessed"]["occupancy_probability"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    evidence_confidence = (
        branch["guessed"]["evidence_confidence"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    classification_confidence = (
        branch["guessed"]["classification_confidence"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    confidence = (evidence_confidence * classification_confidence).clip(0.0, 1.0)
    colors = np.where(
        (occupied >= 0.5)[..., None], OCCUPIED_COLOR, FREE_COLOR
    )
    rendered = CONFIDENCE_LOW_COLOR + confidence[..., None] * (
        colors - CONFIDENCE_LOW_COLOR
    )
    display_domain = (support >= 0.5) & (guessed_region >= 0.5)
    rendered[~display_domain] = UNKNOWN_COLOR
    return rendered.clip(0, 255).astype(np.uint8)


class Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.state = torch.load(
            args.checkpoint.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        self.system = build_system(
            self.state,
            backbone_source=args.backbone_source.expanduser().resolve(),
            backbone_checkpoint=args.backbone_checkpoint.expanduser().resolve(),
            device=self.device,
        )
        data = self.state["config"]["data"]
        self.preprocess = RGBResizePad(
            int(data["image_height"]), int(data["image_width"])
        )
        self.max_history = int(args.max_history)
        if self.max_history < 1 or self.max_history > 10:
            raise ValueError("max history must be in [1, 10]")
        self.segment_id = ""
        self.frames: list[torch.Tensor] = []
        self.frame_seqs: list[int] = []
        self.lock = threading.Lock()

    def health(self) -> dict:
        model = self.state["config"]["model"]
        head = self.system.unwrapped_head()
        return {
            "ready": True,
            "model": "P1B-NLL",
            "single_extent_m": float(model["single_bev_extent_m"]),
            "single_output_size": int(head.single_bev_decoder.output_size),
            "single_latent_size": int(head.single_bev_decoder.guessed.latent_size),
            "merged_enabled": False,
            "max_history": self.max_history,
            "checkpoint_epoch": int(self.state.get("epoch", -1)),
            "checkpoint_global_step": int(self.state.get("global_step", -1)),
            "runtime_input_contract": {
                "rgb_window": True,
                "camera_height": False,
                "intrinsics": False,
                "extrinsics": False,
                "depth": False,
                "ground_truth": False,
                "vggt_geometry_heads_executed": True,
                "vggt_camera_head_executed": True,
                "vggt_depth_head_executed": False,
            },
            "shared_vggt_backbone": True,
            "runtime_output_contract": {
                "predicted_extrinsics": "VGGT world-to-camera 3x4",
                "predicted_extrinsic_units": "VGGT native translation units",
                "depth": False,
            },
        }

    def reset(self, segment_id: str) -> dict:
        with self.lock:
            self.segment_id = segment_id
            self.frames.clear()
            self.frame_seqs.clear()
        return {"accepted": True, "segment_id": segment_id}

    def predict(self, payload: dict) -> dict:
        segment_id = str(payload["segment_id"])
        image = Image.open(
            io.BytesIO(base64.b64decode(payload["image_png_base64"], validate=True))
        ).convert("RGB")
        tensor = self.preprocess(image)
        threshold = float(payload.get("threshold", 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        with self.lock:
            if segment_id != self.segment_id:
                self.segment_id = segment_id
                self.frames.clear()
                self.frame_seqs.clear()
            self.frames.append(tensor)
            self.frames = self.frames[-self.max_history :]
            self.frame_seqs.append(int(payload["frame_seq"]))
            self.frame_seqs = self.frame_seqs[-self.max_history :]
            images = torch.stack(self.frames)[None].to(self.device, non_blocking=True)
            started = time.monotonic()
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                extraction = self.system.extract(images)
                prediction = self.system.forward_head(
                    extraction, enabled_bev_branches=("single",), include_scale=True
                )
            branch = prediction["single_bev"]
            scale = prediction["scale"]
            camera_from_world_vggt = decode_vggt_camera_extrinsics(
                self.system, extraction, images
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.monotonic() - started
            gate_contour = predicted_gate_contour(branch)
            raw_semantic_png = _encode_png(render_semantic(branch, threshold))
            semantic_png = _encode_png(
                overlay_gate_contour(
                    render_semantic(branch, threshold), gate_contour
                )
            )
            observed_confidence_png = _encode_png(
                overlay_gate_contour(
                    render_observed_gate_confidence(branch), gate_contour
                )
            )
            guessed_occupancy_confidence_png = _encode_png(
                overlay_gate_contour(
                    render_guessed_occupancy_confidence(branch), gate_contour
                )
            )
            return {
                **self.health(),
                "history_frame_count": len(self.frames),
                "frame_seq": int(payload["frame_seq"]),
                "inference_seconds": elapsed,
                "model_single_png_base64": semantic_png,
                # Verifier/planner input without the red display-only Gate line.
                "model_single_semantic_png_base64": raw_semantic_png,
                "confidence_single_png_base64": _encode_png(
                    render_confidence(branch, threshold)
                ),
                "observed_gate_single_png_base64": _encode_png(
                    render_observed_gate(branch)
                ),
                "observed_gate_confidence_png_base64": observed_confidence_png,
                "guessed_occupancy_confidence_png_base64": (
                    guessed_occupancy_confidence_png
                ),
                # Backward-compatible alias for older comparison clients.
                "guessed_confidence_png_base64": guessed_occupancy_confidence_png,
                "depth_scale": float(scale["lambda_m_per_vggt"][0].detach().cpu()),
                "scale_std_m_per_vggt": float(
                    scale["scale_std_m_per_vggt"][0].detach().cpu()
                ),
                "shared_vggt_extraction": True,
                "vggt_pose_frame_seqs": list(self.frame_seqs),
                "vggt_predicted_camera_from_world": (
                    camera_from_world_vggt[0].detach().float().cpu().tolist()
                ),
            }


def make_handler(runtime: Runtime) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "P1BRuntime/1.0"

        def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if urlparse(self.path).path == "/health":
                self.send_json(runtime.health())
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 8 * 1024 * 1024:
                    raise ValueError("request too large")
                payload = json.loads(self.rfile.read(length))
                path = urlparse(self.path).path
                if path == "/reset":
                    result = runtime.reset(str(payload["segment_id"]))
                elif path == "/predict":
                    result = runtime.predict(payload)
                else:
                    self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                self.send_json(result)
            except Exception as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format_string: str, *args: object) -> None:
            if "/health" not in str(args[0]):
                super().log_message(format_string, *args)

    return Handler


def main() -> None:
    args = parse_args()
    runtime = Runtime(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    server.daemon_threads = True
    print(json.dumps(runtime.health(), indent=2), flush=True)
    print(f"P1B runtime ready: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
