#!/usr/bin/env python3
"""RGB-only live inference service for the private P1B visualizer."""

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
from vggt_bev_method1.models import LiveVGGTOmegaAdapter, Method1System


CLASS_VALUES = np.asarray((255, 0), dtype=np.uint8)
CLASS_COLORS = np.asarray(
    ((73, 206, 122), (242, 78, 78)),
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve live P1B RGB-only inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-source", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8875)
    parser.add_argument("--max-history", type=int, default=10)
    return parser.parse_args()


def build_system(
    state: dict,
    *,
    backbone_source: Path,
    backbone_checkpoint: Path,
    device: torch.device,
) -> Method1System:
    if state.get("checkpoint_schema") != "p1b-fixed-metric-fov-complete-evidential-v6":
        raise ValueError("the live visualizer requires a P1B v6 checkpoint")
    model = state["config"]["model"]
    layers = tuple(int(value) for value in model["cached_layers"])
    adapter = LiveVGGTOmegaAdapter(
        backbone_source,
        backbone_checkpoint,
        device=device,
        patch_size=int(model["patch_size"]),
        cached_layers=layers,
    )
    system = Method1System(
        adapter,
        cached_layers=layers,
        spatial_scales=tuple(float(value) for value in model["spatial_scales"]),
        vggt_token_dim=int(model["vggt_token_dim"]),
        hidden_dim=int(model["hidden_dim"]),
        heads=int(model["attention_heads"]),
        decoder_layers=int(model["decoder_layers"]),
        scale_decoder_layers=int(model["scale_decoder_layers"]),
        self_attention_mode=str(model["self_attention_mode"]),
        cross_attention_mode=str(model["cross_attention_mode"]),
        deformable_samples=int(model["deformable_samples"]),
        cross_query_chunk_size=int(model["cross_query_chunk_size"]),
        single_latent_bev_size=int(model["single_latent_bev_size"]),
        merged_latent_bev_size=int(model["merged_latent_bev_size"]),
        single_output_size=int(model["single_bev_output_size"]),
        merged_output_size=int(model["merged_bev_output_size"]),
        single_bev_extent_m=float(model["single_bev_extent_m"]),
        merged_bev_extent_m=float(model["merged_bev_extent_m"]),
        predict_scale_uncertainty=bool(model.get("predict_scale_uncertainty", True)),
    ).to(device)
    system.head.load_state_dict(state["head_state_dict"], strict=True)
    return system.eval()


def encode_png_base64(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def render_semantic(branch: dict[str, torch.Tensor], threshold: float) -> np.ndarray:
    occupied = branch["occupancy_probability"][0].detach().cpu().numpy() >= threshold
    support = branch["fov_support_probability"][0].detach().cpu().numpy() >= 0.5
    result = np.full(occupied.shape, 112, dtype=np.uint8)
    result[support & ~occupied] = CLASS_VALUES[0]
    result[support & occupied] = CLASS_VALUES[1]
    return result


def decode_vggt_camera_extrinsics(
    system: Method1System,
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


def render_confidence(branch: dict[str, torch.Tensor], threshold: float) -> np.ndarray:
    probability = branch["occupancy_probability"][0].detach().cpu().numpy()
    classes = (probability >= threshold).astype(np.int64)
    confidence = branch["evidence_confidence"][0].detach().cpu().numpy()[..., None]
    support = branch["fov_support_probability"][0].detach().cpu().numpy()[..., None]
    color = CLASS_COLORS[classes]
    rendered = 28.0 * (1.0 - confidence) + color * confidence
    rendered = 112.0 * (1.0 - support) + rendered * support
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
        output_contract = self.state.get("output_contract", {})
        trained_outputs = set(output_contract.get("trained_outputs", ()))
        disabled_outputs = set(
            output_contract.get("disabled_untrained_outputs", ())
        )
        if trained_outputs != {"single_bev", "scale"}:
            raise ValueError(
                "checkpoint output contract must train only single_bev + scale"
            )
        if "merged_bev" not in disabled_outputs:
            raise ValueError("checkpoint does not mark merged_bev as disabled")
        self.system = build_system(
            self.state,
            backbone_source=args.backbone_source.expanduser().resolve(),
            backbone_checkpoint=args.backbone_checkpoint.expanduser().resolve(),
            device=self.device,
        )
        data = self.state["config"]["data"]
        self.preprocess = RGBResizePad(
            int(data["image_height"]),
            int(data["image_width"]),
        )
        self.max_history = int(args.max_history)
        self.segment_id = ""
        self.frames: list[torch.Tensor] = []
        self.frame_seqs: list[int] = []
        self.lock = threading.Lock()

    def health(self) -> dict:
        model = self.state["config"]["model"]
        head = self.system.unwrapped_head()
        return {
            "ready": True,
            "model": "P1B",
            "single_extent_m": float(model["single_bev_extent_m"]),
            "single_output_size": int(head.single_bev_decoder.output_size),
            "single_latent_size": int(head.single_bev_decoder.latent_size),
            "merged_enabled": False,
            "trained_outputs": ["single_bev", "scale"],
            "disabled_outputs": ["merged_bev"],
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
        frame = Image.open(
            io.BytesIO(base64.b64decode(payload["image_png_base64"], validate=True))
        ).convert("RGB")
        tensor = self.preprocess(frame)
        threshold = float(payload.get("threshold", 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be inside [0,1]")
        with self.lock:
            if segment_id != self.segment_id:
                self.segment_id = segment_id
                self.frames.clear()
                self.frame_seqs.clear()
            self.frames.append(tensor)
            if len(self.frames) > self.max_history:
                self.frames = self.frames[-self.max_history :]
            self.frame_seqs.append(int(payload["frame_seq"]))
            self.frame_seqs = self.frame_seqs[-self.max_history :]
            images = torch.stack(self.frames)[None].to(
                self.device,
                non_blocking=True,
            )
            started = time.monotonic()
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                extraction = self.system.extract(images)
                prediction = self.system.forward_head(
                    extraction,
                    enabled_bev_branches=("single",),
                    include_scale=True,
                )
            single = prediction["single_bev"]
            scale = prediction["scale"]
            camera_from_world_vggt = decode_vggt_camera_extrinsics(
                self.system, extraction, images
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.monotonic() - started
            health = self.health()
            semantic_png = encode_png_base64(
                render_semantic(single, threshold)
            )
            return {
                **health,
                "history_frame_count": len(self.frames),
                "frame_seq": int(payload["frame_seq"]),
                "inference_seconds": elapsed,
                "model_single_png_base64": semantic_png,
                "model_single_semantic_png_base64": semantic_png,
                "confidence_single_png_base64": encode_png_base64(
                    render_confidence(single, threshold)
                ),
                "depth_scale": float(scale["lambda_m_per_vggt"][0].detach().cpu()),
                "lambda_m_per_vggt": float(
                    scale["lambda_m_per_vggt"][0].detach().cpu()
                ),
                "scale_std_m_per_vggt": float(
                    scale.get(
                        "scale_std_m_per_vggt",
                        torch.zeros(1, device=self.device),
                    )[0].detach().cpu()
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
