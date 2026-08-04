#!/usr/bin/env python3
"""RGB-only runtime for the historical P1B masked single+merged checkpoint."""

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


SCHEMA = "p1b-fixed-metric-dual-v3"
CLASS_VALUES = np.asarray((112, 255, 0), dtype=np.uint8)
CLASS_COLORS = np.asarray(
    (
        (128, 128, 128),
        (73, 206, 122),
        (242, 78, 78),
    ),
    dtype=np.float32,
)
LOW_SCORE_COLOR = np.asarray((18, 23, 31), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-source", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8898)
    parser.add_argument("--max-history", type=int, default=10)
    return parser.parse_args()


def _migrate_shared_projector(state_dict: dict[str, torch.Tensor]) -> dict:
    """Recreate the old shared BEV/scale projector in the current topology."""

    migrated = dict(state_dict)
    prefix = "token_projector."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            suffix = key.removeprefix(prefix)
            migrated.setdefault(f"scale_token_projector.{suffix}", value)
    return migrated


def build_system(
    state: dict,
    *,
    backbone_source: Path,
    backbone_checkpoint: Path,
    device: torch.device,
) -> Method1System:
    if state.get("checkpoint_schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}, got {state.get('checkpoint_schema')}")
    values = state["config"]["model"]
    layers = tuple(int(value) for value in values["cached_layers"])
    adapter = LiveVGGTOmegaAdapter(
        backbone_source,
        backbone_checkpoint,
        device=torch.device("cpu"),
        patch_size=int(values["patch_size"]),
        cached_layers=layers,
    )
    system = Method1System(
        adapter,
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
        predict_scale_uncertainty=bool(
            values.get("predict_scale_uncertainty", True)
        ),
    )
    compatible_state = _migrate_shared_projector(state["head_state_dict"])
    system.head.load_state_dict(compatible_state, strict=True)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    system.adapter.backbone.aggregator.to(device=device, dtype=dtype)
    head = system.unwrapped_head()
    for module in (
        head.token_projector,
        head.scale_token_projector,
        head.single_bev_decoder,
        head.merged_bev_decoder,
        head.scale_decoder,
    ):
        module.to(device)
    return system.eval()


def _encode_png(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _categorical_probabilities(branch: dict) -> np.ndarray:
    return (
        branch["raw_output"][0]
        .detach()
        .float()
        .softmax(dim=0)
        .cpu()
        .numpy()
    )


def render_semantic(branch: dict) -> np.ndarray:
    labels = _categorical_probabilities(branch).argmax(axis=0)
    return CLASS_VALUES[labels]


def render_categorical_score(branch: dict) -> np.ndarray:
    probabilities = _categorical_probabilities(branch)
    labels = probabilities.argmax(axis=0)
    score = probabilities.max(axis=0).clip(0.0, 1.0)
    colors = CLASS_COLORS[labels]
    rendered = LOW_SCORE_COLOR + score[..., None] * (
        colors - LOW_SCORE_COLOR
    )
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
        self.lock = threading.Lock()

    def health(self) -> dict:
        model = self.state["config"]["model"]
        head = self.system.unwrapped_head()
        return {
            "ready": True,
            "model": "P1B masked-only",
            "visualization_mode": "legacy_masked_dual",
            "checkpoint_schema": self.state["checkpoint_schema"],
            "single_extent_m": float(model["single_bev_extent_m"]),
            "single_output_size": int(head.single_bev_decoder.output_size),
            "single_latent_size": int(head.single_bev_decoder.latent_size),
            "merged_enabled": True,
            "merged_extent_m": float(model["merged_bev_extent_m"]),
            "merged_output_size": int(head.merged_bev_decoder.output_size),
            "merged_latent_size": int(head.merged_bev_decoder.latent_size),
            "max_history": self.max_history,
            "checkpoint_epoch": int(self.state.get("epoch", -1)),
            "checkpoint_global_step": int(self.state.get("global_step", -1)),
            "output_class_order": ["unknown", "free", "occupied"],
            "confidence_contract": "categorical softmax score; not evidential",
            "runtime_input_contract": {
                "rgb_window": True,
                "camera_height": False,
                "intrinsics": False,
                "extrinsics": False,
                "depth": False,
                "ground_truth": False,
                "vggt_geometry_heads_executed": False,
            },
            "shared_vggt_backbone": True,
        }

    def reset(self, segment_id: str) -> dict:
        with self.lock:
            self.segment_id = segment_id
            self.frames.clear()
        return {"accepted": True, "segment_id": segment_id}

    def predict(self, payload: dict) -> dict:
        segment_id = str(payload["segment_id"])
        image = Image.open(
            io.BytesIO(base64.b64decode(payload["image_png_base64"], validate=True))
        ).convert("RGB")
        tensor = self.preprocess(image)
        with self.lock:
            if segment_id != self.segment_id:
                self.segment_id = segment_id
                self.frames.clear()
            self.frames.append(tensor)
            self.frames = self.frames[-self.max_history :]
            images = torch.stack(self.frames)[None].to(self.device, non_blocking=True)
            started = time.monotonic()
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                extraction = self.system.extract(images)
                prediction = self.system.forward_head(
                    extraction,
                    enabled_bev_branches=("single", "merged"),
                    include_scale=True,
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            single = prediction["single_bev"]
            merged = prediction["merged_bev"]
            scale = prediction["scale"]
            return {
                **self.health(),
                "history_frame_count": len(self.frames),
                "frame_seq": int(payload["frame_seq"]),
                "inference_seconds": time.monotonic() - started,
                "model_single_png_base64": _encode_png(render_semantic(single)),
                "model_merged_png_base64": _encode_png(render_semantic(merged)),
                "merged_score_png_base64": _encode_png(
                    render_categorical_score(merged)
                ),
                "depth_scale": float(scale["lambda_m_per_vggt"][0].detach().cpu()),
                "scale_std_m_per_vggt": float(
                    scale["scale_std_m_per_vggt"][0].detach().cpu()
                ),
                "shared_vggt_extraction": True,
            }


def make_handler(runtime: Runtime) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "P1BLegacyMaskedRuntime/1.0"

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
    print(f"P1B legacy masked runtime ready: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
