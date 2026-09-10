#!/usr/bin/env python3
"""Baseline Single BEV runtime with VGGT-extrinsic hard fusion.

Every RGB frame is decoded independently by the accepted baseline Single head.
A second VGGT pass estimates one coherent camera-pose chain for the current
window. Single predictions are warped into the latest camera frame and written
oldest-to-newest, so the latest valid FOV sample owns overlap pixels while
older frames remain visible outside the latest FOV.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


ROOT = Path(os.environ.get('BEV_MODEL_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "src"))

from vggt_bev_method1.data.preprocess import RGBResizePad
from vggt_bev_method1.metric_calibration import (
    calibrate_bev_scale_from_camera_height,
)
from vggt_bev_method1.models import LiveVGGTOmegaAdapter, P1BSystem
from shm_support import POLICY as SUPPORT_POLICY, trusted_latest_mask


SUPPORTED_SCHEMAS = {
    "p2b-three-region-evidential-v5",
    "p1b-three-region-evidential-v6",
}
FREE_COLOR = np.asarray((73, 206, 122), dtype=np.float32)
OCCUPIED_COLOR = np.asarray((242, 78, 78), dtype=np.float32)
UNKNOWN_COLOR = np.asarray((112, 112, 112), dtype=np.uint8)


@dataclass
class CachedSingle:
    frame_seq: int
    image: torch.Tensor
    occupancy: torch.Tensor
    confidence: torch.Tensor
    support: torch.Tensor
    observed_gate: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-source", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18897)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--gate-overwrite-inset-pixels", type=int, default=2)
    parser.add_argument(
        "--single-navigation-only",
        action="store_true",
        help="serve calibrated current-frame Single BEV without hard-fusion work",
    )
    return parser.parse_args()


def build_system(
    state: dict,
    *,
    backbone_source: Path,
    backbone_checkpoint: Path,
    device: torch.device,
) -> P1BSystem:
    schema = str(state.get("checkpoint_schema"))
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported baseline checkpoint schema: {schema}")
    values = state["config"]["model"]
    layers = tuple(int(value) for value in values["cached_layers"])
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
    system.adapter.backbone.dense_head.to(device=device, dtype=dtype)
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


def decode_vggt_geometry(
    system: P1BSystem,
    extraction: dict,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode predicted geometry solely for metric calibration diagnostics."""

    with torch.no_grad(), torch.autocast(
        device_type=images.device.type,
        dtype=torch.bfloat16,
        enabled=images.device.type == "cuda",
    ):
        depth, confidence = system.adapter.backbone.dense_head(
            extraction["_aggregated"],
            images=images,
            patch_token_start=extraction["_patch_start"],
        )
        pose_encoding = system.adapter.backbone.camera_head(
            extraction["_aggregated"],
            patch_token_start=extraction["_patch_start"],
        )
    from vggt_omega.utils.pose_enc import encoding_to_camera

    camera_from_world, intrinsics = encoding_to_camera(
        pose_encoding.float(), images.shape[-2:]
    )
    return depth.float(), confidence.float(), intrinsics.float(), camera_from_world


def _encode_png(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _encode_probability_png(tensor: torch.Tensor) -> str:
    """Encode an exact [0, 1] planner field as lossless uint16 PNG."""

    array = tensor.detach().float().cpu().numpy().clip(0.0, 1.0)
    encoded = np.rint(array * 65535.0).astype(np.uint16)
    buffer = io.BytesIO()
    Image.fromarray(encoded, mode="I;16").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def render_semantic(
    occupancy: torch.Tensor,
    support: torch.Tensor,
    threshold: float,
) -> np.ndarray:
    occupied = occupancy.detach().float().cpu().numpy()
    valid = support.detach().float().cpu().numpy() >= 0.5
    semantic = np.full(occupied.shape, 112, dtype=np.uint8)
    semantic[valid & (occupied < threshold)] = 255
    semantic[valid & (occupied >= threshold)] = 0
    return semantic


def render_confidence(
    occupancy: torch.Tensor,
    confidence: torch.Tensor,
    support: torch.Tensor,
    threshold: float,
) -> np.ndarray:
    occupied = occupancy.detach().float().cpu().numpy()
    certainty = confidence.detach().float().cpu().numpy().clip(0.0, 1.0)
    valid = support.detach().float().cpu().numpy() >= 0.5
    colors = np.where(
        (occupied >= threshold)[..., None], OCCUPIED_COLOR, FREE_COLOR
    )
    rendered = 20.0 + certainty[..., None] * (colors - 20.0)
    rendered[~valid] = UNKNOWN_COLOR
    return rendered.clip(0, 255).astype(np.uint8)


def render_support(support: torch.Tensor) -> np.ndarray:
    valid = support.detach().float().cpu().numpy() >= 0.5
    rendered = np.full((*valid.shape, 3), UNKNOWN_COLOR, dtype=np.uint8)
    rendered[valid] = FREE_COLOR.astype(np.uint8)
    return rendered


def render_gate(gate: torch.Tensor, support: torch.Tensor) -> np.ndarray:
    """Render the binary Observed/Guessed routing decision inside FOV."""

    observed = gate.detach().float().cpu().numpy() >= 0.5
    valid = support.detach().float().cpu().numpy() >= 0.5
    rendered = np.full((*valid.shape, 3), UNKNOWN_COLOR, dtype=np.uint8)
    rendered[valid] = (25, 54, 104)
    rendered[valid & observed] = (255, 197, 61)
    return rendered


def render_owner(owner: torch.Tensor, frame_count: int) -> np.ndarray:
    values = owner.detach().cpu().numpy()
    rendered = np.full((*values.shape, 3), UNKNOWN_COLOR, dtype=np.uint8)
    if frame_count < 1:
        return rendered
    valid = values >= 0
    age = values.astype(np.float32) / max(frame_count - 1, 1)
    old = np.asarray((72, 61, 139), dtype=np.float32)
    new = np.asarray((255, 197, 61), dtype=np.float32)
    colors = old + age[..., None] * (new - old)
    rendered[valid] = colors[valid].astype(np.uint8)
    return rendered


def camera_to_latest_planar_transform(
    source_camera_from_world: torch.Tensor,
    latest_camera_from_world: torch.Tensor,
    metric_scale_m_per_vggt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return p_latest = A @ p_source + b for planar [x, z] points."""

    if source_camera_from_world.shape != (3, 4):
        raise ValueError("source camera matrix must be 3x4")
    if latest_camera_from_world.shape != (3, 4):
        raise ValueError("latest camera matrix must be 3x4")
    source_rotation = source_camera_from_world[:, :3]
    source_translation = source_camera_from_world[:, 3]
    latest_rotation = latest_camera_from_world[:, :3]
    latest_translation = latest_camera_from_world[:, 3]
    relative_rotation = latest_rotation @ source_rotation.transpose(0, 1)
    relative_translation = (
        latest_translation - relative_rotation @ source_translation
    ) * metric_scale_m_per_vggt
    planar_rotation = relative_rotation.index_select(
        0, torch.tensor((0, 2), device=relative_rotation.device)
    ).index_select(
        1, torch.tensor((0, 2), device=relative_rotation.device)
    )
    planar_translation = relative_translation.index_select(
        0, torch.tensor((0, 2), device=relative_translation.device)
    )
    if torch.linalg.det(planar_rotation).abs() < 1e-5:
        raise ValueError("VGGT planar camera transform is singular")
    return planar_rotation, planar_translation


def warp_latest_ego(
    raster: torch.Tensor,
    source_camera_from_world: torch.Tensor,
    latest_camera_from_world: torch.Tensor,
    metric_scale_m_per_vggt: torch.Tensor,
    extent_m: float,
) -> torch.Tensor:
    """Nearest-neighbor warp from a source ego grid into the latest ego grid."""

    if raster.ndim != 2 or raster.shape[0] != raster.shape[1]:
        raise ValueError("BEV raster must be square")
    size = int(raster.shape[0])
    device = raster.device
    dtype = torch.float32
    planar_rotation, planar_translation = camera_to_latest_planar_transform(
        source_camera_from_world.float(),
        latest_camera_from_world.float(),
        metric_scale_m_per_vggt.float(),
    )
    inverse_rotation = torch.linalg.inv(planar_rotation)
    coordinates = (
        torch.arange(size, device=device, dtype=dtype) + 0.5
    ) * (extent_m / size) - extent_m / 2.0
    z_coordinates = coordinates.flip(0)
    z_latest, x_latest = torch.meshgrid(
        z_coordinates, coordinates, indexing="ij"
    )
    latest = torch.stack((x_latest, z_latest), dim=-1)
    source = (latest - planar_translation) @ inverse_rotation.transpose(0, 1)
    grid = torch.stack(
        (2.0 * source[..., 0] / extent_m, -2.0 * source[..., 1] / extent_m),
        dim=-1,
    )
    return F.grid_sample(
        raster.float()[None, None],
        grid[None],
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0]


def erode_support(support: torch.Tensor, inset_pixels: int) -> torch.Tensor:
    """Remove a thin FOV boundary band without wrapping at image edges."""

    if inset_pixels < 0:
        raise ValueError("gate overwrite inset must be non-negative")
    if inset_pixels == 0:
        return (support >= 0.5).float()
    invalid = (support < 0.5).float()[None, None]
    invalid = F.pad(
        invalid,
        (inset_pixels,) * 4,
        mode="constant",
        value=1.0,
    )
    invalid_nearby = F.max_pool2d(
        invalid,
        kernel_size=2 * inset_pixels + 1,
        stride=1,
    )[0, 0]
    return (invalid_nearby < 0.5).float()


def hard_fuse_latest_wins(
    singles: list[CachedSingle],
    camera_from_world: torch.Tensor,
    metric_scale_m_per_vggt: torch.Tensor,
    extent_m: float,
    gate_overwrite_inset_pixels: int = 2,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """Latest trusted interior overwrites; unsupported boundary stays unknown."""

    if len(singles) != int(camera_from_world.shape[0]):
        raise ValueError("Single cache and VGGT pose chain differ in length")
    size = int(singles[-1].occupancy.shape[0])
    device = camera_from_world.device
    fused_occupancy = torch.full((size, size), 0.5, device=device)
    fused_confidence = torch.zeros((size, size), device=device)
    fused_gate = torch.zeros((size, size), device=device)
    latest_only_gate = torch.zeros((size, size), device=device)
    gate_seam_overlap = torch.zeros((size, size), dtype=torch.bool, device=device)
    fused_support = torch.zeros((size, size), dtype=torch.bool, device=device)
    owner = torch.full((size, size), -1, dtype=torch.int16, device=device)
    latest_pose = camera_from_world[-1]
    for index, (single, pose) in enumerate(zip(singles, camera_from_world)):
        warped_support = warp_latest_ego(
            single.support.to(device), pose, latest_pose, metric_scale_m_per_vggt, extent_m
        ) >= 0.5
        warped_gate_interior = warp_latest_ego(
            erode_support(
                single.support.to(device), gate_overwrite_inset_pixels
            ),
            pose,
            latest_pose,
            metric_scale_m_per_vggt,
            extent_m,
        ) >= 0.5
        warped_occupancy = warp_latest_ego(
            single.occupancy.to(device), pose, latest_pose, metric_scale_m_per_vggt, extent_m
        )
        warped_confidence = warp_latest_ego(
            single.confidence.to(device), pose, latest_pose, metric_scale_m_per_vggt, extent_m
        )
        warped_gate = warp_latest_ego(
            single.observed_gate.to(device),
            pose,
            latest_pose,
            metric_scale_m_per_vggt,
            extent_m,
        )
        trusted = trusted_latest_mask(warped_support, warped_gate_interior)
        fused_occupancy[trusted] = warped_occupancy[trusted]
        fused_confidence[trusted] = warped_confidence[trusted]
        latest_only_gate[warped_support] = warped_gate[warped_support]
        # New cells and FOV interior retain strict latest-frame ownership. In
        # the thin overlap seam, observed evidence from either frame wins. A
        # later guessed edge can therefore no longer cut a blue crack through
        # an already observed region, while the support domain never expands.
        new_or_interior = warped_support & (~fused_support | warped_gate_interior)
        seam_overlap = warped_support & fused_support & ~warped_gate_interior
        gate_seam_overlap |= seam_overlap
        fused_gate[new_or_interior] = warped_gate[new_or_interior]
        fused_gate[seam_overlap] = torch.maximum(
            fused_gate[seam_overlap], warped_gate[seam_overlap]
        )
        # Do not dilate a thresholded FOV seam into a hard obstacle around the
        # robot. This is a validity mask, NOT relabeling obstacles as free.
        # Old trusted evidence survives a new untrusted border observation.
        fused_support[trusted] = True
        owner[trusted] = index
    return (
        fused_occupancy,
        fused_confidence,
        fused_support.float(),
        fused_gate,
        owner,
        {
            "latest_only_gate": latest_only_gate,
            "gate_seam_overlap": gate_seam_overlap,
        },
    )


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
        model = self.state["config"]["model"]
        self.extent_m = float(model["single_bev_extent_m"])
        self.output_size = int(model["single_bev_output_size"])
        self.max_history = int(args.max_history)
        if not 1 <= self.max_history <= 10:
            raise ValueError("max history must be in [1, 10]")
        self.gate_overwrite_inset_pixels = int(args.gate_overwrite_inset_pixels)
        if not 0 <= self.gate_overwrite_inset_pixels <= 8:
            raise ValueError("gate overwrite inset must be in [0, 8]")
        self.single_navigation_only = bool(args.single_navigation_only)
        self.segment_id = ""
        self.singles: list[CachedSingle] = []
        self.lock = threading.Lock()

    def health(self) -> dict:
        return {
            "ready": True,
            "model": (
                "accepted baseline calibrated Single"
                if self.single_navigation_only
                else "accepted baseline Single + VGGT hard fusion"
            ),
            "visualization_mode": (
                "baseline_single_scale_pose_navigation"
                if self.single_navigation_only
                else "baseline_single_vggt_hard_fusion"
            ),
            "single_extent_m": self.extent_m,
            "single_output_size": self.output_size,
            "merged_extent_m": self.extent_m,
            "merged_output_size": self.output_size,
            "max_history": self.max_history,
            "checkpoint_schema": self.state.get("checkpoint_schema"),
            "checkpoint_epoch": int(self.state.get("epoch", -1)),
            "checkpoint_global_step": int(self.state.get("global_step", -1)),
            "runtime_input_contract": {
                "rgb_frames": True,
                "simulator_pose_or_extrinsic": False,
                "ground_truth": False,
                "physical_camera_height": True,
                "simulator_metric_scale": False,
            },
            "fusion_contract": (
                {
                    "enabled": False,
                    "planner_input": "current-frame accepted Single BEV only",
                    "history_use": "predicted metric calibration only",
                    "predicted_extrinsic_target_tracking": False,
                }
                if self.single_navigation_only
                else {
                    "enabled": True,
                    "pose_source": "VGGT camera head world-to-camera extrinsics",
                    "translation_scale": "predicted camera-height metric scale",
                    "valid_domain": "predicted per-frame FOV interior; untrusted border stays unknown",
                    "planner_support_policy": SUPPORT_POLICY,
                    "planner_support_inset_pixels": self.gate_overwrite_inset_pixels,
                    "overlap": "latest trusted interior overwrites occupancy and confidence",
                    "non_overlap": "retain the newest frame that covers each cell",
                    "resampling": "nearest neighbor",
                    "gate_overlap": (
                        "latest wins except the outer "
                        f"{self.gate_overwrite_inset_pixels}px FOV seam, where "
                        "observed evidence from either overlapping frame wins"
                    ),
                }
            ),
        }

    def reset(self, segment_id: str) -> dict:
        with self.lock:
            self.segment_id = segment_id
            self.singles.clear()
        return {"accepted": True, "segment_id": segment_id}

    def _decode_poses(self, extraction: dict, image_size: tuple[int, int]) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            encoding = self.system.adapter.backbone.camera_head(
                extraction["_aggregated"],
                patch_token_start=extraction["_patch_start"],
            )
        from vggt_omega.utils.pose_enc import encoding_to_camera

        camera_from_world, _ = encoding_to_camera(encoding.float(), image_size)
        return camera_from_world[0].float()

    def predict(self, payload: dict) -> dict:
        segment_id = str(payload["segment_id"])
        frame_seq = int(payload["frame_seq"])
        threshold = float(payload.get("threshold", 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        physical_camera_height_m = float(payload["physical_camera_height_m"])
        if not 0.05 <= physical_camera_height_m <= 3.0:
            raise ValueError("physical_camera_height_m must be in [0.05, 3.0]")
        image = Image.open(
            io.BytesIO(base64.b64decode(payload["image_png_base64"], validate=True))
        ).convert("RGB")
        tensor = self.preprocess(image)
        with self.lock:
            if segment_id != self.segment_id:
                self.segment_id = segment_id
                self.singles.clear()
            if self.singles and frame_seq <= self.singles[-1].frame_seq:
                raise ValueError("frame_seq must increase within one segment")
            started = time.monotonic()
            single_images = tensor[None, None].to(self.device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                single_extraction = self.system.extract(single_images)
                single_prediction = self.system.forward_head(
                    single_extraction,
                    enabled_bev_branches=("single",),
                    include_scale=False,
                )["single_bev"]
            cached = CachedSingle(
                frame_seq=frame_seq,
                image=tensor.cpu(),
                occupancy=single_prediction["occupancy_probability"][0].detach().float().cpu(),
                confidence=single_prediction["navigation_confidence"][0].detach().float().cpu(),
                support=single_prediction["fov_support_probability"][0].detach().float().cpu(),
                observed_gate=single_prediction["observed_gate_probability"][0]
                .detach()
                .float()
                .cpu(),
            )
            self.singles.append(cached)
            self.singles = self.singles[-self.max_history :]

            window = torch.stack([entry.image for entry in self.singles])[None].to(
                self.device, non_blocking=True
            )
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                window_extraction = self.system.extract(window)
                scale = self.system.forward_head(
                    window_extraction,
                    enabled_bev_branches=(),
                    include_scale=True,
                )["scale"]
            (
                depth_vggt,
                confidence_vggt,
                intrinsics_vggt,
                camera_from_world,
            ) = decode_vggt_geometry(
                self.system, window_extraction, window
            )
            model_scale_token = scale["lambda_m_per_vggt"].float()
            calibration = calibrate_bev_scale_from_camera_height(
                depth_vggt=depth_vggt,
                confidence_vggt=confidence_vggt,
                intrinsics=intrinsics_vggt,
                camera_from_world_vggt=camera_from_world,
                physical_camera_height_m=torch.full_like(
                    model_scale_token, physical_camera_height_m
                ),
                model_scale_token_bev_per_vggt=model_scale_token,
            )
            camera_metric_scale = calibration.camera_metric_scale_m_per_vggt[0]
            bev_metric_scale = calibration.bev_metric_scale_m_per_bev[0]
            calibrated_extent_m = float(
                (self.extent_m * bev_metric_scale).detach().cpu()
            )
            if self.single_navigation_only:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                elapsed = time.monotonic() - started
                latest = self.singles[-1]
                return {
                    **self.health(),
                    "segment_id": self.segment_id,
                    "frame_seq": frame_seq,
                    "history_frame_count": len(self.singles),
                    "history_frame_seqs": [entry.frame_seq for entry in self.singles],
                    "inference_seconds": elapsed,
                    "model_single_semantic_png_base64": _encode_png(
                        render_semantic(latest.occupancy, latest.support, threshold)
                    ),
                    "planner_occupancy_probability_u16_png_base64": (
                        _encode_probability_png(latest.occupancy)
                    ),
                    "planner_navigation_confidence_u16_png_base64": (
                        _encode_probability_png(latest.confidence)
                    ),
                    "single_metric_extent_m": calibrated_extent_m,
                    "model_scale_token_bev_per_vggt": float(
                        model_scale_token[0].detach().cpu()
                    ),
                    "camera_metric_scale_m_per_vggt": float(
                        camera_metric_scale.detach().cpu()
                    ),
                    "bev_metric_scale_m_per_bev": float(
                        bev_metric_scale.detach().cpu()
                    ),
                    "physical_camera_height_m": physical_camera_height_m,
                    "vggt_camera_height": float(
                        calibration.camera_height_vggt[0].detach().cpu()
                    ),
                    "ground_inlier_fraction": float(
                        calibration.ground_inlier_fraction[0].detach().cpu()
                    ),
                    "ground_fallback_used": bool(
                        calibration.ground_fallback_used[0].detach().cpu()
                    ),
                    "vggt_pose_frame_seqs": [
                        entry.frame_seq for entry in self.singles
                    ],
                    "vggt_predicted_camera_from_world": (
                        camera_from_world.detach().float().cpu().tolist()
                    ),
                    "predicted_extrinsic_used_for_target_tracking": False,
                    "runtime_mode": "single BEV + predicted metric calibration",
                }
            (
                fused_occupancy,
                fused_confidence,
                fused_support,
                fused_gate,
                owner,
                fusion_diagnostics,
            ) = hard_fuse_latest_wins(
                    self.singles,
                    camera_from_world[0],  # remove batch dimension, retain ordered T poses
                    camera_metric_scale,
                    calibrated_extent_m,
                    self.gate_overwrite_inset_pixels,
                )
            latest_only_gate = fusion_diagnostics["latest_only_gate"]
            seam_overlap = fusion_diagnostics["gate_seam_overlap"]
            recovered_observed = (
                seam_overlap
                & (fused_gate >= 0.5)
                & (latest_only_gate < 0.5)
            )
            regressed_to_guessed = (
                seam_overlap
                & (fused_gate < 0.5)
                & (latest_only_gate >= 0.5)
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.monotonic() - started
            latest = self.singles[-1]
            return {
                **self.health(),
                "segment_id": self.segment_id,
                "frame_seq": frame_seq,
                "history_frame_count": len(self.singles),
                "history_frame_seqs": [entry.frame_seq for entry in self.singles],
                "inference_seconds": elapsed,
                "planner_bev_source": "shm_latest_wins",
                "shm_occupancy_probability_u16_png_base64": _encode_probability_png(fused_occupancy),
                "shm_navigation_confidence_u16_png_base64": _encode_probability_png(fused_confidence),
                "shm_metric_extent_m": calibrated_extent_m,
                "shm_historical_owner_pixels": int(((owner >= 0) & (owner < len(self.singles)-1)).sum().item()),
                "shm_total_supported_pixels": int((fused_support >= .5).sum().item()),
                "metric_scale_m_per_vggt": float(
                    camera_metric_scale.detach().cpu()
                ),
                "camera_metric_scale_m_per_vggt": float(
                    camera_metric_scale.detach().cpu()
                ),
                "bev_metric_scale_m_per_bev": float(
                    bev_metric_scale.detach().cpu()
                ),
                "model_scale_token_bev_per_vggt": float(
                    model_scale_token[0].detach().cpu()
                ),
                "metric_scale_std_m_per_vggt": float(
                    scale["scale_std_m_per_vggt"][0].detach().float().cpu()
                ),
                "physical_camera_height_m": physical_camera_height_m,
                "vggt_camera_height": float(
                    calibration.camera_height_vggt[0].detach().cpu()
                ),
                "ground_inlier_fraction": float(
                    calibration.ground_inlier_fraction[0].detach().cpu()
                ),
                "ground_fallback_used": bool(
                    calibration.ground_fallback_used[0].detach().cpu()
                ),
                "gate_seam_overlap_pixels": int(seam_overlap.sum().detach().cpu()),
                "gate_seam_observed_recovered_pixels": int(
                    recovered_observed.sum().detach().cpu()
                ),
                "gate_seam_observed_regressed_pixels": int(
                    regressed_to_guessed.sum().detach().cpu()
                ),
                "vggt_pose_frame_seqs": [entry.frame_seq for entry in self.singles],
                "latest_single_semantic_png_base64": _encode_png(
                    render_semantic(latest.occupancy, latest.support, threshold)
                ),
                # Stable planner-facing aliases.  These expose the actual
                # Single-head fields; no hard-fused history or visualization
                # colour inversion enters the benchmark planner input.
                "model_single_semantic_png_base64": _encode_png(
                    render_semantic(latest.occupancy, latest.support, threshold)
                ),
                "planner_occupancy_probability_u16_png_base64": (
                    _encode_probability_png(latest.occupancy)
                ),
                "planner_navigation_confidence_u16_png_base64": (
                    _encode_probability_png(latest.confidence)
                ),
                "single_metric_extent_m": calibrated_extent_m,
                "target_tracking_uses_vggt_extrinsic": False,
                "latest_single_confidence_png_base64": _encode_png(
                    render_confidence(
                        latest.occupancy, latest.confidence, latest.support, threshold
                    )
                ),
                "hard_merged_semantic_png_base64": _encode_png(
                    render_semantic(fused_occupancy, fused_support, threshold)
                ),
                "hard_merged_confidence_png_base64": _encode_png(
                    render_confidence(
                        fused_occupancy, fused_confidence, fused_support, threshold
                    )
                ),
                "hard_merged_support_png_base64": _encode_png(
                    render_support(fused_support)
                ),
                "hard_merged_gate_png_base64": _encode_png(
                    render_gate(fused_gate, fused_support)
                ),
                "hard_merged_owner_png_base64": _encode_png(
                    render_owner(owner, len(self.singles))
                ),
                "vggt_predicted_camera_from_world": camera_from_world.detach().cpu().tolist(),
            }


def make_handler(runtime: Runtime) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "P1BBaselineHardFusion/1.0"

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
    print(f"Baseline hard-fusion runtime: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
