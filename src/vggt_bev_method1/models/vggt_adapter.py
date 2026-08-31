from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn


def _masked_moments(values: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid_float = valid.to(values.dtype)
    count = valid_float.sum(dim=tuple(range(1, values.ndim))).clamp_min(1.0)
    mean = (values * valid_float).sum(dim=tuple(range(1, values.ndim))) / count
    centered = values - mean.view(-1, *([1] * (values.ndim - 1)))
    variance = (centered.square() * valid_float).sum(
        dim=tuple(range(1, values.ndim))
    ) / count
    return mean, variance.sqrt()


def world_camera_centers(camera_from_world: torch.Tensor) -> torch.Tensor:
    """Convert PDF Equation 6 world-to-camera extrinsics to Equation 7 centers."""

    if camera_from_world.shape[-2:] != (3, 4):
        raise ValueError("camera_from_world must end in a 3x4 matrix")
    rotation = camera_from_world[..., :3, :3]
    translation = camera_from_world[..., :3, 3]
    return -(rotation.transpose(-1, -2) @ translation[..., None])[..., 0]


def _scene_radius(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_from_world: torch.Tensor,
    *,
    sample_stride: int = 4,
) -> torch.Tensor:
    """Equation 16 mean point radius in VGGT's arbitrary scene units."""

    _, _, height, width = depth.shape
    rows = torch.arange(
        0,
        height,
        sample_stride,
        device=depth.device,
        dtype=depth.dtype,
    )
    columns = torch.arange(
        0,
        width,
        sample_stride,
        device=depth.device,
        dtype=depth.dtype,
    )
    v, u = torch.meshgrid(rows, columns, indexing="ij")
    sampled_depth = depth[..., ::sample_stride, ::sample_stride]
    fx = intrinsics[..., 0, 0, None, None].clamp_min(1e-6)
    fy = intrinsics[..., 1, 1, None, None].clamp_min(1e-6)
    cx = intrinsics[..., 0, 2, None, None]
    cy = intrinsics[..., 1, 2, None, None]
    x = sampled_depth * (u.view(1, 1, *u.shape) - cx) / fx
    y = sampled_depth * (v.view(1, 1, *v.shape) - cy) / fy
    camera_points = torch.stack((x, y, sampled_depth), dim=-1)

    rotation = camera_from_world[..., :3, :3]
    translation = camera_from_world[..., :3, 3]
    world_points = (
        rotation.transpose(-1, -2)[..., None, None, :, :]
        @ (camera_points - translation[..., None, None, :])[..., None]
    )[..., 0]
    reference_rotation = rotation[:, :1]
    reference_translation = translation[:, :1]
    reference_points = (
        reference_rotation[..., None, None, :, :]
        @ world_points[..., None]
    )[..., 0] + reference_translation[..., None, None, :]
    point_radius = torch.linalg.vector_norm(reference_points, dim=-1)
    valid = torch.isfinite(point_radius) & (point_radius > 0)
    valid_float = valid.to(point_radius.dtype)
    count = valid_float.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return (torch.where(valid, point_radius, 0.0).sum(dim=(1, 2, 3)) / count).clamp_min(
        1e-6
    )


def stabilize_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    """PDF Equation 59: use one robust live-VGGT K for a fixed-camera sequence."""

    if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [B, N, 3, 3]")
    sequence_intrinsic = intrinsics.median(dim=1).values
    sequence_intrinsic[..., 0, 1] = 0.0
    sequence_intrinsic[..., 1, 0] = 0.0
    sequence_intrinsic[..., 2, 0] = 0.0
    sequence_intrinsic[..., 2, 1] = 0.0
    sequence_intrinsic[..., 2, 2] = 1.0
    return sequence_intrinsic[:, None].expand_as(intrinsics).clone()


def predicted_geometry_summary(
    depth: torch.Tensor,
    confidence: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_from_world: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> dict[str, torch.Tensor]:
    """Return scale-free cues expressed only in normalized VGGT units."""

    dense_depth = depth[..., 0] if depth.ndim == 5 else depth
    stable_intrinsics = stabilize_intrinsics(intrinsics)
    valid_depth = torch.isfinite(dense_depth) & (dense_depth > 0)
    scene_radius_vggt = _scene_radius(
        dense_depth,
        stable_intrinsics,
        camera_from_world,
    )
    normalized_depth = (
        dense_depth / scene_radius_vggt[:, None, None, None]
    )
    safe_log_depth = torch.where(
        valid_depth,
        normalized_depth.clamp_min(1e-6).log(),
        torch.zeros_like(dense_depth),
    )
    depth_mean, depth_std = _masked_moments(safe_log_depth, valid_depth)

    confidence_probability = ((confidence - 1.0) / confidence.clamp_min(1.0)).clamp(
        0.0, 1.0
    )
    valid_confidence = torch.isfinite(confidence_probability)
    confidence_mean, confidence_std = _masked_moments(
        confidence_probability, valid_confidence
    )

    latest_intrinsic = stable_intrinsics[:, -1]
    intrinsic_cue = torch.stack(
        (
            latest_intrinsic[:, 0, 0] / image_width,
            latest_intrinsic[:, 1, 1] / image_height,
            latest_intrinsic[:, 0, 2] / image_width,
            latest_intrinsic[:, 1, 2] / image_height,
        ),
        dim=-1,
    )

    camera_centers = world_camera_centers(camera_from_world)
    relative_translation = camera_centers - camera_centers[:, :1]
    translation_norm = torch.linalg.vector_norm(relative_translation, dim=-1)
    normalized_translation = translation_norm / scene_radius_vggt[:, None]
    translation_cue = torch.stack(
        (
            normalized_translation.mean(dim=1),
            normalized_translation.amax(dim=1),
            normalized_translation[:, -1],
        ),
        dim=-1,
    )

    first_rotation = camera_from_world[:, 0, :3, :3]
    latest_rotation = camera_from_world[:, -1, :3, :3]
    relative_rotation = latest_rotation @ first_rotation.transpose(-1, -2)
    rotation_cosine = (
        (relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0
    ).clamp(-1.0, 1.0)

    latest_depth = normalized_depth[:, -1]
    latest_valid = valid_depth[:, -1]
    latest_depth_mean, _ = _masked_moments(
        torch.where(
            latest_valid,
            latest_depth.clamp_min(1e-6).log(),
            torch.zeros_like(latest_depth),
        ),
        latest_valid,
    )
    frame_fraction = torch.full_like(
        latest_depth_mean, dense_depth.shape[1] / 34.0
    )
    cue = torch.cat(
        (
            depth_mean[:, None],
            depth_std[:, None],
            confidence_mean[:, None],
            confidence_std[:, None],
            intrinsic_cue,
            translation_cue,
            rotation_cosine[:, None],
            latest_depth_mean[:, None],
            frame_fraction[:, None],
        ),
        dim=-1,
    )
    return {
        "cue": cue,
        "scene_radius_vggt": scene_radius_vggt,
        "vggt_depth": dense_depth,
        "vggt_camera_from_world": camera_from_world,
        "stable_intrinsics": stable_intrinsics,
    }


class LiveVGGTOmegaAdapter(nn.Module):
    """Frozen live VGGT-Ω extractor used identically in training and runtime."""

    geometry_cue_dim = 14
    supports_native_token_dtype = True

    def __init__(
        self,
        source_root: str | Path,
        checkpoint_path: str | Path,
        *,
        device: torch.device,
        patch_size: int = 16,
        cached_layers: tuple[int, ...] = (4, 11, 17, 23),
    ) -> None:
        super().__init__()
        source_root = Path(source_root).expanduser().resolve()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not (source_root / "vggt_omega").is_dir():
            raise FileNotFoundError(f"VGGT-Ω source package is missing: {source_root}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"VGGT-Ω checkpoint is missing: {checkpoint_path}")
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        from vggt_omega.models import VGGTOmega

        model = VGGTOmega(patch_size=patch_size).eval()
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        del state
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.backbone = model.to(device)
        self.cached_layers = cached_layers
        self.patch_size = patch_size
        self.checkpoint_path = str(checkpoint_path)

    def train(self, mode: bool = True) -> LiveVGGTOmegaAdapter:
        super().train(False)
        self.backbone.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
    ) -> dict:
        shared = self.aggregate(images)
        return {**shared, **self.decode_geometry(shared)}

    def aggregate(
        self,
        images: torch.Tensor,
        *,
        preserve_token_dtype: bool = False,
    ) -> dict:
        """Run the frozen shared VGGT trunk once for all parallel heads."""

        if images.ndim != 5:
            raise ValueError("images must have shape [B, N, 3, H, W]")
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("image dimensions must be divisible by VGGT patch size")
        if images.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            amp = torch.autocast("cuda", dtype=dtype)
        else:
            amp = nullcontext()
        with torch.no_grad():
            with amp:
                aggregated, patch_start = self.backbone.aggregator(images)
            tokens = {}
            for layer in self.cached_layers:
                value = aggregated[layer]
                if value is None:
                    raise RuntimeError(f"VGGT did not cache requested layer {layer}")
                patch_tokens = value[:, :, patch_start:]
                tokens[layer] = (
                    patch_tokens
                    if preserve_token_dtype
                    else patch_tokens.float()
                ).detach()
            final_tokens = aggregated[-1]
            if final_tokens is None:
                raise RuntimeError("VGGT did not cache the final token layer")
            camera_register_tokens = final_tokens[:, :, :patch_start]
            if not preserve_token_dtype:
                camera_register_tokens = camera_register_tokens.float()
            camera_register_tokens = camera_register_tokens.detach()
        return {
            "tokens": tokens,
            "camera_register_tokens": camera_register_tokens,
            "patch_grid": (height // self.patch_size, width // self.patch_size),
            # Private frozen shared state is decoded only by the parallel
            # camera/depth branch. Method1Head never receives or consumes its
            # outputs.
            "_aggregated": aggregated,
            "_patch_start": patch_start,
            "_images": images,
        }

    def decode_geometry(self, shared: dict) -> dict:
        """Decode K/depth/poses independently of the P1B BEV head."""

        images = shared["_images"]
        aggregated = shared["_aggregated"]
        patch_start = shared["_patch_start"]
        height, width = images.shape[-2:]
        with torch.no_grad():
            with torch.autocast(
                device_type=images.device.type,
                enabled=False,
            ):
                depth, confidence = self.backbone.dense_head(
                    aggregated,
                    images=images,
                    patch_token_start=patch_start,
                )
                pose_encoding = self.backbone.camera_head(
                    aggregated,
                    patch_token_start=patch_start,
                )
                from vggt_omega.utils.pose_enc import encoding_to_camera

                camera_from_world, intrinsics = encoding_to_camera(
                    pose_encoding,
                    images.shape[-2:],
                )
                geometry = predicted_geometry_summary(
                    depth,
                    confidence,
                    intrinsics,
                    camera_from_world,
                    image_height=height,
                    image_width=width,
                )
        return {
            "geometry_cue": geometry["cue"].detach(),
            "scene_radius_vggt": geometry["scene_radius_vggt"].detach(),
            "estimated_depth_vggt": geometry["vggt_depth"].detach(),
            "estimated_depth_confidence": confidence.detach(),
            "estimated_intrinsics": geometry["stable_intrinsics"].detach(),
            "estimated_camera_from_world_vggt": geometry[
                "vggt_camera_from_world"
            ].detach(),
            "geometry_source": (
                "live VGGT-Omega depth/intrinsics/extrinsics in native scale"
            ),
            "coordinate_mode": "vggt_native_units",
            "normalization_unit": "scene_radius_vggt",
        }

    def decode_scale_teacher(self, shared: dict) -> dict:
        """Decode only depth/confidence for training-only metric scale labels.

        The implicit WTBD Merged pipeline deliberately skips CameraHead:
        intrinsics and extrinsics are neither needed for robust aligned-depth
        scale targets nor permitted as Merged inputs.
        """

        images = shared["_images"]
        aggregated = shared["_aggregated"]
        patch_start = shared["_patch_start"]
        with torch.no_grad():
            with torch.autocast(
                device_type=images.device.type,
                enabled=False,
            ):
                depth, confidence = self.backbone.dense_head(
                    aggregated,
                    images=images,
                    patch_token_start=patch_start,
                )
        dense_depth = depth[..., 0] if depth.ndim == 5 else depth
        return {
            "estimated_depth_vggt": dense_depth.detach(),
            "estimated_depth_confidence": confidence.detach(),
            "teacher_source": "frozen VGGT depth head only",
        }
