from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from vggt_bev.geometry.lift import patch_centers, sample_image_at_pixels


class FrozenVGGTAdapter(nn.Module):
    """Read-only adapter around a sibling VGGT-Omega checkout.

    The adapter calls public submodules to expose final patch tokens and dense depth without
    editing the original repository. All backbone parameters and outputs are detached.
    """

    def __init__(
        self,
        source_root: str | Path,
        checkpoint: str | Path,
        *,
        device: torch.device,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        source_root = Path(source_root).expanduser().resolve()
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not (source_root / "vggt_omega").is_dir():
            raise FileNotFoundError(f"VGGT source package not found under {source_root}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint}")
        source_string = str(source_root)
        if source_string not in sys.path:
            sys.path.insert(0, source_string)

        from vggt_omega.models import VGGTOmega

        backbone = VGGTOmega(patch_size=patch_size).eval()
        state_dict = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        backbone.load_state_dict(state_dict, strict=True)
        del state_dict
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone = backbone.to(device)
        self.patch_size = patch_size

    def train(self, mode: bool = True) -> FrozenVGGTAdapter:
        super().train(False)
        self.backbone.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        *,
        include_geometry: bool = False,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError("images must have shape [B, N, 3, H, W]")
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("image dimensions must be divisible by the VGGT patch size")

        if images.device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            amp_context = nullcontext()

        with torch.no_grad():
            with amp_context:
                aggregated, patch_start = self.backbone.aggregator(images)
            final_tokens = aggregated[-1]
            if final_tokens is None:
                raise RuntimeError("VGGT aggregator did not return its final cached layer")
            patch_features = final_tokens[:, :, patch_start:].float()
            with torch.autocast(device_type=images.device.type, enabled=False):
                depth, confidence = self.backbone.dense_head(
                    aggregated,
                    images=images,
                    patch_token_start=patch_start,
                )
                pose_encoding = None
                estimated_camera_from_world = None
                estimated_intrinsics = None
                if include_geometry:
                    pose_encoding = self.backbone.camera_head(
                        aggregated,
                        patch_token_start=patch_start,
                    )
                    from vggt_omega.utils.pose_enc import encoding_to_camera

                    estimated_camera_from_world, estimated_intrinsics = (
                        encoding_to_camera(
                            pose_encoding,
                            images.shape[-2:],
                        )
                    )

            centers = patch_centers(
                height,
                width,
                self.patch_size,
                device=images.device,
                dtype=depth.dtype,
            )
            depth_samples = sample_image_at_pixels(
                depth.permute(0, 1, 4, 2, 3), centers
            )[..., 0]
            confidence_samples = sample_image_at_pixels(
                confidence[:, :, None], centers
            )[..., 0]

        if patch_features.shape[2] != centers.shape[0]:
            raise RuntimeError(
                f"patch count mismatch: tokens={patch_features.shape[2]} centers={centers.shape[0]}"
            )
        output = {
            "patch_features": patch_features.detach(),
            "depth": depth_samples.detach(),
            "confidence": confidence_samples.detach(),
            "patch_centers": centers,
        }
        if include_geometry:
            assert pose_encoding is not None
            assert estimated_camera_from_world is not None
            assert estimated_intrinsics is not None
            output.update(
                {
                    "dense_depth": depth.detach(),
                    "dense_confidence": confidence.detach(),
                    "pose_encoding": pose_encoding.detach(),
                    "estimated_camera_from_world": (
                        estimated_camera_from_world.detach()
                    ),
                    "estimated_intrinsics": estimated_intrinsics.detach(),
                }
            )
        return output
