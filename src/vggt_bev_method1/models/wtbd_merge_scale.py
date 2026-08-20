from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .attention import DirectDecoderBlock
from .method1 import MetricScaleTokenHead, MultiScaleTokenProjector
from .p1b_probability import (
    ProbabilityModel,
    compose_semantic,
    decode_binary_prediction,
    fuse_pixel_routing,
)


def relative_native_geometry_features(
    camera_from_world: torch.Tensor,
    scene_radius_vggt: torch.Tensor,
    *,
    canonical_extent_vggt: float,
) -> torch.Tensor:
    """Encode frozen VGGT camera geometry without introducing metric scale.

    The returned per-frame feature uses the exact camera-from-world matrices
    predicted in the current VGGT window.  Translations remain VGGT-native;
    they are represented both relative to the canonical BEV extent and to the
    VGGT scene radius.  No GT pose, camera height, Scale Token or Single BEV is
    consumed.
    """

    if camera_from_world.ndim != 4 or camera_from_world.shape[-2:] != (3, 4):
        raise ValueError("camera_from_world must have shape [B,N,3,4]")
    if scene_radius_vggt.shape != camera_from_world.shape[:1]:
        raise ValueError("scene_radius_vggt must have shape [B]")
    if canonical_extent_vggt <= 0.0:
        raise ValueError("canonical extent must be positive")

    batch, frames = camera_from_world.shape[:2]
    bottom = camera_from_world.new_zeros(batch, frames, 1, 4)
    bottom[..., 0, 3] = 1.0
    homogeneous = torch.cat((camera_from_world, bottom), dim=-2)
    world_from_camera = torch.linalg.inv(homogeneous)
    latest_from_frame = homogeneous[:, -1:, :, :] @ world_from_camera
    rotation = latest_from_frame[..., :3, :3].reshape(batch, frames, 9)
    translation = latest_from_frame[..., :3, 3]
    normalized_extent = translation / float(canonical_extent_vggt)
    normalized_radius = translation / scene_radius_vggt[:, None, None].clamp_min(
        1e-6
    )
    if frames == 1:
        age = translation.new_zeros(batch, 1, 1)
    else:
        age_values = torch.arange(
            frames - 1,
            -1,
            -1,
            device=translation.device,
            dtype=translation.dtype,
        ) / float(frames - 1)
        age = age_values.view(1, frames, 1).expand(batch, -1, -1)
    latest = translation.new_zeros(batch, frames, 1)
    latest[:, -1] = 1.0
    return torch.cat(
        (rotation, normalized_extent, normalized_radius, age, latest),
        dim=-1,
    )


def _vggt_unit_query_coordinates(
    size: int,
    extent_vggt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if size <= 0 or extent_vggt <= 0.0:
        raise ValueError("query size and VGGT-unit extent must be positive")
    cell = extent_vggt / size
    pixel = torch.arange(size, dtype=torch.float32)
    x = -extent_vggt / 2.0 + (pixel + 0.5) * cell
    z = extent_vggt / 2.0 - (pixel + 0.5) * cell
    z_grid, x_grid = torch.meshgrid(z, x, indexing="ij")
    coordinates = torch.stack((x_grid, z_grid), dim=-1).reshape(-1, 2)
    reference = torch.stack(
        (x_grid / (extent_vggt / 2.0), -z_grid / (extent_vggt / 2.0)),
        dim=-1,
    ).reshape(-1, 2)
    return coordinates, reference


class DenseVGGTUnitQueryDecoder(nn.Module):
    """Dense BEV decoder whose query coordinates are VGGT runtime units."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        layers: int,
        feature_levels: int,
        output_channels: int,
        latent_size: int,
        output_size: int,
        extent_vggt: float,
        self_attention_mode: str,
        cross_attention_mode: str,
        deformable_samples: int,
        cross_query_chunk_size: int,
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size)
        self.output_size = int(output_size)
        self.extent_vggt = float(extent_vggt)
        coordinates, reference = _vggt_unit_query_coordinates(
            self.latent_size, self.extent_vggt
        )
        self.register_buffer(
            "coordinates_vggt", coordinates, persistent=True
        )
        self.register_buffer("reference_grid", reference, persistent=True)
        self.query_content = nn.Parameter(
            torch.empty(self.latent_size * self.latent_size, hidden_dim)
        )
        self.coordinate_position = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                DirectDecoderBlock(
                    hidden_dim,
                    heads,
                    self_attention_mode=self_attention_mode,
                    cross_attention_mode=cross_attention_mode,
                    feature_levels=feature_levels,
                    deformable_samples=deformable_samples,
                    cross_query_chunk_size=cross_query_chunk_size,
                )
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_channels, 1),
        )
        nn.init.normal_(self.query_content, std=0.02)

    def forward(self, pyramid: list[torch.Tensor]) -> torch.Tensor:
        batch = pyramid[0].shape[0]
        position = self.coordinate_position(
            self.coordinates_vggt.to(dtype=self.query_content.dtype)
            / (self.extent_vggt / 2.0)
        )
        query = (self.query_content + position).unsqueeze(0).expand(batch, -1, -1)
        if self.blocks[0].cross_attention_mode == "deformable":
            context: torch.Tensor | list[torch.Tensor] = pyramid
        else:
            context = torch.cat(
                [
                    level.permute(0, 1, 3, 4, 2).reshape(
                        batch, -1, level.shape[2]
                    )
                    for level in pyramid
                ],
                dim=1,
            )
        for block in self.blocks:
            query = block(query, context, self.reference_grid)
        latent = self.output_norm(query).transpose(1, 2).reshape(
            batch, -1, self.latent_size, self.latent_size
        )
        raw = self.output_projection(latent)
        if self.latent_size != self.output_size:
            raw = F.interpolate(
                raw,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return raw


class VGGTUnitPixelRoutedBEVDecoder(nn.Module):
    """Merged observed/guessed evidential output in VGGT-native units."""

    def __init__(self, *, probability_model: ProbabilityModel, **arguments) -> None:
        super().__init__()
        self.probability_model = probability_model
        channels = 2 if probability_model == "evidential" else 1
        self.guessed = DenseVGGTUnitQueryDecoder(
            output_channels=channels, **arguments
        )
        self.routing = DenseVGGTUnitQueryDecoder(
            output_channels=2, **arguments
        )
        self.output_size = self.guessed.output_size
        self.extent_vggt = self.guessed.extent_vggt

    def forward(
        self,
        guessed_pyramid: list[torch.Tensor],
        routing_pyramid: list[torch.Tensor],
        *,
        assemble_runtime_outputs: bool = True,
    ) -> dict:
        guessed = decode_binary_prediction(
            self.guessed(guessed_pyramid),
            self.probability_model,
            include_diagnostics=assemble_runtime_outputs,
        )
        routing_raw = self.routing(routing_pyramid).float()
        observed_logit = routing_raw[:, 0]
        support_logit = routing_raw[:, 1]
        support_probability = torch.sigmoid(support_logit)
        output = {
            "guessed": guessed,
            "observed_gate_logit": observed_logit,
            "fov_support_logit": support_logit,
            "fov_support_probability": support_probability,
        }
        if not assemble_runtime_outputs:
            return output
        observed_probability = torch.sigmoid(observed_logit)
        guessed_region = 1.0 - observed_probability
        routing_probability = torch.stack(
            (
                observed_probability,
                guessed_region * (1.0 - guessed["occupancy_probability"].float()),
                guessed_region * guessed["occupancy_probability"].float(),
            ),
            dim=1,
        )
        fused = fuse_pixel_routing(
            routing_probability,
            guessed,
            support_probability,
            self.probability_model,
        )
        output.update(
            {
                "routing_probability": routing_probability,
                "routing_class": routing_probability.argmax(dim=1),
                "observed_gate_probability": observed_probability,
                "guessed_region_probability": guessed_region,
                "fused": fused,
                "occupancy_probability": fused["occupancy_probability"],
                "navigation_confidence": fused["navigation_confidence"],
                "fov_complete_semantic": compose_semantic(
                    fused, support_probability
                ),
            }
        )
        return output


class WTBDMergeScaleHead(nn.Module):
    """No-Single Merged head plus an independent metric Scale Token."""

    pipeline_id = "WTBD-MERGE-SCALE-NLL"

    def __init__(
        self,
        *,
        probability_model: ProbabilityModel = "evidential",
        cached_layers: tuple[int, ...] = (4, 11, 17, 23),
        spatial_scales: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
        vggt_token_dim: int = 2048,
        hidden_dim: int = 64,
        heads: int = 8,
        decoder_layers: int = 2,
        scale_decoder_layers: int = 2,
        self_attention_mode: str = "linear",
        cross_attention_mode: str = "deformable",
        deformable_samples: int = 4,
        cross_query_chunk_size: int = 4096,
        merged_latent_bev_size: int = 80,
        merged_output_size: int = 800,
        merged_extent_vggt: float = 6.5,
        predict_scale_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        if probability_model != "evidential":
            raise ValueError("WTBD Merge-Scale uses evidential NLL")
        projector_args = (
            cached_layers,
            vggt_token_dim,
            hidden_dim,
            spatial_scales,
        )
        self.probability_model = probability_model
        self.merged_guessed_token_projector = MultiScaleTokenProjector(
            *projector_args
        )
        self.merged_routing_token_projector = MultiScaleTokenProjector(
            *projector_args
        )
        self.scale_token_projector = MultiScaleTokenProjector(*projector_args)
        self.native_geometry_embedding = nn.Sequential(
            nn.LayerNorm(17),
            nn.Linear(17, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        common = dict(
            hidden_dim=hidden_dim,
            heads=heads,
            layers=decoder_layers,
            feature_levels=len(cached_layers),
            latent_size=merged_latent_bev_size,
            output_size=merged_output_size,
            extent_vggt=merged_extent_vggt,
            self_attention_mode=self_attention_mode,
            cross_attention_mode=cross_attention_mode,
            deformable_samples=deformable_samples,
            cross_query_chunk_size=cross_query_chunk_size,
        )
        self.merged_bev_decoder = VGGTUnitPixelRoutedBEVDecoder(
            probability_model=probability_model,
            **common,
        )
        self.scale_decoder = MetricScaleTokenHead(
            hidden_dim,
            heads,
            layers=scale_decoder_layers,
            predict_uncertainty=predict_scale_uncertainty,
        )

    def forward(
        self,
        extraction: dict,
        geometry: dict,
        *,
        include_merged: bool = True,
        include_scale: bool = True,
        assemble_runtime_outputs: bool = True,
    ) -> dict:
        output: dict = {}
        if include_merged:
            geometry_features = relative_native_geometry_features(
                geometry["estimated_camera_from_world_vggt"],
                geometry["scene_radius_vggt"],
                canonical_extent_vggt=self.merged_bev_decoder.extent_vggt,
            )
            frame_embedding = self.native_geometry_embedding(geometry_features)
            guessed = self.merged_guessed_token_projector(
                extraction["tokens"],
                extraction["patch_grid"],
                frame_embedding=frame_embedding,
            )
            routing = self.merged_routing_token_projector(
                extraction["tokens"],
                extraction["patch_grid"],
                frame_embedding=frame_embedding,
            )
            output["merged_bev"] = self.merged_bev_decoder(
                guessed,
                routing,
                assemble_runtime_outputs=assemble_runtime_outputs,
            )
        if include_scale:
            # Deliberately parallel: predicted scale never enters the Merged
            # projector, query coordinates, decoder or target construction.
            scale_pyramid = self.scale_token_projector(
                extraction["tokens"], extraction["patch_grid"]
            )
            output["scale"] = self.scale_decoder(scale_pyramid)
        return output


class WTBDMergeScaleSystem(nn.Module):
    def __init__(self, adapter: nn.Module, **head_arguments) -> None:
        super().__init__()
        self.adapter = adapter
        self.head = WTBDMergeScaleHead(**head_arguments)

    def unwrapped_head(self) -> WTBDMergeScaleHead:
        head = getattr(self.head, "module", self.head)
        if not isinstance(head, WTBDMergeScaleHead):
            raise TypeError("unexpected WTBD head type")
        return head

    def extract(self, images: torch.Tensor) -> dict:
        return self.adapter.aggregate(images)

    def decode_teacher_geometry(self, extraction: dict) -> dict:
        return self.adapter.decode_geometry(extraction)

    def forward_head(self, extraction: dict, geometry: dict, **arguments) -> dict:
        prediction = self.head(extraction, geometry, **arguments)
        head = self.unwrapped_head()
        return {
            **prediction,
            "pipeline_id": head.pipeline_id,
            "runtime_inputs": ("rgb_window",),
            "internal_frozen_geometry": (
                "VGGT depth/intrinsics/camera_from_world from the same window"
            ),
            "coordinate_mode": "vggt_native_units",
            "merged_extent_vggt": head.merged_bev_decoder.extent_vggt,
            "merged_output_size": head.merged_bev_decoder.output_size,
            "scale_unit": "meter_per_vggt_runtime_unit",
            "scale_is_merged_input": False,
            "single_bev_present": False,
            "orientation": "latest ego centered; forward is image-up",
        }

    def forward(self, images: torch.Tensor) -> dict:
        extraction = self.extract(images)
        geometry = self.decode_teacher_geometry(extraction)
        return self.forward_head(extraction, geometry)
