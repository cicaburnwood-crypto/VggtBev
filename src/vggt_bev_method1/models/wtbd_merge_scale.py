from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .attention import DirectDecoderBlock, MultiheadAttention
from .method1 import MetricScaleTokenHead, MultiScaleTokenProjector
from .p1b_probability import (
    ProbabilityModel,
    compose_semantic,
    decode_binary_prediction,
    fuse_pixel_routing,
)


class ImplicitGeometryBlock(nn.Module):
    """Head-local cross-frame reasoning block, following VGGT CameraHead."""

    def __init__(self, hidden_dim: int, heads: int, expansion: int = 4) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiheadAttention(hidden_dim, heads, mode="exact")
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        tokens = tokens + self.attention(normalized, normalized)
        return tokens + self.ffn(self.ffn_norm(tokens))


class ImplicitGeometryContextTrunk(nn.Module):
    """Infer frame geometry latents without predicting or consuming poses.

    VGGT's released CameraHead mixes final camera/register tokens from all
    frames through four head-local transformer blocks before regressing pose.
    This trunk keeps the same useful reasoning pattern, but ends at a latent
    per-frame context.  It has no pose output, pose loss, extrinsic input or
    geometry post-processing path.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        heads: int,
        layers: int,
        maximum_history: int,
        maximum_prefix_tokens: int = 17,
    ) -> None:
        super().__init__()
        if layers <= 0 or maximum_history <= 0 or maximum_prefix_tokens <= 0:
            raise ValueError("implicit geometry trunk dimensions must be positive")
        self.maximum_history = int(maximum_history)
        self.maximum_prefix_tokens = int(maximum_prefix_tokens)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        self.frame_age_embedding = nn.Parameter(
            torch.empty(maximum_history, hidden_dim)
        )
        self.prefix_type_embedding = nn.Parameter(
            torch.empty(maximum_prefix_tokens, hidden_dim)
        )
        self.latest_reference_embedding = nn.Parameter(torch.empty(hidden_dim))
        self.blocks = nn.ModuleList(
            [ImplicitGeometryBlock(hidden_dim, heads) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.normal_(self.frame_age_embedding, std=0.02)
        nn.init.normal_(self.prefix_type_embedding, std=0.02)
        nn.init.normal_(self.latest_reference_embedding, std=0.02)

    def forward(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        if prefix_tokens.ndim != 4:
            raise ValueError("camera/register tokens must have shape [B,N,P,C]")
        batch, frames, prefix_count, _ = prefix_tokens.shape
        if frames > self.maximum_history:
            raise ValueError("frame count exceeds implicit trunk maximum_history")
        if prefix_count > self.maximum_prefix_tokens:
            raise ValueError("prefix token count exceeds configured maximum")
        tokens = self.input_projection(prefix_tokens)
        frame_age = torch.arange(
            frames - 1,
            -1,
            -1,
            device=tokens.device,
        )
        tokens = (
            tokens
            + self.frame_age_embedding[frame_age][None, :, None, :]
            + self.prefix_type_embedding[:prefix_count][None, None, :, :]
        )
        tokens = tokens.clone()
        tokens[:, -1] = tokens[:, -1] + self.latest_reference_embedding
        tokens = tokens.reshape(batch, frames * prefix_count, -1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.output_norm(
            tokens.reshape(batch, frames, prefix_count, -1)
        )
        camera_context = tokens[:, :, 0]
        register_context = tokens.mean(dim=2)
        return self.output_projection(
            torch.cat((camera_context, register_context), dim=-1)
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
    """Implicit multi-view Merged head plus an independent Scale Token."""

    pipeline_id = "WTBD-IMPLICIT-MERGE-SCALE-NLL"

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
        cross_attention_mode: str = "linear",
        deformable_samples: int = 4,
        cross_query_chunk_size: int = 4096,
        merged_latent_bev_size: int = 80,
        merged_output_size: int = 800,
        merged_extent_vggt: float = 6.5,
        predict_scale_uncertainty: bool = True,
        implicit_geometry_hidden_dim: int = 256,
        implicit_geometry_heads: int = 8,
        implicit_geometry_layers: int = 4,
        maximum_history: int = 10,
        maximum_prefix_tokens: int = 17,
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
        self.implicit_geometry_trunk = ImplicitGeometryContextTrunk(
            input_dim=vggt_token_dim,
            hidden_dim=implicit_geometry_hidden_dim,
            output_dim=hidden_dim,
            heads=implicit_geometry_heads,
            layers=implicit_geometry_layers,
            maximum_history=maximum_history,
            maximum_prefix_tokens=maximum_prefix_tokens,
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
        *,
        include_merged: bool = True,
        include_scale: bool = True,
        assemble_runtime_outputs: bool = True,
    ) -> dict:
        output: dict = {}
        if include_merged:
            if "camera_register_tokens" not in extraction:
                raise KeyError("Merged head requires frozen aggregator prefix tokens")
            frame_embedding = self.implicit_geometry_trunk(
                extraction["camera_register_tokens"]
            )
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

    def decode_scale_teacher(self, extraction: dict) -> dict:
        return self.adapter.decode_scale_teacher(extraction)

    def forward_head(self, extraction: dict, **arguments) -> dict:
        prediction = self.head(extraction, **arguments)
        head = self.unwrapped_head()
        return {
            **prediction,
            "pipeline_id": head.pipeline_id,
            "runtime_inputs": ("rgb_window",),
            "merged_source": (
                "frozen aggregator patch+camera/register tokens only"
            ),
            "coordinate_mode": "vggt_native_units",
            "merged_extent_vggt": head.merged_bev_decoder.extent_vggt,
            "merged_output_size": head.merged_bev_decoder.output_size,
            "scale_unit": "meter_per_vggt_runtime_unit",
            "scale_is_merged_input": False,
            "single_bev_present": False,
            "relative_pose_head_present": False,
            "extrinsic_input_present": False,
            "bev_waits_for_geometry_heads": False,
            "geometry_conditioning": (
                "implicit_multiview_token_cross_attention"
            ),
            "maximum_history": head.implicit_geometry_trunk.maximum_history,
            "orientation": "latest ego centered; forward is image-up",
        }

    def forward(self, images: torch.Tensor) -> dict:
        extraction = self.extract(images)
        return self.forward_head(extraction)
