from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .attention import DirectDecoderBlock, MultiheadAttention


def compose_fov_complete_semantic(
    prediction: dict[str, torch.Tensor],
    *,
    support_threshold: float = 0.5,
    occupancy_threshold: float = 0.5,
    occupied_value: int = 0,
    unknown_value: int = 112,
    free_value: int = 255,
) -> torch.Tensor:
    """Assemble the deployable masked-complete raster from probabilistic output."""

    support = prediction["fov_support_probability"]
    occupancy = prediction["occupancy_probability"]
    if support.shape != occupancy.shape:
        raise ValueError("FOV support and occupancy probability shapes must match")
    if not 0.0 <= support_threshold <= 1.0:
        raise ValueError("support threshold must be inside [0, 1]")
    if not 0.0 <= occupancy_threshold <= 1.0:
        raise ValueError("occupancy threshold must be inside [0, 1]")
    output = torch.full_like(support, unknown_value, dtype=torch.uint8)
    inside = support >= support_threshold
    output[inside & (occupancy >= occupancy_threshold)] = occupied_value
    output[inside & (occupancy < occupancy_threshold)] = free_value
    return output


class MultiScaleTokenProjector(nn.Module):
    """Project frozen VGGT patch tokens into the trainable P1B width."""

    def __init__(
        self,
        layers: tuple[int, ...],
        input_dim: int,
        hidden_dim: int,
        spatial_scales: tuple[float, ...],
    ) -> None:
        super().__init__()
        if len(layers) != len(spatial_scales) or not layers:
            raise ValueError("every cached token layer needs one spatial scale")
        if any(scale <= 0 for scale in spatial_scales):
            raise ValueError("spatial token scales must be positive")
        self.layers = layers
        self.spatial_scales = spatial_scales
        self.projections = nn.ModuleDict(
            {
                str(layer): nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                )
                for layer in layers
            }
        )
        self.level_embedding = nn.Parameter(
            torch.empty(len(layers), 1, 1, 1, hidden_dim)
        )
        nn.init.normal_(self.level_embedding, std=0.02)

    def forward(
        self,
        tokens: dict[int, torch.Tensor],
        patch_grid: tuple[int, int],
        frame_embedding: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        patch_height, patch_width = patch_grid
        output = []
        for level_index, layer in enumerate(self.layers):
            if layer not in tokens:
                raise KeyError(f"missing cached VGGT layer {layer}")
            projected = self.projections[str(layer)](tokens[layer])
            batch, frames, patches, channels = projected.shape
            if frame_embedding is not None:
                if frame_embedding.shape != (batch, frames, channels):
                    raise ValueError(
                        "frame_embedding must have shape [B,N,hidden_dim]"
                    )
                projected = projected + frame_embedding[:, :, None, :]
            if patches != patch_height * patch_width:
                raise ValueError(
                    f"layer {layer} has {patches} tokens but grid is "
                    f"{patch_height}x{patch_width}"
                )
            projected = projected + self.level_embedding[level_index]
            feature = projected.view(
                batch,
                frames,
                patch_height,
                patch_width,
                channels,
            ).permute(0, 1, 4, 2, 3)
            scale = self.spatial_scales[level_index]
            level_height = max(1, round(patch_height * scale))
            level_width = max(1, round(patch_width * scale))
            feature = F.interpolate(
                feature.flatten(0, 1),
                size=(level_height, level_width),
                mode="bilinear",
                align_corners=False,
            ).view(batch, frames, channels, level_height, level_width)
            output.append(feature)
        return output


def _metric_query_coordinates(
    size: int,
    extent_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return centered metric cell coordinates and deformable reference grid.

    Existing simulator labels place the latest ego at the raster centre and
    define image-up as forward.  This centered 6.5 m convention is kept
    exactly across data, model, validation, and runtime.
    """

    cell = extent_m / size
    pixel = torch.arange(size, dtype=torch.float32)
    x = -extent_m / 2 + (pixel + 0.5) * cell
    z = extent_m / 2 - (pixel + 0.5) * cell
    z_grid, x_grid = torch.meshgrid(z, x, indexing="ij")
    metric = torch.stack((x_grid, z_grid), dim=-1).reshape(-1, 2)
    reference = torch.stack(
        (
            x_grid / (extent_m / 2),
            -z_grid / (extent_m / 2),
        ),
        dim=-1,
    ).reshape(-1, 2)
    return metric, reference


class FixedMetricBEVDecoder(nn.Module):
    """Decode FOV support and occupied/free Beta evidence in that support."""

    class_order = ("free", "occupied")

    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        layers: int,
        feature_levels: int,
        latent_size: int = 64,
        output_size: int = 512,
        extent_m: float = 6.5,
        self_attention_mode: str = "linear",
        cross_attention_mode: str = "deformable",
        deformable_samples: int = 4,
        cross_query_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        if latent_size <= 0 or output_size <= 0 or extent_m <= 0:
            raise ValueError("BEV sizes and extent must be positive")
        self.latent_size = latent_size
        self.output_size = output_size
        self.extent_m = extent_m
        metric, reference = _metric_query_coordinates(latent_size, extent_m)
        self.register_buffer("metric_coordinates_m", metric, persistent=True)
        self.register_buffer("reference_grid", reference, persistent=True)
        self.query_content = nn.Parameter(
            torch.empty(latent_size * latent_size, hidden_dim)
        )
        self.metric_position = nn.Sequential(
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
        self.upsampler = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, 1),
        )
        nn.init.normal_(self.query_content, std=0.02)

    def forward(self, pyramid: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = pyramid[0].shape[0]
        position = self.metric_position(
            self.metric_coordinates_m.to(dtype=self.query_content.dtype)
            / (self.extent_m / 2)
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
            batch,
            -1,
            self.latent_size,
            self.latent_size,
        )
        latent_raw_output = self.upsampler(latent)
        raw_output = F.interpolate(
            latent_raw_output,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        # Confidence is not a sigmoid output. The two channels parameterize
        # occupied/free evidence for a Beta distribution. The independent
        # third channel predicts geometric FOV support, not confidence.
        raw_evidence = raw_output[:, :2]
        fov_support_logit = raw_output[:, 2]
        evidence = F.softplus(raw_evidence.float())
        alpha_occupied = evidence[:, 0] + 1.0
        beta_free = evidence[:, 1] + 1.0
        strength = alpha_occupied + beta_free
        return {
            "raw_output": raw_output,
            "raw_evidence": raw_evidence,
            "fov_support_logit": fov_support_logit,
            "fov_support_probability": torch.sigmoid(
                fov_support_logit.float()
            ),
            "alpha_occupied": alpha_occupied,
            "beta_free": beta_free,
            "evidence_strength": strength,
            "occupancy_probability": alpha_occupied / strength,
            "occupancy_distribution_variance": (
                alpha_occupied
                * beta_free
                / (strength.square() * (strength + 1.0))
            ),
            "epistemic_uncertainty": 2.0 / strength,
            "evidence_confidence": (1.0 - 2.0 / strength).clamp(0.0, 1.0),
        }


class MetricScaleTokenHead(nn.Module):
    """Predict log metric scale from the same frozen VGGT token memory."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        *,
        layers: int = 2,
        predict_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        self.scale_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.level_embedding = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.blocks = nn.ModuleList(
            [MultiheadAttention(hidden_dim, heads, mode="exact") for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.log_scale = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_variance = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if predict_uncertainty
            else None
        )
        nn.init.normal_(self.scale_token, std=0.02)
        nn.init.normal_(self.level_embedding, std=0.02)

    def forward(self, pyramid: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = pyramid[0].shape[0]
        # Each frame/feature-level contributes equally; high-resolution levels
        # cannot dominate the scalar estimate merely by containing more patches.
        memory = torch.cat(
            [level.mean(dim=(-2, -1)) for level in pyramid],
            dim=1,
        )
        memory = memory + self.level_embedding
        token = self.scale_token.expand(batch, -1, -1)
        for norm, attention in zip(self.norms, self.blocks, strict=True):
            token = token + attention(norm(token), memory)
        feature = self.final_norm(token[:, 0])
        log_lambda = self.log_scale(feature)[:, 0].clamp(-8.0, 8.0)
        output = {
            "log_lambda_m_per_vggt": log_lambda,
            "lambda_m_per_vggt": log_lambda.exp(),
        }
        if self.log_variance is not None:
            log_variance = self.log_variance(feature)[:, 0].clamp(-8.0, 8.0)
            output["log_variance"] = log_variance
            output["scale_std_m_per_vggt"] = (0.5 * log_variance).exp()
        return output


class Method1Head(nn.Module):
    """P1B: FOV-complete evidential BEVs plus parallel Scale Token."""

    def __init__(
        self,
        *,
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
        single_latent_bev_size: int = 64,
        merged_latent_bev_size: int = 80,
        single_output_size: int = 512,
        merged_output_size: int = 800,
        single_bev_extent_m: float = 6.5,
        merged_bev_extent_m: float = 10.0,
        predict_scale_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        self.token_projector = MultiScaleTokenProjector(
            cached_layers,
            vggt_token_dim,
            hidden_dim,
            spatial_scales,
        )
        # The scale branch gets its own adapter. BEV gradients must not move
        # the representation used by the scalar scale estimate.
        self.scale_token_projector = MultiScaleTokenProjector(
            cached_layers,
            vggt_token_dim,
            hidden_dim,
            spatial_scales,
        )
        self.single_bev_decoder = FixedMetricBEVDecoder(
            hidden_dim=hidden_dim,
            heads=heads,
            layers=decoder_layers,
            feature_levels=len(cached_layers),
            latent_size=single_latent_bev_size,
            output_size=single_output_size,
            extent_m=single_bev_extent_m,
            self_attention_mode=self_attention_mode,
            cross_attention_mode=cross_attention_mode,
            deformable_samples=deformable_samples,
            cross_query_chunk_size=cross_query_chunk_size,
        )
        self.merged_bev_decoder = FixedMetricBEVDecoder(
            hidden_dim=hidden_dim,
            heads=heads,
            layers=decoder_layers,
            feature_levels=len(cached_layers),
            latent_size=merged_latent_bev_size,
            output_size=merged_output_size,
            extent_m=merged_bev_extent_m,
            self_attention_mode=self_attention_mode,
            cross_attention_mode=cross_attention_mode,
            deformable_samples=deformable_samples,
            cross_query_chunk_size=cross_query_chunk_size,
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
        enabled_bev_branches: tuple[str, ...] = ("single", "merged"),
        include_scale: bool = True,
    ) -> dict:
        unknown = set(enabled_bev_branches).difference(("single", "merged"))
        if unknown:
            raise ValueError(f"unknown BEV branches: {sorted(unknown)}")
        if len(set(enabled_bev_branches)) != len(enabled_bev_branches):
            raise ValueError("enabled BEV branches must be unique")
        if not enabled_bev_branches and not include_scale:
            raise ValueError("at least one P1B output branch must be enabled")

        prediction = {}
        if enabled_bev_branches:
            pyramid = self.token_projector(
                extraction["tokens"],
                extraction["patch_grid"],
            )
            if "single" in enabled_bev_branches:
                # Single-frame BEV is decoded only from latest-frame features.
                prediction["single_bev"] = self.single_bev_decoder(
                    [level[:, -1:] for level in pyramid]
                )
            if "merged" in enabled_bev_branches:
                # Merged BEV directly attends the complete RGB-window memory.
                # It never consumes or fuses the predicted single-frame BEV.
                prediction["merged_bev"] = self.merged_bev_decoder(pyramid)
        if include_scale:
            scale_pyramid = self.scale_token_projector(
                extraction["tokens"],
                extraction["patch_grid"],
            )
            prediction["scale"] = self.scale_decoder(scale_pyramid)
        return prediction


class Method1System(nn.Module):
    """Frozen VGGT aggregator plus the FOV-complete evidential P1B head.

    ``forward`` is the runtime path and deliberately does not execute VGGT's
    camera/depth heads.  Training may call ``decode_teacher_geometry`` after
    the one shared aggregator pass to construct the metric-scale target.
    """

    def __init__(self, adapter: nn.Module, **head_arguments) -> None:
        super().__init__()
        self.adapter = adapter
        self.head = Method1Head(**head_arguments)

    def unwrapped_head(self) -> Method1Head:
        module = getattr(self.head, "module", self.head)
        if not isinstance(module, Method1Head):
            raise TypeError("P1B trainable head has an unexpected module type")
        return module

    def extract(self, images: torch.Tensor) -> dict:
        return self.adapter.aggregate(images)

    def forward_head(
        self,
        extraction: dict,
        *,
        enabled_bev_branches: tuple[str, ...] = ("single", "merged"),
        include_scale: bool = True,
    ) -> dict:
        prediction = self.head(
            extraction,
            enabled_bev_branches=enabled_bev_branches,
            include_scale=include_scale,
        )
        for branch_name in (
            f"{branch}_bev" for branch in enabled_bev_branches
        ):
            branch = prediction[branch_name]
            branch["fov_complete_semantic"] = compose_fov_complete_semantic(branch)
            # Evidence confidence is meaningful only where this parallel
            # support branch says the camera-window FOV applies.
            branch["navigation_confidence"] = (
                branch["fov_support_probability"]
                * branch["evidence_confidence"]
            )
        single = self.unwrapped_head().single_bev_decoder
        merged = self.unwrapped_head().merged_bev_decoder
        output = {
            **prediction,
            "coordinate_mode": "p1b_fixed_metric",
            "enabled_bev_branches": enabled_bev_branches,
            "scale_enabled": include_scale,
            "orientation": "latest ego centered; forward is image-up",
            "bev_content": (
                "occupied/free inside predicted camera-FOV union; "
                "unknown outside"
            ),
            "bev_support": "parallel RGB-only FOV-support probability",
            "bev_confidence": "Beta evidence; no sigmoid confidence head",
            "bev_waits_for_geometry_heads": False,
            "runtime_inputs": ("rgb_window",),
        }
        if "single" in enabled_bev_branches:
            output.update(
                {
                    "single_bev_extent_m": single.extent_m,
                    "single_bev_output_size": single.output_size,
                    "single_bev_cell_size_m": (
                        single.extent_m / single.output_size
                    ),
                    "single_bev_bounds_m": (
                        -single.extent_m / 2,
                        single.extent_m / 2,
                        -single.extent_m / 2,
                        single.extent_m / 2,
                    ),
                }
            )
        if "merged" in enabled_bev_branches:
            output.update(
                {
                    "merged_bev_extent_m": merged.extent_m,
                    "merged_bev_output_size": merged.output_size,
                    "merged_bev_cell_size_m": (
                        merged.extent_m / merged.output_size
                    ),
                    "merged_bev_bounds_m": (
                        -merged.extent_m / 2,
                        merged.extent_m / 2,
                        -merged.extent_m / 2,
                        merged.extent_m / 2,
                    ),
                }
            )
        return output

    def decode_teacher_geometry(self, extraction: dict) -> dict:
        return self.adapter.decode_geometry(extraction)

    def forward(self, images: torch.Tensor) -> dict:
        return self.forward_head(self.extract(images))


MetricP1BHead = Method1Head
MetricP1BSystem = Method1System
