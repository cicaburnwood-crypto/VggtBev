from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .attention import DirectDecoderBlock
from .method1 import (
    MetricScaleTokenHead,
    MultiScaleTokenProjector,
    _metric_query_coordinates,
)
from .p1b_probability import (
    ProbabilityModel,
    compose_semantic,
    decode_binary_prediction,
    fuse_pixel_routing,
)

_LEGACY_PROJECTOR_PREFIXES = {
    "guessed_token_projector.": (
        "single_guessed_token_projector.",
        "merged_guessed_token_projector.",
    ),
    "routing_token_projector.": (
        "single_routing_token_projector.",
        "merged_routing_token_projector.",
    ),
}


def branch_specific_projector_state_dict(state_dict: dict) -> dict:
    """Expand a historical shared-projector head state without changing outputs.

    Old checkpoints used one Guessed and one Routing projector for both BEV
    branches. Copying each tensor into both new branch-local projectors keeps
    old Single and Merged runtime predictions exactly equivalent at load time.
    """

    has_legacy = any(
        key.startswith(prefix)
        for prefix in _LEGACY_PROJECTOR_PREFIXES
        for key in state_dict
    )
    if not has_legacy:
        return state_dict
    has_branch_specific = any(
        key.startswith(target)
        for targets in _LEGACY_PROJECTOR_PREFIXES.values()
        for target in targets
        for key in state_dict
    )
    if has_branch_specific:
        raise RuntimeError(
            "checkpoint mixes shared and branch-specific P1B projectors"
        )
    migrated = state_dict.copy()
    if hasattr(state_dict, "_metadata"):
        migrated._metadata = state_dict._metadata  # type: ignore[attr-defined]
    for source_prefix, target_prefixes in _LEGACY_PROJECTOR_PREFIXES.items():
        source_keys = [
            key for key in tuple(migrated) if key.startswith(source_prefix)
        ]
        for source_key in source_keys:
            value = migrated.pop(source_key)
            suffix = source_key.removeprefix(source_prefix)
            for target_prefix in target_prefixes:
                migrated[target_prefix + suffix] = value
    return migrated


class DenseMetricQueryDecoder(nn.Module):
    """Independent metric-query decoder used by one P1B role."""

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
        extent_m: float,
        self_attention_mode: str,
        cross_attention_mode: str,
        deformable_samples: int,
        cross_query_chunk_size: int,
    ) -> None:
        super().__init__()
        if output_channels <= 0:
            raise ValueError("decoder output channels must be positive")
        self.latent_size = int(latent_size)
        self.output_size = int(output_size)
        self.extent_m = float(extent_m)
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
        self.output_projection = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_channels, 1),
        )
        nn.init.normal_(self.query_content, std=0.02)

    def forward(self, pyramid: list[torch.Tensor]) -> torch.Tensor:
        batch = pyramid[0].shape[0]
        position = self.metric_position(
            self.metric_coordinates_m.to(dtype=self.query_content.dtype)
            / (self.extent_m / 2.0)
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


class PixelRoutedBEVDecoder(nn.Module):
    """Observed-free Gate plus one learned guessed occupancy decoder."""

    def __init__(self, *, probability_model: ProbabilityModel, **arguments) -> None:
        super().__init__()
        self.probability_model = probability_model
        expert_channels = 2 if probability_model == "evidential" else 1
        self.guessed = DenseMetricQueryDecoder(
            output_channels=expert_channels,
            **arguments,
        )
        # Observed-free Gate and independent FOV Support. Occupied boundary
        # pixels are handled by the guessed expert; there is no Surface Gate.
        self.routing = DenseMetricQueryDecoder(output_channels=2, **arguments)
        self.output_size = self.guessed.output_size
        self.extent_m = self.guessed.extent_m

    def forward(
        self,
        guessed_pyramid: list[torch.Tensor],
        routing_pyramid: list[torch.Tensor],
        *,
        assemble_runtime_outputs: bool = True,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        guessed = decode_binary_prediction(
            self.guessed(guessed_pyramid),
            self.probability_model,
            include_diagnostics=assemble_runtime_outputs,
        )
        routing_raw = self.routing(routing_pyramid).float()
        observed_gate_logit = routing_raw[:, 0]
        support_logit = routing_raw[:, 1]
        support_probability = torch.sigmoid(support_logit)
        output: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {
            "guessed": guessed,
            "observed_gate_logit": observed_gate_logit,
            "fov_support_logit": support_logit,
            "fov_support_probability": support_probability,
        }
        if not assemble_runtime_outputs:
            return output
        observed_gate_probability = torch.sigmoid(observed_gate_logit)
        guessed_region_probability = 1.0 - observed_gate_probability
        guessed_occupied_probability = (
            guessed_region_probability * guessed["occupancy_probability"].float()
        )
        guessed_free_probability = (
            guessed_region_probability
            * (1.0 - guessed["occupancy_probability"].float())
        )
        routing_probability = torch.stack(
            (
                observed_gate_probability,
                guessed_free_probability,
                guessed_occupied_probability,
            ),
            dim=1,
        )
        fused = fuse_pixel_routing(
            routing_probability,
            guessed,
            support_probability,
            self.probability_model,
        )
        output.update({
            "routing_probability": routing_probability,
            "routing_class": routing_probability.argmax(dim=1),
            "observed_gate_logit": observed_gate_logit,
            "observed_gate_probability": observed_gate_probability,
            "observed_free_probability": routing_probability[:, 0],
            "guessed_free_probability": routing_probability[:, 1],
            "guessed_occupied_probability": routing_probability[:, 2],
            "guessed_region_probability": guessed_region_probability,
            "fov_support_logit": support_logit,
            "fov_support_probability": support_probability,
            "fused": fused,
            "occupancy_probability": fused["occupancy_probability"],
            "navigation_confidence": fused["navigation_confidence"],
            "fov_complete_semantic": compose_semantic(
                fused, support_probability
            ),
        })
        return output

    def forward_routing_geometry(
        self,
        routing_pyramid: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Decode only Observed Gate and FOV support for the Merged stage.

        The Guessed decoder is deliberately not executed.  Both routing
        channels receive direct pixelwise supervision in the combined stage.
        """

        routing_raw = self.routing(routing_pyramid).float()
        observed_gate_logit = routing_raw[:, 0]
        support_logit = routing_raw[:, 1]
        return {
            "observed_gate_logit": observed_gate_logit,
            "observed_gate_probability": torch.sigmoid(observed_gate_logit),
            "fov_support_logit": support_logit,
            "fov_support_probability": torch.sigmoid(support_logit),
        }


class P1BHead(nn.Module):
    """Observed-free routing, guessed completion and metric Scale Token."""

    def __init__(
        self,
        *,
        probability_model: ProbabilityModel,
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
        single_latent_bev_size: int = 512,
        merged_latent_bev_size: int = 80,
        single_output_size: int = 512,
        merged_output_size: int = 800,
        single_bev_extent_m: float = 6.5,
        merged_bev_extent_m: float = 10.0,
        predict_scale_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        if probability_model not in ("evidential", "bce"):
            raise ValueError("probability_model must be evidential or bce")
        if int(single_latent_bev_size) != int(single_output_size):
            raise ValueError(
                "P1B single BEV must decode natively at output resolution; "
                "latent upsampling is forbidden"
            )
        self.probability_model: ProbabilityModel = probability_model
        projector_arguments = (
            cached_layers,
            vggt_token_dim,
            hidden_dim,
            spatial_scales,
        )
        # Single and Merged share only frozen raw VGGT tokens. Every trainable
        # projector and decoder is branch-local so Merged optimization cannot
        # change a frozen Single baseline.
        self.single_guessed_token_projector = MultiScaleTokenProjector(
            *projector_arguments
        )
        self.single_routing_token_projector = MultiScaleTokenProjector(
            *projector_arguments
        )
        self.merged_guessed_token_projector = MultiScaleTokenProjector(
            *projector_arguments
        )
        self.merged_routing_token_projector = MultiScaleTokenProjector(
            *projector_arguments
        )
        self.scale_token_projector = MultiScaleTokenProjector(
            *projector_arguments
        )
        common = dict(
            hidden_dim=hidden_dim,
            heads=heads,
            layers=decoder_layers,
            feature_levels=len(cached_layers),
            self_attention_mode=self_attention_mode,
            cross_attention_mode=cross_attention_mode,
            deformable_samples=deformable_samples,
            cross_query_chunk_size=cross_query_chunk_size,
        )
        self.single_bev_decoder = PixelRoutedBEVDecoder(
            probability_model=probability_model,
            latent_size=single_latent_bev_size,
            output_size=single_output_size,
            extent_m=single_bev_extent_m,
            **common,
        )
        self.merged_bev_decoder = PixelRoutedBEVDecoder(
            probability_model=probability_model,
            latent_size=merged_latent_bev_size,
            output_size=merged_output_size,
            extent_m=merged_bev_extent_m,
            **common,
        )
        self.scale_decoder = MetricScaleTokenHead(
            hidden_dim,
            heads,
            layers=scale_decoder_layers,
            predict_uncertainty=predict_scale_uncertainty,
        )

    def load_state_dict(
        self,
        state_dict: dict,
        strict: bool = True,
        assign: bool = False,
    ):
        return super().load_state_dict(
            branch_specific_projector_state_dict(state_dict),
            strict=strict,
            assign=assign,
        )

    def _pyramids(
        self,
        extraction: dict,
        branch: str,
    ) -> tuple[list[torch.Tensor], ...]:
        if branch not in ("single", "merged"):
            raise ValueError(f"unknown BEV branch: {branch}")
        tokens = extraction["tokens"]
        grid = extraction["patch_grid"]
        return (
            getattr(self, f"{branch}_guessed_token_projector")(tokens, grid),
            getattr(self, f"{branch}_routing_token_projector")(tokens, grid),
        )

    def forward(
        self,
        extraction: dict,
        *,
        enabled_bev_branches: tuple[str, ...] = ("single", "merged"),
        include_scale: bool = True,
        assemble_runtime_outputs: bool = True,
        bev_objective: str = "full",
    ) -> dict:
        unknown = set(enabled_bev_branches).difference(("single", "merged"))
        if unknown:
            raise ValueError(f"unknown BEV branches: {sorted(unknown)}")
        if bev_objective not in (
            "full",
            "fov_support_only",
            "fov_support_and_observed_gate",
        ):
            raise ValueError(f"unknown BEV objective: {bev_objective}")
        if bev_objective != "full" and enabled_bev_branches != ("merged",):
            raise ValueError("routing-geometry decoding requires Merged only")
        output: dict = {}
        if "single" in enabled_bev_branches:
            guessed, routing = self._pyramids(extraction, "single")
            output["single_bev"] = self.single_bev_decoder(
                [level[:, -1:] for level in guessed],
                [level[:, -1:] for level in routing],
                assemble_runtime_outputs=assemble_runtime_outputs,
            )
        if "merged" in enabled_bev_branches:
            if bev_objective != "full":
                routing = self.merged_routing_token_projector(
                    extraction["tokens"], extraction["patch_grid"]
                )
                output["merged_bev"] = (
                    self.merged_bev_decoder.forward_routing_geometry(routing)
                )
            else:
                guessed, routing = self._pyramids(extraction, "merged")
                output["merged_bev"] = self.merged_bev_decoder(
                    guessed,
                    routing,
                    assemble_runtime_outputs=assemble_runtime_outputs,
                )
        if include_scale:
            scale_pyramid = self.scale_token_projector(
                extraction["tokens"], extraction["patch_grid"]
            )
            output["scale"] = self.scale_decoder(scale_pyramid)
        return output


class P1BSystem(nn.Module):
    """One frozen VGGT aggregation followed by the trainable P1B head."""

    def __init__(self, adapter: nn.Module, **head_arguments) -> None:
        super().__init__()
        self.adapter = adapter
        self.head = P1BHead(**head_arguments)

    def unwrapped_head(self) -> P1BHead:
        module = getattr(self.head, "module", self.head)
        if not isinstance(module, P1BHead):
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
        assemble_runtime_outputs: bool = True,
        bev_objective: str = "full",
    ) -> dict:
        prediction = self.head(
            extraction,
            enabled_bev_branches=enabled_bev_branches,
            include_scale=include_scale,
            assemble_runtime_outputs=assemble_runtime_outputs,
            bev_objective=bev_objective,
        )
        head = self.unwrapped_head()
        output = {
            **prediction,
            "pipeline_id": (
                "P1B-NLL"
                if head.probability_model == "evidential"
                else "P1B-BCE"
            ),
            "probability_model": head.probability_model,
            "coordinate_mode": "p1b_fixed_metric",
            "enabled_bev_branches": enabled_bev_branches,
            "scale_enabled": include_scale,
            "orientation": "latest ego centered; forward is image-up",
            "runtime_inputs": ("rgb_window",),
            "bev_waits_for_geometry_heads": False,
        }
        for name in enabled_bev_branches:
            decoder = getattr(head, f"{name}_bev_decoder")
            output[f"{name}_bev_extent_m"] = decoder.extent_m
            output[f"{name}_bev_output_size"] = decoder.output_size
            output[f"{name}_bev_cell_size_m"] = (
                decoder.extent_m / decoder.output_size
            )
        return output

    def decode_teacher_geometry(self, extraction: dict) -> dict:
        return self.adapter.decode_geometry(extraction)

    def forward(self, images: torch.Tensor) -> dict:
        return self.forward_head(self.extract(images))
