from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vggt_bev_method1.config import Supervision

from .decoder import DecoderOutputMode, DirectBEVDecoder


def _assemble_runtime_fov_output(
    prediction: dict,
    geometry: dict[str, torch.Tensor],
) -> dict:
    """Attach explicit geometry support and deployable Model-B rasters."""

    quality = geometry["geometry_quality"][:, None, None]
    valid = geometry["geometry_valid"][:, None, None]
    for branch in ("single", "merged"):
        branch_prediction = prediction[branch]
        support = geometry[f"{branch}_fov_support"] & valid
        probability = branch_prediction["occupancy_probability"]
        confidence = branch_prediction["evidence_confidence"]
        semantic = torch.full(
            probability.shape,
            112,
            dtype=torch.uint8,
            device=probability.device,
        )
        semantic[support & (probability >= 0.5)] = 0
        semantic[support & (probability < 0.5)] = 255
        branch_prediction.update(
            {
                "runtime_fov_support": support,
                "fov_complete_semantic": semantic,
                "navigation_confidence": (
                    confidence * support.to(confidence.dtype) * quality
                ),
            }
        )
    return prediction


def _assemble_runtime_observed_output(
    prediction: dict,
    geometry: dict[str, torch.Tensor],
) -> dict:
    """Expose Model A as a support-capped masked semantic BEV."""

    valid = geometry["geometry_valid"][:, None, None]
    for branch in ("single", "merged"):
        branch_prediction = prediction[branch]
        support = geometry[f"{branch}_fov_support"] & valid
        predicted_class = branch_prediction["class_logits"].argmax(dim=1)
        semantic = torch.full(
            predicted_class.shape,
            112,
            dtype=torch.uint8,
            device=predicted_class.device,
        )
        semantic[support & (predicted_class == 1)] = 255
        semantic[support & (predicted_class == 2)] = 0
        branch_prediction.update(
            {
                "runtime_fov_support": support,
                "masked_observed_semantic": semantic,
            }
        )
    return prediction


class MultiScaleTokenProjector(nn.Module):
    """Equation 24: project each level to C_B, then concatenate tokens."""

    def __init__(
        self,
        layers: tuple[int, ...],
        input_dim: int,
        hidden_dim: int,
        spatial_scales: tuple[float, ...],
    ) -> None:
        super().__init__()
        if len(layers) != len(spatial_scales):
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
    ) -> list[torch.Tensor]:
        patch_height, patch_width = patch_grid
        output = []
        for level_index, layer in enumerate(self.layers):
            projected = self.projections[str(layer)](tokens[layer])
            batch, frames, patches, channels = projected.shape
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


class Method1Head(nn.Module):
    """Trainable Method I token projector and dual BEV decoders."""

    def __init__(
        self,
        *,
        supervision: Supervision,
        cached_layers: tuple[int, ...] = (4, 11, 17, 23),
        spatial_scales: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
        vggt_token_dim: int = 2048,
        hidden_dim: int = 64,
        geometry_cue_dim: int = 24,
        heads: int = 8,
        decoder_layers: int = 2,
        self_attention_mode: str = "linear",
        cross_attention_mode: str = "deformable",
        deformable_samples: int = 4,
        cross_query_chunk_size: int = 8192,
        query_parameter_chunk_size: int = 32768,
        single_output_size: int = 512,
        merged_output_size: int = 800,
        gradient_checkpointing: bool = True,
        output_mode: DecoderOutputMode = "legacy",
    ) -> None:
        super().__init__()
        self.supervision = supervision
        self.token_projector = MultiScaleTokenProjector(
            cached_layers,
            vggt_token_dim,
            hidden_dim,
            spatial_scales,
        )
        decoder_arguments = {
            "supervision": supervision,
            "hidden_dim": hidden_dim,
            "geometry_cue_dim": geometry_cue_dim,
            "heads": heads,
            "layers": decoder_layers,
            "self_attention_mode": self_attention_mode,
            "cross_attention_mode": cross_attention_mode,
            "feature_levels": len(cached_layers),
            "deformable_samples": deformable_samples,
            "cross_query_chunk_size": cross_query_chunk_size,
            "query_parameter_chunk_size": query_parameter_chunk_size,
            "gradient_checkpointing": gradient_checkpointing,
            "output_mode": output_mode,
        }
        self.single_decoder = DirectBEVDecoder(
            output_size=single_output_size,
            **decoder_arguments,
        )
        self.merged_decoder = DirectBEVDecoder(
            output_size=merged_output_size,
            **decoder_arguments,
        )

    def forward(
        self,
        extraction: dict,
        *,
        geometry_gate: float = 1.0,
    ) -> dict:
        projected_levels = self.token_projector(
            extraction["tokens"],
            extraction["patch_grid"],
        )
        single_context = [level[:, -1:] for level in projected_levels]
        merged_context = projected_levels
        geometry_cue = extraction["geometry_cue"]
        p1a_geometry = extraction["p1a_geometry"]
        return {
            "single": self.single_decoder(
                single_context,
                geometry_cue,
                extent_normalized_scale=p1a_geometry[
                    "single_extent_normalized_scale"
                ],
                geometry_gate=geometry_gate,
            ),
            "merged": self.merged_decoder(
                merged_context,
                geometry_cue,
                extent_normalized_scale=p1a_geometry[
                    "merged_extent_normalized_scale"
                ],
                geometry_gate=geometry_gate,
            ),
        }


class Method1System(nn.Module):
    """Frozen live VGGT-Ω plus a separately wrappable trainable Method I head."""

    def __init__(
        self,
        adapter: nn.Module,
        *,
        supervision: Supervision,
        cached_layers: tuple[int, ...] = (4, 11, 17, 23),
        spatial_scales: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
        vggt_token_dim: int = 2048,
        hidden_dim: int = 64,
        geometry_cue_dim: int = 24,
        heads: int = 8,
        decoder_layers: int = 2,
        self_attention_mode: str = "linear",
        cross_attention_mode: str = "deformable",
        deformable_samples: int = 4,
        cross_query_chunk_size: int = 8192,
        query_parameter_chunk_size: int = 32768,
        single_output_size: int = 512,
        merged_output_size: int = 800,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.supervision = supervision
        self.head = Method1Head(
            supervision=supervision,
            cached_layers=cached_layers,
            spatial_scales=spatial_scales,
            vggt_token_dim=vggt_token_dim,
            hidden_dim=hidden_dim,
            geometry_cue_dim=geometry_cue_dim,
            heads=heads,
            decoder_layers=decoder_layers,
            self_attention_mode=self_attention_mode,
            cross_attention_mode=cross_attention_mode,
            deformable_samples=deformable_samples,
            cross_query_chunk_size=cross_query_chunk_size,
            query_parameter_chunk_size=query_parameter_chunk_size,
            single_output_size=single_output_size,
            merged_output_size=merged_output_size,
            gradient_checkpointing=gradient_checkpointing,
        )

    def unwrapped_head(self) -> Method1Head:
        module = getattr(self.head, "module", self.head)
        if not isinstance(module, Method1Head):
            raise TypeError("Method I trainable head has an unexpected module type")
        return module

    @property
    def token_projector(self) -> MultiScaleTokenProjector:
        return self.unwrapped_head().token_projector

    @property
    def single_decoder(self) -> DirectBEVDecoder:
        return self.unwrapped_head().single_decoder

    @property
    def merged_decoder(self) -> DirectBEVDecoder:
        return self.unwrapped_head().merged_decoder

    def forward(
        self,
        images: torch.Tensor,
        camera_height_m: torch.Tensor,
        *,
        geometry_gate: float = 1.0,
    ) -> dict:
        extraction = self.adapter(images, camera_height_m)
        prediction = self.head(
            extraction,
            geometry_gate=geometry_gate,
        )
        return {
            **prediction,
            "estimated_geometry_cue": extraction["geometry_cue"],
            "scene_radius_vggt": extraction["scene_radius_vggt"],
            "estimated_depth_vggt": extraction.get("estimated_depth_vggt"),
            "estimated_intrinsics": extraction.get("estimated_intrinsics"),
            "estimated_camera_from_world_vggt": extraction.get(
                "estimated_camera_from_world_vggt"
            ),
            "geometry_source": extraction["geometry_source"],
            "coordinate_mode": "camera_height_anchored_fixed_normalized_scale",
            "extent_mode": "fixed_6p5_single_10_merged",
            "single_output_size": self.single_decoder.output_size,
            "merged_output_size": self.merged_decoder.output_size,
            "single_extent_normalized_scale": prediction["single"][
                "extent_normalized_scale"
            ],
            "merged_extent_normalized_scale": prediction["merged"][
                "extent_normalized_scale"
            ],
            "single_cell_size_normalized_scale": prediction["single"][
                "cell_size_normalized_scale"
            ],
            "merged_cell_size_normalized_scale": prediction["merged"][
                "cell_size_normalized_scale"
            ],
            "camera_height_used": True,
            "metric_scale_used": True,
            "runtime_aligned": True,
            "p1a_geometry": extraction["p1a_geometry"],
        }


class PairedMethod1System(nn.Module):
    """One frozen VGGT pass feeding one or two Method I models.

    The observed and complete paths share no trainable parameters. They only
    consume the same detached live-VGGT extraction, so either loss can be
    optimized and checkpointed without changing the other model. Model A can
    be omitted entirely for complete-evidential-only training.
    """

    def __init__(
        self,
        adapter: nn.Module,
        *,
        cached_layers: tuple[int, ...] = (4, 11, 17, 23),
        spatial_scales: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
        vggt_token_dim: int = 2048,
        hidden_dim: int = 64,
        geometry_cue_dim: int = 24,
        heads: int = 8,
        decoder_layers: int = 2,
        self_attention_mode: str = "linear",
        cross_attention_mode: str = "deformable",
        deformable_samples: int = 4,
        cross_query_chunk_size: int = 8192,
        query_parameter_chunk_size: int = 32768,
        single_output_size: int = 512,
        merged_output_size: int = 800,
        gradient_checkpointing: bool = True,
        enable_observed: bool = True,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        common = {
            "cached_layers": cached_layers,
            "spatial_scales": spatial_scales,
            "vggt_token_dim": vggt_token_dim,
            "hidden_dim": hidden_dim,
            "geometry_cue_dim": geometry_cue_dim,
            "heads": heads,
            "decoder_layers": decoder_layers,
            "self_attention_mode": self_attention_mode,
            "cross_attention_mode": cross_attention_mode,
            "deformable_samples": deformable_samples,
            "cross_query_chunk_size": cross_query_chunk_size,
            "query_parameter_chunk_size": query_parameter_chunk_size,
            "single_output_size": single_output_size,
            "merged_output_size": merged_output_size,
            "gradient_checkpointing": gradient_checkpointing,
        }
        self.observed_model = (
            Method1Head(
                supervision="observed",
                output_mode="observed_categorical",
                **common,
            )
            if enable_observed
            else None
        )
        self.complete_model = Method1Head(
            supervision="complete",
            output_mode="complete_evidential",
            **common,
        )

    @staticmethod
    def _unwrapped(module: nn.Module) -> Method1Head:
        candidate = getattr(module, "module", module)
        if not isinstance(candidate, Method1Head):
            raise TypeError("Method I path has an unexpected module type")
        return candidate

    def unwrapped_observed_model(self) -> Method1Head:
        if self.observed_model is None:
            raise RuntimeError("the observed Model A path is disabled")
        return self._unwrapped(self.observed_model)

    def unwrapped_complete_model(self) -> Method1Head:
        return self._unwrapped(self.complete_model)

    def extract(
        self,
        images: torch.Tensor,
        camera_height_m: torch.Tensor,
    ) -> dict:
        extraction = self.adapter(images, camera_height_m)
        return {
            key: (
                {layer: value.detach() for layer, value in item.items()}
                if key == "tokens"
                else item.detach()
                if torch.is_tensor(item)
                else item
            )
            for key, item in extraction.items()
        }

    def forward_observed(
        self,
        extraction: dict,
        *,
        geometry_gate: float = 1.0,
    ) -> dict:
        if self.observed_model is None:
            raise RuntimeError("the observed Model A path is disabled")
        prediction = self.observed_model(
            extraction,
            geometry_gate=geometry_gate,
        )
        return _assemble_runtime_observed_output(
            prediction,
            extraction["p1a_geometry"],
        )

    def forward_complete(
        self,
        extraction: dict,
        *,
        geometry_gate: float = 1.0,
    ) -> dict:
        prediction = self.complete_model(
            extraction,
            geometry_gate=geometry_gate,
        )
        return _assemble_runtime_fov_output(
            prediction,
            extraction["p1a_geometry"],
        )

    def diagnostics(self, extraction: dict) -> dict:
        return {
            "estimated_geometry_cue": extraction["geometry_cue"],
            "scene_radius_vggt": extraction["scene_radius_vggt"],
            "estimated_depth_vggt": extraction.get("estimated_depth_vggt"),
            "estimated_intrinsics": extraction.get("estimated_intrinsics"),
            "estimated_camera_from_world_vggt": extraction.get(
                "estimated_camera_from_world_vggt"
            ),
            "geometry_source": extraction["geometry_source"],
            "coordinate_mode": "camera_height_anchored_fixed_normalized_scale",
            "extent_mode": "fixed_6p5_single_10_merged",
            "single_output_size": (
                self.unwrapped_complete_model().single_decoder.output_size
            ),
            "merged_output_size": (
                self.unwrapped_complete_model().merged_decoder.output_size
            ),
            "camera_height_used": True,
            "metric_scale_used": True,
            "runtime_aligned": True,
            "shared_vggt_forward_count": 1,
            "p1a_geometry": extraction["p1a_geometry"],
        }

    def forward(
        self,
        images: torch.Tensor,
        camera_height_m: torch.Tensor,
        *,
        geometry_gate: float = 1.0,
    ) -> dict:
        extraction = self.extract(images, camera_height_m)
        output = {
            "complete": self.forward_complete(
                extraction,
                geometry_gate=geometry_gate,
            ),
            **self.diagnostics(extraction),
        }
        output.update(
            {
                "single_extent_normalized_scale": output["complete"][
                    "single"
                ]["extent_normalized_scale"],
                "merged_extent_normalized_scale": output["complete"][
                    "merged"
                ]["extent_normalized_scale"],
                "single_cell_size_normalized_scale": output["complete"]["single"][
                    "cell_size_normalized_scale"
                ],
                "merged_cell_size_normalized_scale": output["complete"]["merged"][
                    "cell_size_normalized_scale"
                ],
            }
        )
        if self.observed_model is not None:
            output["observed"] = self.forward_observed(
                extraction,
                geometry_gate=geometry_gate,
            )
        return output
