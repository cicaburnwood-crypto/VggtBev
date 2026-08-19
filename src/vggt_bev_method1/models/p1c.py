from __future__ import annotations

import math

import torch
from torch import nn

from .attention import MultiheadAttention
from .p1b import P1BHead


def compose_se2_residual(
    pose: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Left-compose an SE(2) residual with [tx,tz,sin(yaw),cos(yaw)]."""

    if pose.shape[-1] != 4 or residual.shape != (*pose.shape[:-1], 3):
        raise ValueError("pose/residual shapes must end in 4 and 3")
    tx, tz, sine, cosine = pose.unbind(dim=-1)
    dx, dz, delta_yaw = residual.unbind(dim=-1)
    delta_sine = delta_yaw.sin()
    delta_cosine = delta_yaw.cos()
    new_tx = delta_cosine * tx - delta_sine * tz + dx
    new_tz = delta_sine * tx + delta_cosine * tz + dz
    new_sine = delta_sine * cosine + delta_cosine * sine
    new_cosine = delta_cosine * cosine - delta_sine * sine
    norm = torch.sqrt(new_sine.square() + new_cosine.square()).clamp_min(1e-6)
    return torch.stack(
        (new_tx, new_tz, new_sine / norm, new_cosine / norm),
        dim=-1,
    )


class CrossFramePoseBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, ffn_ratio: int = 4) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiheadAttention(hidden_dim, heads, mode="exact")
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * ffn_ratio, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        tokens = tokens + self.attention(normalized, normalized)
        return tokens + self.ffn(self.ffn_norm(tokens))


class RelativeSE2PoseHead(nn.Module):
    """Predict latest-from-frame metric SE(2) from VGGT prefix tokens."""

    def __init__(
        self,
        *,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        heads: int = 8,
        layers: int = 4,
        refinements: int = 3,
        maximum_history: int = 10,
        maximum_prefix_tokens: int = 17,
    ) -> None:
        super().__init__()
        if layers <= 0 or refinements <= 0 or maximum_history <= 0:
            raise ValueError("pose-head dimensions must be positive")
        if hidden_dim % heads:
            raise ValueError("pose hidden_dim must be divisible by heads")
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
        self.blocks = nn.ModuleList(
            [CrossFramePoseBlock(hidden_dim, heads) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.refinement_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim + 4, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 3),
                )
                for _ in range(refinements)
            ]
        )
        nn.init.normal_(self.frame_age_embedding, std=0.02)
        nn.init.normal_(self.prefix_type_embedding, std=0.02)
        for head in self.refinement_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, prefix_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if prefix_tokens.ndim != 4:
            raise ValueError("prefix_tokens must have shape [B,N,P,C]")
        batch, frames, prefix_count, _ = prefix_tokens.shape
        if frames > self.maximum_history:
            raise ValueError("frame count exceeds pose-head maximum_history")
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
        tokens = tokens.reshape(batch, frames * prefix_count, -1)
        for block in self.blocks:
            tokens = block(tokens)
        camera_features = self.output_norm(
            tokens.view(batch, frames, prefix_count, -1)[:, :, 0]
        )
        pose = camera_features.new_zeros(batch, frames, 4)
        pose[..., 3] = 1.0
        reference = pose.new_tensor((0.0, 0.0, 0.0, 1.0))
        stages = []
        for head in self.refinement_heads:
            raw_residual = head(torch.cat((camera_features, pose), dim=-1))
            residual = torch.cat(
                (
                    raw_residual[..., :2],
                    math.pi * torch.tanh(raw_residual[..., 2:3]),
                ),
                dim=-1,
            )
            pose = compose_se2_residual(pose, residual)
            pose = pose.clone()
            pose[:, -1] = reference
            stages.append(pose)
        return {
            "relative_pose": pose,
            "refinement_stages": torch.stack(stages, dim=0),
        }


class P1CHead(P1BHead):
    """P1B outputs augmented with explicit RGB-only cross-frame pose reasoning."""

    def __init__(
        self,
        *,
        pose_hidden_dim: int = 256,
        pose_attention_heads: int = 8,
        pose_layers: int = 4,
        pose_refinements: int = 3,
        maximum_history: int = 10,
        pose_conditioning_detach: bool = False,
        **arguments,
    ) -> None:
        super().__init__(**arguments)
        token_dim = int(arguments.get("vggt_token_dim", 2048))
        bev_hidden_dim = int(arguments.get("hidden_dim", 64))
        self.relative_pose_head = RelativeSE2PoseHead(
            input_dim=token_dim,
            hidden_dim=pose_hidden_dim,
            heads=pose_attention_heads,
            layers=pose_layers,
            refinements=pose_refinements,
            maximum_history=maximum_history,
        )
        self.merged_pose_embedding = nn.Sequential(
            nn.Linear(4, bev_hidden_dim),
            nn.GELU(),
            nn.Linear(bev_hidden_dim, bev_hidden_dim),
        )
        self.maximum_history = int(maximum_history)
        self.pose_conditioning_detach = bool(pose_conditioning_detach)

    def _conditioned_merged_pyramids(
        self,
        extraction: dict,
        pose: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        extent = float(self.merged_bev_decoder.extent_m)
        normalized_pose = pose.clone()
        normalized_pose[..., :2] = normalized_pose[..., :2] / extent
        if self.pose_conditioning_detach:
            normalized_pose = normalized_pose.detach()
        frame_embedding = self.merged_pose_embedding(normalized_pose)
        tokens = extraction["tokens"]
        grid = extraction["patch_grid"]
        return (
            self.merged_guessed_token_projector(
                tokens, grid, frame_embedding=frame_embedding
            ),
            self.merged_routing_token_projector(
                tokens, grid, frame_embedding=frame_embedding
            ),
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
        if bev_objective not in (
            "full",
            "fov_support_only",
            "fov_support_and_observed_gate",
            "pose_only",
        ):
            raise ValueError(f"unknown BEV objective: {bev_objective}")
        if "merged" not in enabled_bev_branches:
            raise ValueError("P1C requires the Merged branch")
        if "camera_register_tokens" not in extraction:
            raise KeyError("P1C extraction lacks VGGT camera/register tokens")

        pose_output = self.relative_pose_head(
            extraction["camera_register_tokens"]
        )
        output: dict = {"pose": pose_output}
        if "single" in enabled_bev_branches:
            guessed, routing = self._pyramids(extraction, "single")
            output["single_bev"] = self.single_bev_decoder(
                [level[:, -1:] for level in guessed],
                [level[:, -1:] for level in routing],
                assemble_runtime_outputs=assemble_runtime_outputs,
            )
        if bev_objective != "pose_only":
            guessed, routing = self._conditioned_merged_pyramids(
                extraction,
                pose_output["relative_pose"],
            )
            if bev_objective != "full":
                output["merged_bev"] = (
                    self.merged_bev_decoder.forward_routing_geometry(routing)
                )
            else:
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


class P1CSystem(nn.Module):
    """Frozen VGGT plus a geometry-aware, non-cascaded P1C head."""

    def __init__(self, adapter: nn.Module, **head_arguments) -> None:
        super().__init__()
        self.adapter = adapter
        self.head = P1CHead(**head_arguments)

    def unwrapped_head(self) -> P1CHead:
        module = getattr(self.head, "module", self.head)
        if not isinstance(module, P1CHead):
            raise TypeError("P1C trainable head has an unexpected module type")
        return module

    def extract(self, images: torch.Tensor) -> dict:
        return self.adapter.aggregate(images)

    def forward_head(self, extraction: dict, **arguments) -> dict:
        prediction = self.head(extraction, **arguments)
        head = self.unwrapped_head()
        enabled = tuple(arguments.get("enabled_bev_branches", ("single", "merged")))
        output = {
            **prediction,
            "pipeline_id": "P1C-NLL",
            "probability_model": head.probability_model,
            "coordinate_mode": "p1b_fixed_metric",
            "enabled_bev_branches": enabled,
            "scale_enabled": bool(arguments.get("include_scale", True)),
            "orientation": "latest ego centered; forward is image-up",
            "runtime_inputs": ("rgb_window",),
            "bev_waits_for_geometry_heads": False,
            "geometry_conditioning": "camera-register-relative-se2-embedding",
        }
        for name in enabled:
            if f"{name}_bev" not in prediction:
                continue
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
