from __future__ import annotations

import torch
from torch import nn

from .wtbd_merge_scale import WTBDMergeScaleHead


class FrameReliabilityHead(nn.Module):
    """Learn bounded relative reliability for every RGB-frame token group.

    Reliability is normalized to mean one, so this module can redistribute
    temporal attention but cannot globally amplify or suppress the decoder.
    The final layer starts at zero, making a fresh P1D exactly uniform at
    initialization.  No target, threshold, pose, or hand-authored frame rule is
    used.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        minimum: float = 0.25,
        maximum: float = 1.75,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("frame reliability dimensions must be positive")
        if not 0.0 < minimum < 1.0 < maximum:
            raise ValueError("reliability bounds must straddle one")
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        if prefix_tokens.ndim != 4:
            raise ValueError("prefix tokens must have shape [B,N,P,C]")
        frame_feature = prefix_tokens.mean(dim=2)
        raw = self.network(frame_feature).squeeze(-1)
        reliability = self.minimum + (
            self.maximum - self.minimum
        ) * torch.sigmoid(raw)
        return reliability / reliability.mean(dim=1, keepdim=True).clamp_min(1e-6)


class P1DHead(WTBDMergeScaleHead):
    """Direct RGB-window to Merged BEV, evidence and metric-scale head.

    The frozen VGGT camera/depth outputs are never runtime inputs.  All Merged
    outputs branch in parallel from shared aggregator tokens.  Learned frame
    reliability enters attention weights inside the head; it does not select,
    warp, overwrite, or externally fuse frames.
    """

    pipeline_id = "P1D-DIRECT-MERGED-SCALE-NLL"

    def __init__(
        self,
        *,
        frame_reliability_hidden_dim: int = 256,
        frame_reliability_minimum: float = 0.25,
        frame_reliability_maximum: float = 1.75,
        training_frame_dropout_probability: float = 0.15,
        **arguments,
    ) -> None:
        if not 0.0 <= training_frame_dropout_probability < 1.0:
            raise ValueError("training frame dropout must be in [0,1)")
        token_dim = int(arguments.get("vggt_token_dim", 2048))
        super().__init__(**arguments)
        self.frame_reliability = FrameReliabilityHead(
            token_dim,
            frame_reliability_hidden_dim,
            minimum=frame_reliability_minimum,
            maximum=frame_reliability_maximum,
        )
        self.training_frame_dropout_probability = float(
            training_frame_dropout_probability
        )

    def _frame_weights(
        self,
        prefix_tokens: torch.Tensor,
        *,
        frame_keep_mask: torch.Tensor | None,
        apply_training_frame_dropout: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        learned = self.frame_reliability(prefix_tokens)
        if frame_keep_mask is None:
            keep = torch.ones_like(learned, dtype=torch.bool)
            if (
                apply_training_frame_dropout
                and self.training
                and self.training_frame_dropout_probability > 0.0
                and learned.shape[1] > 1
            ):
                keep[:, :-1] = torch.rand_like(learned[:, :-1]) >= (
                    self.training_frame_dropout_probability
                )
        else:
            if frame_keep_mask.shape != learned.shape:
                raise ValueError("frame_keep_mask must have shape [B,N]")
            keep = frame_keep_mask.bool().clone()
        # The latest/reference RGB is always available.  This also guarantees
        # a non-empty attention memory without a hand-authored choice among
        # historical frames.
        keep[:, -1] = True
        effective = learned * keep.to(learned.dtype)
        kept = keep.sum(dim=1, keepdim=True).to(learned.dtype)
        effective = effective * (
            kept / effective.sum(dim=1, keepdim=True).clamp_min(1e-6)
        )
        return learned, effective, keep

    def forward(
        self,
        extraction: dict,
        *,
        include_merged: bool = True,
        include_scale: bool = True,
        assemble_runtime_outputs: bool = True,
        frame_keep_mask: torch.Tensor | None = None,
        apply_training_frame_dropout: bool = True,
    ) -> dict:
        if "camera_register_tokens" not in extraction:
            raise KeyError("P1D requires frozen aggregator prefix tokens")
        learned, effective, keep = self._frame_weights(
            extraction["camera_register_tokens"],
            frame_keep_mask=frame_keep_mask,
            apply_training_frame_dropout=apply_training_frame_dropout,
        )
        output: dict = {
            "frame_reliability": learned,
            "effective_frame_reliability": effective,
            "frame_keep_mask": keep,
        }
        if include_merged:
            frame_embedding = self.implicit_geometry_trunk(
                extraction["camera_register_tokens"],
                frame_reliability=effective,
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
                frame_reliability=effective,
            )
        if include_scale:
            scale_pyramid = self.scale_token_projector(
                extraction["tokens"], extraction["patch_grid"]
            )
            output["scale"] = self.scale_decoder(
                scale_pyramid,
                frame_reliability=effective,
            )
        return output


class P1DSystem(nn.Module):
    """Frozen VGGT aggregator plus one direct, parallel P1D head."""

    def __init__(self, adapter: nn.Module, **head_arguments) -> None:
        super().__init__()
        self.adapter = adapter
        self.head = P1DHead(**head_arguments)

    def unwrapped_head(self) -> P1DHead:
        head = getattr(self.head, "module", self.head)
        if not isinstance(head, P1DHead):
            raise TypeError("unexpected P1D head type")
        return head

    def extract(self, images: torch.Tensor) -> dict:
        return self.adapter.aggregate(images)

    def decode_scale_teacher(self, extraction: dict) -> dict:
        # Training-label construction only.  This branch is absent at runtime.
        return self.adapter.decode_scale_teacher(extraction)

    def forward_head(self, extraction: dict, **arguments) -> dict:
        prediction = self.head(extraction, **arguments)
        head = self.unwrapped_head()
        return {
            **prediction,
            "pipeline_id": head.pipeline_id,
            "runtime_inputs": ("rgb_window",),
            "runtime_passes": 1,
            "single_bev_present": False,
            "relative_pose_head_present": False,
            "extrinsic_input_present": False,
            "camera_height_input_present": False,
            "external_fusion_present": False,
            "runtime_postprocessing_present": False,
            "merged_source": "frozen VGGT aggregator tokens",
            "coordinate_mode": "vggt_native_units",
            "merged_extent_vggt": head.merged_bev_decoder.extent_vggt,
            "merged_output_size": head.merged_bev_decoder.output_size,
            "scale_unit": "meter_per_vggt_runtime_unit",
            "scale_is_merged_input": False,
            "geometry_conditioning": (
                "implicit_temporal_cross_attention_with_learned_reliability"
            ),
            "maximum_history": head.implicit_geometry_trunk.maximum_history,
            "orientation": "latest ego centered; forward is image-up",
        }

    def forward(self, images: torch.Tensor) -> dict:
        extraction = self.extract(images)
        return self.forward_head(extraction)
