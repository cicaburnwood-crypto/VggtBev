from __future__ import annotations

import torch

from .cli_train_m05 import run_training
from .m05_plus_config import (
    CHECKPOINT_SCHEMA,
    PIPELINE_ID,
    load_m05_plus_config,
)
from .m05_plus_contract import (
    BEV_REFERENCE_FRAME,
    DATASET_INPUT_ORDER,
    EXPECTED_PREFIX_TOKENS,
    GEOMETRY_CONDITIONING,
    HISTORY_ORDER,
    PATCH_FUSION,
    PATCH_TOKEN_STREAMS,
    PREFIX_CONDITIONING,
    PREFIX_CONTEXT_TRAINING,
    TEMPORAL_EXECUTION,
    VGGT_INPUT_ORDER,
)
from .models import LiveVGGTOmegaAdapter, M05PlusSystem


def build_model(config: dict, device: torch.device) -> M05PlusSystem:
    values = config["model"]
    layers = tuple(int(value) for value in values["cached_layers"])
    adapter = LiveVGGTOmegaAdapter(
        values["vggt_source"],
        values["checkpoint"],
        device=device,
        patch_size=int(values["patch_size"]),
        cached_layers=layers,
    )
    return M05PlusSystem(
        adapter,
        cached_layers=layers,
        spatial_scales=tuple(float(value) for value in values["spatial_scales"]),
        vggt_token_dim=int(values["vggt_token_dim"]),
        hidden_dim=int(values["hidden_dim"]),
        query_content_dim=int(values["query_content_dim"]),
        heads=int(values["attention_heads"]),
        latest_decoder_layers=int(values["latest_decoder_layers"]),
        temporal_proposal_layers=int(values["temporal_proposal_layers"]),
        scale_hidden_dim=int(values["scale_hidden_dim"]),
        scale_decoder_layers=int(values["scale_decoder_layers"]),
        self_attention_mode=str(values["self_attention_mode"]),
        deformable_samples=int(values["deformable_samples"]),
        cross_query_chunk_size=int(values["cross_query_chunk_size"]),
        merged_bev_size=int(values["merged_bev_output_size"]),
        merged_extent_m=float(values["merged_bev_extent_m"]),
        query_fourier_bands=int(values["query_fourier_bands"]),
        shared_refinement_layers=int(values["shared_refinement_layers"]),
        routing_refinement_layers=int(values["routing_refinement_layers"]),
        evidence_refinement_layers=int(values["evidence_refinement_layers"]),
        predict_scale_uncertainty=bool(values.get("predict_scale_uncertainty", True)),
        patch_stream_dim=int(values["patch_stream_dim"]),
        prefix_context_hidden_dim=int(values["prefix_context_hidden_dim"]),
        prefix_context_heads=int(values["prefix_context_heads"]),
        prefix_context_layers=int(values["prefix_context_layers"]),
        maximum_history=int(values["maximum_history"]),
        maximum_prefix_tokens=int(values["maximum_prefix_tokens"]),
        frame_reliability_hidden_dim=int(values["frame_reliability_hidden_dim"]),
        frame_reliability_minimum=float(values["frame_reliability_minimum"]),
        frame_reliability_maximum=float(values["frame_reliability_maximum"]),
        temporal_null_initial_probability=float(
            values["temporal_null_initial_probability"]
        ),
        history_proposal_batch_size=int(
            values.get("history_proposal_batch_size", 1)
        ),
    ).to(device)


def build_checkpoint_contract(
    config: dict,
    manifest_sha256: str,
    vggt_sha256: str,
    *,
    pipeline_id: str,
    checkpoint_schema: str,
) -> dict:
    # Import locally to keep the common trainer independent from M05+.
    from .cli_train_m05 import _contract

    contract = _contract(
        config,
        manifest_sha256,
        vggt_sha256,
        pipeline_id=pipeline_id,
        checkpoint_schema=checkpoint_schema,
    )
    model = config["model"]
    training = config["training"]
    contract.update(
        {
            "geometry_conditioning": GEOMETRY_CONDITIONING,
            "temporal_execution": TEMPORAL_EXECUTION,
            "history_order": HISTORY_ORDER,
            "dataset_input_order": DATASET_INPUT_ORDER,
            "vggt_input_order": VGGT_INPUT_ORDER,
            "bev_reference_frame": BEV_REFERENCE_FRAME,
            "strict_zero_history_residual": True,
            "temporal_null_history_count_invariant": True,
            "temporal_null_initial_probability": float(
                model["temporal_null_initial_probability"]
            ),
            "expected_prefix_tokens": EXPECTED_PREFIX_TOKENS,
            "patch_fusion": PATCH_FUSION,
            "patch_token_streams": PATCH_TOKEN_STREAMS,
            "patch_local_input_dim": int(model["vggt_token_dim"]) // 2,
            "patch_global_input_dim": int(model["vggt_token_dim"]) // 2,
            "patch_stream_dim": int(model["patch_stream_dim"]),
            "prefix_conditioning": PREFIX_CONDITIONING,
            "prefix_context_training": PREFIX_CONTEXT_TRAINING,
            "prefix_token_pooling_present": False,
            "prefix_context_hidden_dim": int(
                model["prefix_context_hidden_dim"]
            ),
            "prefix_context_heads": int(model["prefix_context_heads"]),
            "prefix_context_layers": int(model["prefix_context_layers"]),
            "explicit_geometry_module_present": False,
            "geometry_auxiliary_loss_present": False,
            "latest_auxiliary_loss_mode": training[
                "latest_auxiliary_loss_mode"
            ],
            "latest_auxiliary_multiplier": float(
                training["latest_auxiliary_multiplier"]
            ),
            "latest_auxiliary_interval": int(
                training.get("latest_auxiliary_interval", 1)
            ),
            "latest_auxiliary_full_fraction": float(
                training.get("latest_auxiliary_full_fraction", 1.0)
            ),
            "history_cap_by_epoch": tuple(
                int(value)
                for value in training.get("history_cap_by_epoch", ())
            ),
            "history_proposal_batch_size": int(
                model.get("history_proposal_batch_size", 1)
            ),
            "latest_merged_shared_decoder_batching": True,
        }
    )
    return contract


def main() -> None:
    run_training(
        config_loader=load_m05_plus_config,
        model_builder=build_model,
        pipeline_id=PIPELINE_ID,
        checkpoint_schema=CHECKPOINT_SCHEMA,
        checkpoint_prefix="m05_plus",
        contract_builder=build_checkpoint_contract,
    )


if __name__ == "__main__":
    main()
