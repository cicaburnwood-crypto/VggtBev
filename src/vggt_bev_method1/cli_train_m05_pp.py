from __future__ import annotations

import torch

from .cli_train_m05 import run_training
from .cli_train_m05_plus import build_checkpoint_contract as _plus_contract
from .m05_pp_config import CHECKPOINT_SCHEMA, PIPELINE_ID, load_m05_pp_config
from .m05_pp_contract import RESOLUTION_HIERARCHY, TEMPORAL_EXECUTION
from .models import LiveVGGTOmegaAdapter, M05PPSystem


def head_arguments(config: dict) -> dict:
    values = config["model"]
    layers = tuple(int(value) for value in values["cached_layers"])
    return {
        "cached_layers": layers,
        "spatial_scales": tuple(float(value) for value in values["spatial_scales"]),
        "vggt_token_dim": int(values["vggt_token_dim"]),
        "hidden_dim": int(values["hidden_dim"]),
        "query_content_dim": int(values["query_content_dim"]),
        "heads": int(values["attention_heads"]),
        "latest_decoder_layers": int(values["latest_decoder_layers"]),
        "temporal_proposal_layers": int(values["temporal_proposal_layers"]),
        "fine_correction_layers": int(values["fine_correction_layers"]),
        "scale_hidden_dim": int(values["scale_hidden_dim"]),
        "scale_decoder_layers": int(values["scale_decoder_layers"]),
        "self_attention_mode": str(values["self_attention_mode"]),
        "deformable_samples": int(values["deformable_samples"]),
        "cross_query_chunk_size": int(values["cross_query_chunk_size"]),
        "coarse_bev_size": int(values["coarse_bev_size"]),
        "merged_bev_size": int(values["merged_bev_output_size"]),
        "merged_extent_m": float(values["merged_bev_extent_m"]),
        "query_fourier_bands": int(values["query_fourier_bands"]),
        "shared_refinement_layers": int(values["shared_refinement_layers"]),
        "routing_refinement_layers": int(values["routing_refinement_layers"]),
        "evidence_refinement_layers": int(values["evidence_refinement_layers"]),
        "predict_scale_uncertainty": bool(
            values.get("predict_scale_uncertainty", True)
        ),
        "patch_stream_dim": int(values["patch_stream_dim"]),
        "prefix_context_hidden_dim": int(values["prefix_context_hidden_dim"]),
        "prefix_context_heads": int(values["prefix_context_heads"]),
        "prefix_context_layers": int(values["prefix_context_layers"]),
        "maximum_history": int(values["maximum_history"]),
        "maximum_prefix_tokens": int(values["maximum_prefix_tokens"]),
        "frame_reliability_hidden_dim": int(
            values["frame_reliability_hidden_dim"]
        ),
        "frame_reliability_minimum": float(values["frame_reliability_minimum"]),
        "frame_reliability_maximum": float(values["frame_reliability_maximum"]),
        "temporal_null_initial_probability": float(
            values["temporal_null_initial_probability"]
        ),
        "history_proposal_batch_size": int(
            values.get("history_proposal_batch_size", 1)
        ),
    }


def build_model(config: dict, device: torch.device) -> M05PPSystem:
    values = config["model"]
    layers = tuple(int(value) for value in values["cached_layers"])
    adapter = LiveVGGTOmegaAdapter(
        values["vggt_source"],
        values["checkpoint"],
        device=device,
        patch_size=int(values["patch_size"]),
        cached_layers=layers,
    )
    return M05PPSystem(adapter, **head_arguments(config)).to(device)


def build_checkpoint_contract(
    config: dict,
    manifest_sha256: str,
    vggt_sha256: str,
    *,
    pipeline_id: str,
    checkpoint_schema: str,
) -> dict:
    contract = _plus_contract(
        config,
        manifest_sha256,
        vggt_sha256,
        pipeline_id=pipeline_id,
        checkpoint_schema=checkpoint_schema,
    )
    model = config["model"]
    contract.update(
        {
            "resolution_hierarchy": RESOLUTION_HIERARCHY,
            "temporal_execution": TEMPORAL_EXECUTION,
            "coarse_bev_size": int(model["coarse_bev_size"]),
            "heavy_spatial_refinement_size": int(model["coarse_bev_size"]),
            "final_bev_size": int(model["merged_bev_output_size"]),
            "fine_correction_layers": int(model["fine_correction_layers"]),
            "latest_high_resolution_skip_present": True,
            "fine_correction_patch_source": "latest_frame_dpt_pyramid",
            "final_refinement_mode": "depthwise_separable",
            "missing_depth_policy": config["data"].get(
                "missing_depth_policy",
                "error",
            ),
        }
    )
    return contract


def main() -> None:
    run_training(
        config_loader=load_m05_pp_config,
        model_builder=build_model,
        pipeline_id=PIPELINE_ID,
        checkpoint_schema=CHECKPOINT_SCHEMA,
        checkpoint_prefix="m05pp",
        contract_builder=build_checkpoint_contract,
    )


if __name__ == "__main__":
    main()
