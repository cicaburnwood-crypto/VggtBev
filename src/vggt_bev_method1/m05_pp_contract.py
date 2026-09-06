from __future__ import annotations

PIPELINE_ID = "M05PP-COARSE-TEMPORAL-HIGHRES-LATEST-CORRECTION-METRIC-NLL"
CHECKPOINT_SCHEMA = "m05pp-coarse256-latest-skip512-fixed-metric-v1"

GEOMETRY_CONDITIONING = "none"
PATCH_FUSION = "dpt_lite_top_down"
PATCH_TOKEN_STREAMS = "local_global_separate_until_fpn_output"
PREFIX_CONDITIONING = "per_cell_role_separated_token_attention"
PREFIX_CONTEXT_TRAINING = "joint_bev_only"
TEMPORAL_EXECUTION = (
    "coarse_latest_anchor_parallel_history_proposals_per_cell_softmax"
)
HISTORY_ORDER = "latest_anchor_then_newest_to_oldest_history"
DATASET_INPUT_ORDER = "oldest_to_latest"
VGGT_INPUT_ORDER = "latest_to_oldest"
BEV_REFERENCE_FRAME = "latest"
EXPECTED_PREFIX_TOKENS = 17
RESOLUTION_HIERARCHY = "coarse256_reasoning_latest_patch_skip_fine512"

