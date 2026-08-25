#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/p1d_direct_merged_scale_template.toml}"
GPU_COUNT="${GPU_COUNT:-1}"

exec torchrun \
  --standalone \
  --nproc-per-node="${GPU_COUNT}" \
  -m vggt_bev_method1.cli_train_p1d \
  --config "${CONFIG_PATH}"
