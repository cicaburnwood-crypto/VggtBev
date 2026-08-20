#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/wtbd_merge_scale_template.toml}"
GPU_COUNT="${GPU_COUNT:-1}"

exec torchrun \
  --standalone \
  --nproc-per-node="${GPU_COUNT}" \
  -m vggt_bev_method1.cli_train_wtbd \
  --config "${CONFIG_PATH}"
