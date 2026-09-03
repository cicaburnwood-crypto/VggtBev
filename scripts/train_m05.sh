#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/m05_reverse_gated_10m_template.toml}
GPU_COUNT=${GPU_COUNT:-1}
if (($#)); then
  shift
fi

exec torchrun \
  --standalone \
  --nproc-per-node="${GPU_COUNT}" \
  -m vggt_bev_method1.cli_train_m05 \
  --config "${CONFIG}" \
  "$@"
