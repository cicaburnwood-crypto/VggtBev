#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${1:-"$project_root/configs/p2b_active_local.toml"}
shift || true

exec python -m vggt_bev_method1.cli_train_p2b --config "$config" "$@"
