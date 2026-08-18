#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${1:-"$project_root/configs/p1b_active_local.toml"}
shift || true
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

python -c 'import pathlib, sys, vggt_bev_method1; expected=(pathlib.Path(sys.argv[1])/"src").resolve(); actual=pathlib.Path(vggt_bev_method1.__file__).resolve(); assert actual.is_relative_to(expected), f"wrong P1B package: {actual}"' "$project_root"

exec python -m vggt_bev_method1.cli_train_p1b --config "$config" "$@"
