#!/usr/bin/env bash
#SBATCH --job-name=p2b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

project_root=${P2B_PROJECT_ROOT:?set P2B_PROJECT_ROOT}
environment=${P2B_ENVIRONMENT:-"$project_root/.conda"}
config=${P2B_CONFIG:-"$project_root/configs/p2b_active_remote_gpu.toml"}

mkdir -p "$project_root/logs"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$environment/bin/python" -c 'import pathlib, sys, vggt_bev_method1; expected=(pathlib.Path(sys.argv[1])/"src").resolve(); actual=pathlib.Path(vggt_bev_method1.__file__).resolve(); assert actual.is_relative_to(expected), f"wrong P2B package: {actual}"' "$project_root"
exec "$environment/bin/python" -m vggt_bev_method1.cli_train_p2b \
  --config "$config"
