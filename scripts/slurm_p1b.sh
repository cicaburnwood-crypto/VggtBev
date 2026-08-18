#!/usr/bin/env bash
#SBATCH --job-name=p1b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

project_root=${P1B_PROJECT_ROOT:?set P1B_PROJECT_ROOT}
environment=${P1B_ENVIRONMENT:-"$project_root/.conda"}
config=${P1B_CONFIG:-"$project_root/configs/p1b_active_remote_gpu.toml"}

mkdir -p "$project_root/logs"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$environment/bin/python" -c 'import pathlib, sys, vggt_bev_method1; expected=(pathlib.Path(sys.argv[1])/"src").resolve(); actual=pathlib.Path(vggt_bev_method1.__file__).resolve(); assert actual.is_relative_to(expected), f"wrong P1B package: {actual}"' "$project_root"
exec "$environment/bin/python" -m vggt_bev_method1.cli_train_p1b \
  --config "$config"
