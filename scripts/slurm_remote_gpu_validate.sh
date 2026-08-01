#!/usr/bin/env bash
#SBATCH --job-name=p1a-validate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/VGGT/method1_train/logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="/home/user/VGGT/method1_train"
MODE="${MODE:-validate}"

source /etc/profile
source /opt/conda/etc/profile.d/conda.sh
conda activate method1_train
cd "${PROJECT_ROOT}"

case "${MODE}" in
  freeze)
    python -m vggt_bev_method1.cli_freeze_split \
      --config configs/p1a_remote_gpu.toml
    ;;
  validate)
    python -m vggt_bev_method1.cli_validate \
      --config configs/p1a_remote_gpu.toml
    ;;
  *)
    echo "MODE must be freeze or validate, got: ${MODE}" >&2
    exit 2
    ;;
esac
