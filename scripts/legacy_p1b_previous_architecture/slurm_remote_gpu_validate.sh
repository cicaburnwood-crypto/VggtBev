#!/usr/bin/env bash
#SBATCH --job-name=p1b-validate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/VGGT/method1_train_p1b/logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="/home/user/VGGT/method1_train_p1b"
MODE="${MODE:-validate}"
CONFIG="${CONFIG:-configs/p1b_remote_gpu_direct_20k.toml}"

source /etc/profile
source /opt/conda/etc/profile.d/conda.sh
conda activate method1_train_p1b
cd "${PROJECT_ROOT}"

case "${MODE}" in
  freeze)
    python -m vggt_bev_method1.cli_freeze_split \
      --config "${CONFIG}"
    ;;
  validate)
    python -m vggt_bev_method1.cli_train_metric \
      --config "${CONFIG}" --data-only
    ;;
  *)
    echo "MODE must be freeze or validate, got: ${MODE}" >&2
    exit 2
    ;;
esac
