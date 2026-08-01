#!/bin/bash
#SBATCH --job-name=vggtbev-p2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out

source /etc/profile
set -euo pipefail
conda activate vggtbev-p2

VGGTBEV_ROOT="/home/user/VGGT/method2_train"
VGGTBEV_CONFIG="${VGGTBEV_CONFIG:-${VGGTBEV_ROOT}/configs/remote_gpu_p2.toml}"
VGGTBEV_NPROC="${VGGTBEV_NPROC:-1}"
VGGTBEV_DATA_ROOT="/home/user/VGGT/databuilder/output/data_build_unlimited"
VGGTBEV_CHECKPOINT_PATH="${VGGTBEV_ROOT}/checkpoints/VGGT-Omega-1B-512/model.pt"
VGGTBEV_CHECKPOINT_BYTES="4576706117"

if [[ ! -f "${VGGTBEV_CONFIG}" ]]; then
  echo "Config does not exist: ${VGGTBEV_CONFIG}"
  exit 2
fi
if [[ "${VGGTBEV_CONFIG}" == *"template"* ]]; then
  echo "Refusing to train from a template config."
  exit 2
fi
if [[ ! -f "${VGGTBEV_DATA_ROOT}/database_manifest.json" ]]; then
  echo "Dataset manifest is not ready: ${VGGTBEV_DATA_ROOT}/database_manifest.json"
  exit 2
fi
if [[ ! -f "${VGGTBEV_CHECKPOINT_PATH}" ]]; then
  echo "VGGT-Omega checkpoint is not ready: ${VGGTBEV_CHECKPOINT_PATH}"
  exit 2
fi
if [[ "$(stat -c%s "${VGGTBEV_CHECKPOINT_PATH}")" -ne "${VGGTBEV_CHECKPOINT_BYTES}" ]]; then
  echo "VGGT-Omega checkpoint has the wrong byte size."
  exit 2
fi

cd "${VGGTBEV_ROOT}"
mkdir -p logs

if [[ "${VGGTBEV_NPROC}" -eq 1 ]]; then
  python -m vggt_bev.cli_train \
    --config "${VGGTBEV_CONFIG}" \
    --batch-size-per-gpu 1
else
  VGGTBEV_DISTRIBUTED_BACKEND=gloo \
    python -m torch.distributed.run \
      --standalone \
      --nproc_per_node="${VGGTBEV_NPROC}" \
      -m vggt_bev.cli_train \
      --config "${VGGTBEV_CONFIG}" \
      --batch-size-per-gpu 1
fi
