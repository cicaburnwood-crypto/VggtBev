#!/usr/bin/env bash
#SBATCH --job-name=p1b-complete-v5
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/user/VGGT/method1_train_p1b/logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="/home/user/VGGT/method1_train_p1b"
METHOD1_ENV="/home/user/.conda/envs/method1_train_p1b"
CONFIG="${CONFIG:-configs/p1b_remote_gpu_direct_20k.toml}"

source /etc/profile
source /opt/conda/etc/profile.d/conda.sh
conda activate method1_train_p1b

cd "${PROJECT_ROOT}"
export METHOD1_PYTHON="${METHOD1_ENV}/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
"${METHOD1_PYTHON}" -c \
  'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} nccl={torch.cuda.nccl.version()}")'
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader

scripts/torchrun_method1.sh \
  --standalone \
  --nproc-per-node=4 \
  -m vggt_bev_method1.cli_train_metric \
  --config "${CONFIG}"
