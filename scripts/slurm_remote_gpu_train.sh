#!/usr/bin/env bash
#SBATCH --job-name=p1a-model-b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/user/VGGT/method1_train/logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="/home/user/VGGT/method1_train"
METHOD1_ENV="/home/user/.conda/envs/method1_train"

source /etc/profile
source /opt/conda/etc/profile.d/conda.sh
conda activate method1_train

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
  -m vggt_bev_method1.cli_train_paired \
  --config configs/p1a_remote_gpu.toml
