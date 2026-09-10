#!/usr/bin/env bash
set -euo pipefail

root=/mnt/data/benyun/Leju-Kuavo5W/vbev
code="$root/m05_plus_train"
python_bin="$root/micromamba/envs/openpi/bin/python"
manifest="$code/manifests/m05_plus_union_pro6000_5090_a100_valid_20260906.json"
session_cache="$code/cache/m05_plus_union_valid_session_records.pkl"
incomplete_index="$code/cache/m05_plus_union_incomplete_sessions_20260907.json"
checkpoint="$code/runs/m05_plus_a100_8gpu_10e_256_frozen_20260906/m05_plus_step_00265000.pt"

for required in \
    "$python_bin" \
    "$manifest" \
    "$session_cache" \
    "$incomplete_index" \
    "$checkpoint" \
    "$code/configs/m05_plus_a100_8gpu_10e_256_frozen.toml"; do
    [[ -e "$required" ]] || {
        printf 'missing required resume input: %s\n' "$required" >&2
        exit 2
    }
done

manifest_sha256=$("$python_bin" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["content_sha256"])' \
    "$manifest")

cd "$code"
export PYTHONPATH="$code/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VGGT_BEV_SESSION_RECORD_CACHE="$session_cache"
export VGGT_BEV_SESSION_RECORD_CACHE_MANIFEST_SHA256="$manifest_sha256"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

printf '%s watchdog_resume checkpoint=%s manifest_sha256=%s\n' \
    "$(date -Is)" "$checkpoint" "$manifest_sha256"

exec "$python_bin" -m torch.distributed.run \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port=29605 \
    -m vggt_bev_method1.cli_train_m05_plus \
    --config configs/m05_plus_a100_8gpu_10e_256_frozen.toml \
    --resume "$checkpoint" \
    --skip-validation
