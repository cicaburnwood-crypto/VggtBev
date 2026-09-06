#!/usr/bin/env bash
set -euo pipefail

root=/mnt/data/benyun/Leju-Kuavo5W/vbev
code="$root/m05_plus_train"
python_bin="$root/micromamba/envs/openpi/bin/python"
manifest="$code/manifests/m05_plus_union_pro6000_5090_a100_valid_20260906.json"
session_cache="$code/cache/m05_plus_union_valid_session_records.pkl"

while [[ ! -f "$session_cache" ]]; do
    if ! pgrep -f 'scripts/build_session_record_cache.py' >/dev/null; then
        echo "session-record cache builder stopped before publishing $session_cache" >&2
        exit 1
    fi
    echo "waiting_for_session_record_cache path=$session_cache"
    sleep 30
done

manifest_sha256="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_sha256"])' "$manifest")"

cd "$code"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$code/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VGGT_BEV_SESSION_RECORD_CACHE="$session_cache"
export VGGT_BEV_SESSION_RECORD_CACHE_MANIFEST_SHA256="$manifest_sha256"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "training_launch manifest=$manifest manifest_sha256=$manifest_sha256"
exec "$python_bin" -m torch.distributed.run \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port=29505 \
    -m vggt_bev_method1.cli_train_m05_plus \
    --config configs/m05_plus_a100_8gpu_10e_256_frozen.toml
