#!/usr/bin/env bash
set -euo pipefail

single_config=/opt/vggtbev/configs/docker_4scene_single.toml
ddp2_config=/opt/vggtbev/configs/docker_4scene_ddp2.toml

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${NVIDIA_VISIBLE_DEVICES:-}" ]]; then
    if [[ "${NVIDIA_VISIBLE_DEVICES}" != "all" && "${NVIDIA_VISIBLE_DEVICES}" != "void" ]]; then
        export CUDA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}"
    fi
fi

case "${1:-validate}" in
    validate)
        exec python -m vggt_bev_method1.cli_validate --config "${single_config}"
        ;;
    inspect)
        exec python -c 'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "cuda_devices": torch.cuda.device_count(), "nccl": torch.cuda.nccl.version() if torch.cuda.is_available() else None}))'
        ;;
    smoke-single)
        shift
        exec python -m vggt_bev_method1.cli_train \
            --config "${single_config}" \
            --smoke-first-sample \
            --max-train-steps 1 \
            --skip-validation \
            "$@"
        ;;
    smoke-ddp2)
        shift
        exec torchrun \
            --standalone \
            --nproc-per-node=2 \
            -m vggt_bev_method1.cli_train \
            --config "${ddp2_config}" \
            --smoke-first-sample \
            --max-train-steps 1 \
            --skip-validation \
            "$@"
        ;;
    shell)
        shift
        exec /bin/bash "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
