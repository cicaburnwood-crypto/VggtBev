#!/usr/bin/env bash
set -euo pipefail

project_root="/data/project/vggt/VGGTBEV"
run_tag="${1:?usage: launch_remote_three_dual_extents.sh RUN_TAG}"
mkdir -p "$project_root/logs"

gpus=(2 3 4)
extents=(3p5m 5m 6p5m)

for index in "${!gpus[@]}"; do
    gpu="${gpus[$index]}"
    extent="${extents[$index]}"
    session="vggtbev_m2_dual_${extent}_g${gpu}_${run_tag}"
    log="$project_root/logs/method2_dual_bev_${extent}_gpu${gpu}_${run_tag}.log"
    config="$project_root/configs/remote_hm3d_method2_dual_bev_${extent}.toml"
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "tmux session already exists: $session" >&2
        exit 1
    fi
    if [[ -e "$log" ]]; then
        echo "log already exists: $log" >&2
        exit 1
    fi
    [[ -f "$config" ]] || {
        echo "configuration is missing: $config" >&2
        exit 1
    }
done

for index in "${!gpus[@]}"; do
    gpu="${gpus[$index]}"
    extent="${extents[$index]}"
    session="vggtbev_m2_dual_${extent}_g${gpu}_${run_tag}"
    log="$project_root/logs/method2_dual_bev_${extent}_gpu${gpu}_${run_tag}.log"
    config="$project_root/configs/remote_hm3d_method2_dual_bev_${extent}.toml"
    command="cd $project_root && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu"
    command+=" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4"
    command+=" .venv/bin/vggt-bev-train --config $config --num-workers 2"
    command+=" >>$log 2>&1"
    tmux new-session -d -s "$session" "$command"
    echo "started session=$session physical_gpu=$gpu extent=bev_$extent log=$log"
done
