#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${1:-5m}"
if [[ "$#" -gt 0 ]]; then
    shift
fi
case "$model" in
    3p5m|5m|6p5m) ;;
    *)
        echo "model must be one of: 3p5m, 5m, 6p5m" >&2
        exit 2
        ;;
esac

seed="${VGGTBEV_RANDOM_SEED:-$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')}"
if ! [[ "$seed" =~ ^[0-9]+$ ]] || (( seed > 4294967295 )); then
    echo "VGGTBEV_RANDOM_SEED must be an integer from 0 to 4294967295" >&2
    exit 2
fi

scene_root="${VGGTBEV_SCENE_ROOT:-/home/user/Project/VGGNAV/data/scene_datasets/habitat-test-scenes}"
scenes=(
    "$scene_root/skokloster-castle.glb"
    "$scene_root/apartment_1.glb"
)

mapfile -t sampled < <(
    "$project_root/.venv/bin/python" -c \
        'import random,sys; r=random.Random(int(sys.argv[1])); print(r.randrange(2)); print(f"{r.uniform(0.30,0.80):.6f}"); print(f"{r.uniform(60.0,120.0):.6f}")' \
        "$seed"
)
case "${VGGTBEV_SCENE_NAME:-random}" in
    random)
        scene="${scenes[${sampled[0]}]}"
        ;;
    skokloster-castle|apartment_1)
        scene="$scene_root/${VGGTBEV_SCENE_NAME}.glb"
        ;;
    *)
        echo "VGGTBEV_SCENE_NAME must be random, skokloster-castle, or apartment_1" >&2
        exit 2
        ;;
esac
sensor_height_m="${sampled[1]}"
horizontal_fov_degrees="${sampled[2]}"
navmesh="${scene%.glb}.navmesh"

if [[ ! -s "$scene" || ! -s "$navmesh" ]]; then
    echo "random scene or navmesh is unavailable: $scene / $navmesh" >&2
    exit 1
fi

echo "Random VGGTBEV session"
echo "  seed: $seed"
echo "  scene: $scene"
echo "  sensor height: ${sensor_height_m} m"
echo "  horizontal FOV: ${horizontal_fov_degrees} degrees"
echo "  start: random navigable point"
echo "  yaw: random"

VGGTBEV_CAMERA_HEIGHT_M="$sensor_height_m" \
    exec "$project_root/scripts/run_simulator_compare.sh" \
        "$model" \
        "$scene" \
        --navmesh "$navmesh" \
        --sensor-height-m "$sensor_height_m" \
        --horizontal-fov-degrees "$horizontal_fov_degrees" \
        --seed "$seed" \
        --random-start \
        --random-yaw \
        "$@"
