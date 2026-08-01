#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vggt_root="${VGGT_SOURCE_ROOT:-$(cd "${project_root}/../vggt" && pwd)}"
checkpoint="${VGGT_CHECKPOINT:-${vggt_root}/checkpoints/VGGT-Omega-1B-512/model.pt}"
image_tag="${1:-vggtbev-method1:0.7.0}"
engine="${CONTAINER_ENGINE:-docker}"
engine_arguments=()
if [[ "${engine}" == "podman" && -n "${PODMAN_ROOT:-}" ]]; then
    engine_arguments+=(--root "${PODMAN_ROOT}")
fi
if [[ "${engine}" == "podman" && -n "${PODMAN_RUNROOT:-}" ]]; then
    engine_arguments+=(--runroot "${PODMAN_RUNROOT}")
fi

if [[ ! -d "${vggt_root}/vggt_omega" ]]; then
    echo "VGGT-Omega source is missing: ${vggt_root}/vggt_omega" >&2
    exit 2
fi
if [[ ! -f "${checkpoint}" ]]; then
    echo "VGGT-Omega checkpoint is missing: ${checkpoint}" >&2
    exit 2
fi
context="$(mktemp -d)"
cleanup() {
    if [[ -z "${context}" || ! -d "${context}" || "${context}" != /tmp/* ]]; then
        echo "refusing to remove an unexpected build context: ${context}" >&2
        return
    fi
    rm -rf -- "${context}"
}
trap cleanup EXIT

mkdir -p \
    "${context}/method1" \
    "${context}/testdata" \
    "${context}/vggt_omega"
rsync -a --exclude='__pycache__/' \
    "${vggt_root}/vggt_omega/" "${context}/vggt_omega/"
if ! ln "${checkpoint}" "${context}/model.pt" 2>/dev/null; then
    cp --reflink=auto "${checkpoint}" "${context}/model.pt"
fi
rsync -a --exclude='__pycache__/' \
    "${project_root}/src/" "${context}/method1/src/"
rsync -a \
    "${project_root}/configs/" "${context}/method1/configs/"
mkdir -p "${context}/method1/docker"
cp "${project_root}/docker/entrypoint.sh" "${context}/method1/docker/entrypoint.sh"
cp "${project_root}/docker/Dockerfile" "${context}/Dockerfile"
cp "${project_root}/docker/context.dockerignore" "${context}/.dockerignore"
mapfile -d '' sessions < <(
    find "${project_root}/docker/testdata/data_build_4scenes" \
        -mindepth 2 -maxdepth 2 -type d -name 'session_*' -print0
)
if [[ "${#sessions[@]}" -ne 4 ]]; then
    echo "expected exactly four packaged Habitat sessions, found ${#sessions[@]}" >&2
    exit 2
fi

for session in "${sessions[@]}"; do
    relative="${session#${project_root}/docker/testdata/data_build_4scenes/}"
    destination="${context}/testdata/${relative}"
    mkdir -p "${destination}/bev_6p5m"
    cp "${session}/COMPLETE" "${session}/metadata.json" \
        "${session}/ground_truth_trajectory.jsonl" "${destination}/"
    cp -a "${session}/camera" "${destination}/camera"
    for target in masked complete merged_masked_10m merged_complete_10m; do
        cp -a "${session}/bev_6p5m/${target}" \
            "${destination}/bev_6p5m/${target}"
    done
done

echo "Docker build context: $(du -sh "${context}" | cut -f1)"
"${engine}" "${engine_arguments[@]}" build \
    --file "${context}/Dockerfile" \
    --tag "${image_tag}" \
    "${context}"
