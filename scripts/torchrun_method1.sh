#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD1_PYTHON="${METHOD1_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

if [[ ! -x "${METHOD1_PYTHON}" ]]; then
    echo "Method I Python is not executable: ${METHOD1_PYTHON}" >&2
    exit 2
fi

SITE_PACKAGES="$(
    "${METHOD1_PYTHON}" -c \
        'import site; print(next(path for path in site.getsitepackages() if path.endswith("site-packages")))'
)"
NCCL_LIBRARY_DIR="${SITE_PACKAGES}/nvidia/nccl/lib"
NCCL_LIBRARY="${NCCL_LIBRARY_DIR}/libnccl.so.2"

if [[ ! -f "${NCCL_LIBRARY}" ]]; then
    echo "Project-local NCCL is missing: ${NCCL_LIBRARY}" >&2
    echo "Install nvidia-nccl-cu12==2.26.5 into the Method I virtual environment." >&2
    exit 2
fi

export LD_LIBRARY_PATH="${NCCL_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="${NCCL_LIBRARY}${LD_PRELOAD:+:${LD_PRELOAD}}"

NCCL_RUNTIME_VERSION="$(
    "${METHOD1_PYTHON}" -c \
        'import ctypes, sys; library = ctypes.CDLL("libnccl.so.2"); version = ctypes.c_int(); status = library.ncclGetVersion(ctypes.byref(version)); sys.exit(status) if status else print(version.value)'
)"
if (( NCCL_RUNTIME_VERSION < 22605 )); then
    echo "NCCL runtime ${NCCL_RUNTIME_VERSION} is older than required 22605." >&2
    exit 2
fi

echo "Method I project-local NCCL runtime: ${NCCL_RUNTIME_VERSION}" >&2
exec "${METHOD1_PYTHON}" -m torch.distributed.run "$@"
