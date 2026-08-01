#!/bin/bash

source /etc/profile
set -euo pipefail
conda activate vggtbev-p2

VGGTBEV_ROOT="/home/user/VGGT/method2_train"
VGGTBEV_CONFIG="${VGGTBEV_ROOT}/configs/remote_gpu_p2.toml"
VGGTBEV_DATA_ROOT="/home/user/VGGT/databuilder/output/data_build_unlimited"
VGGTBEV_SOURCE_ROOT="${VGGTBEV_ROOT}/backbone"
VGGTBEV_CHECKPOINT_PATH="${VGGTBEV_ROOT}/checkpoints/VGGT-Omega-1B-512/model.pt"
VGGTBEV_CHECKPOINT_BYTES="4576706117"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1"
    exit 2
  fi
}

require_file "${VGGTBEV_CONFIG}"
require_file "${VGGTBEV_SOURCE_ROOT}/vggt_omega/__init__.py"
require_file "${VGGTBEV_DATA_ROOT}/database_manifest.json"
require_file "${VGGTBEV_CHECKPOINT_PATH}"

if [[ "$(stat -c%s "${VGGTBEV_CHECKPOINT_PATH}")" -ne "${VGGTBEV_CHECKPOINT_BYTES}" ]]; then
  echo "Checkpoint size mismatch: ${VGGTBEV_CHECKPOINT_PATH}"
  exit 2
fi

cd "${VGGTBEV_ROOT}"
env PYTHONPATH="${VGGTBEV_SOURCE_ROOT}" python - <<'PY'
from pathlib import Path
import tomllib

import torch
from vggt_bev.models.vggt_adapter import FrozenVGGTAdapter
from vggt_omega.models import VGGTOmega

config_path = Path("configs/remote_gpu_p2.toml")
config = tomllib.loads(config_path.read_text())
assert config["data"]["coordinate_mode"] == "vggt_normalized"
assert config["model"]["metric_scale_mode"] == "vggt_normalized"
assert config["model"]["single_output_size"] == 512
assert config["model"]["merged_output_size"] == 800
assert Path(config["model"]["checkpoint"]).resolve() == Path(
    "/home/user/VGGT/method2_train/checkpoints/VGGT-Omega-1B-512/model.pt"
).resolve()
assert config["training"]["batch_size_per_gpu"] == 1

print(
    {
        "status": "ready",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vggt_class": VGGTOmega.__name__,
        "adapter": FrozenVGGTAdapter.__name__,
        "config": str(config_path.resolve()),
    }
)
PY
