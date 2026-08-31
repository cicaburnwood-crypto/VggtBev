from __future__ import annotations

import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Subset

from vggt_bev_method1.data import (
    RGBResizePad,
    VGGNAVMethod1Dataset,
    load_split_manifest,
    manifest_session_keys,
)
from vggt_bev_method1.nccl import configure_and_validate_nccl


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(
    config: dict,
    *,
    verify_manifest: bool = True,
) -> tuple[VGGNAVMethod1Dataset, VGGNAVMethod1Dataset]:
    data = config["data"]
    manifest = load_split_manifest(
        data["split_manifest"],
        dataset_root=data["root"],
        validation_fraction=float(data["validation_fraction"]),
        seed=int(data["split_seed"]),
        maximum_sessions=(
            int(data["maximum_sessions"])
            if data.get("maximum_sessions") is not None
            else None
        ),
        selection_order=data.get("session_selection_order"),
        verify_metadata=(
            verify_manifest
            and bool(data.get("verify_manifest_metadata_at_startup", True))
        ),
        verify_artifacts=(
            verify_manifest
            and bool(data.get("verify_manifest_artifacts_at_startup", True))
        ),
    )
    train_keys, validation_keys = manifest_session_keys(manifest)
    preprocess = RGBResizePad(
        int(data["image_height"]),
        int(data["image_width"]),
    )
    common = {
        "root": data["root"],
        "supervision": data["supervision"],
        "preprocess": preprocess,
        "sample_stride": int(data["sample_stride"]),
        "minimum_history": int(data["minimum_history"]),
        "maximum_history": int(data["maximum_history"]),
        "void_coverage_index": data.get("void_coverage_index"),
        "expected_manifest_sha256": manifest["content_sha256"],
        "single_bev_extent_m": float(data["single_bev_extent_m"]),
        "single_bev_output_size": int(data["single_bev_output_size"]),
        "merged_source_extent_m": float(
            data.get("merged_source_extent_m", 10.0)
        ),
        "merged_bev_extent_m": float(data["merged_bev_extent_m"]),
        "merged_bev_output_size": int(data["merged_bev_output_size"]),
    }
    train = VGGNAVMethod1Dataset(session_keys=train_keys, **common)
    validation = VGGNAVMethod1Dataset(session_keys=validation_keys, **common)
    train.split_manifest_sha256 = manifest["content_sha256"]
    validation.split_manifest_sha256 = manifest["content_sha256"]
    overlap = train.scene_keys.intersection(validation.scene_keys)
    if overlap:
        raise RuntimeError(f"scene leakage detected: {sorted(overlap)[:10]}")
    return train, validation


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def distributed_runtime(
    training: dict,
) -> tuple[bool, int, int, int, torch.device, dict]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    required_devices = int(training.get("required_cuda_devices", 1))
    if world_size > 1:
        if str(training["device"]) != "cuda":
            raise ValueError("distributed P1B requires training.device='cuda'")
        if world_size != required_devices:
            raise ValueError(
                f"configuration requires {required_devices} processes, got {world_size}"
            )
        if torch.cuda.device_count() != world_size:
            raise RuntimeError(
                "torchrun world size must match CUDA_VISIBLE_DEVICES count"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        preflight = configure_and_validate_nccl(
            world_size=world_size,
            require_same_numa=bool(training.get("require_same_numa", True)),
            p2p_level=str(training.get("nccl_p2p_level", "AUTO")),
        )
        preflight.update(
            {
                "cuda_visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES",
                    "",
                ),
                "logical_device": str(device),
                "device_name": torch.cuda.get_device_name(device),
            }
        )
        init_file = os.environ.get("VGGT_BEV_DIST_INIT_FILE", "").strip()
        init_options = {}
        if init_file:
            resolved_init_file = Path(init_file).expanduser().resolve()
            if not resolved_init_file.parent.is_dir():
                raise RuntimeError(
                    "distributed FileStore parent does not exist: "
                    f"{resolved_init_file.parent}"
                )
            init_options = {
                "init_method": resolved_init_file.as_uri(),
                "rank": rank,
                "world_size": world_size,
            }
            preflight["rendezvous"] = "file_store"
            preflight["rendezvous_path"] = str(resolved_init_file)
        else:
            preflight["rendezvous"] = "env_tcp_store"
        dist.init_process_group(
            backend="nccl",
            device_id=device,
            timeout=timedelta(
                seconds=int(training.get("nccl_timeout_seconds", 300))
            ),
            **init_options,
        )
        return True, rank, world_size, local_rank, device, preflight
    if required_devices > 1:
        raise RuntimeError(
            f"this config requires torchrun with {required_devices} processes"
        )
    device = torch.device(str(training["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    preflight = {
        "backend": "none",
        "physical_devices": [],
        "numa_affinities": [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "logical_device": str(device),
    }
    if device.type == "cuda":
        preflight["device_name"] = torch.cuda.get_device_name(device)
    return False, rank, world_size, local_rank, device, preflight


def smoke_subset(
    dataset: VGGNAVMethod1Dataset,
    sample_count: int,
) -> Subset:
    indices = [
        index
        for index, sample in enumerate(dataset.samples)
        if sample.target_frame == 0
    ][:sample_count]
    if len(indices) != sample_count:
        raise ValueError("dataset does not contain enough one-frame smoke samples")
    return Subset(dataset, indices)
