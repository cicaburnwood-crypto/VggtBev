from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


class TeacherCache:
    """Exact per-window frozen-VGGT cache for optional teacher-pass reuse."""

    schema = "p1b-vggt-teacher-cache-v1"

    def __init__(
        self,
        root: str | Path,
        *,
        checkpoint_sha256: str,
        preprocessing_version: str,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.checkpoint_sha256 = checkpoint_sha256
        self.preprocessing_version = preprocessing_version

    def _identity(self, metadata: dict) -> dict:
        return {
            "schema": self.schema,
            "sample_id": metadata["sample_id"],
            "source_frame_ids": metadata["source_frame_ids"],
            "reference_frame_id": metadata["reference_frame_id"],
            "checkpoint_sha256": self.checkpoint_sha256,
            "preprocessing_version": self.preprocessing_version,
        }

    def path(self, metadata: dict) -> Path:
        identity = self._identity(metadata)
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return self.root / digest[:2] / f"{digest}.pt"

    def load(
        self,
        metadata: list[dict],
        *,
        device: torch.device,
    ) -> tuple[dict, dict] | None:
        records = []
        for item in metadata:
            path = self.path(item)
            if not path.is_file():
                return None
            record = torch.load(path, map_location="cpu", weights_only=False)
            if record.get("identity") != self._identity(item):
                raise ValueError(f"teacher cache identity mismatch: {path}")
            records.append(record)
        layer_keys = set(records[0]["tokens"])
        if any(set(record["tokens"]) != layer_keys for record in records):
            raise ValueError("teacher cache records contain different token layers")
        extraction = {
            "tokens": {
                int(layer): torch.stack(
                    [record["tokens"][int(layer)] for record in records]
                ).to(device=device, dtype=torch.float32)
                for layer in sorted(layer_keys)
            },
            "patch_grid": tuple(records[0]["patch_grid"]),
        }
        if any(tuple(record["patch_grid"]) != extraction["patch_grid"] for record in records):
            raise ValueError("teacher cache records contain different patch grids")
        geometry = {
            key: torch.stack([record[key] for record in records]).to(
                device=device,
                dtype=torch.float32,
            )
            for key in (
                "estimated_depth_vggt",
                "estimated_depth_confidence",
                "estimated_intrinsics",
                "estimated_camera_from_world_vggt",
            )
        }
        geometry["geometry_source"] = "exact frozen-VGGT teacher cache"
        return extraction, geometry

    def save(
        self,
        metadata: list[dict],
        extraction: dict,
        geometry: dict,
    ) -> None:
        for batch_index, item in enumerate(metadata):
            path = self.path(item)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            torch.save(
                {
                    "identity": self._identity(item),
                    "patch_grid": tuple(extraction["patch_grid"]),
                    "tokens": {
                        int(layer): value[batch_index].detach().cpu().to(torch.bfloat16)
                        for layer, value in extraction["tokens"].items()
                    },
                    "estimated_depth_vggt": geometry[
                        "estimated_depth_vggt"
                    ][batch_index].detach().cpu().to(torch.float16),
                    "estimated_depth_confidence": geometry[
                        "estimated_depth_confidence"
                    ][batch_index].detach().cpu().to(torch.float16),
                    "estimated_intrinsics": geometry[
                        "estimated_intrinsics"
                    ][batch_index].detach().cpu().float(),
                    "estimated_camera_from_world_vggt": geometry[
                        "estimated_camera_from_world_vggt"
                    ][batch_index].detach().cpu().float(),
                },
                temporary,
            )
            temporary.replace(path)
