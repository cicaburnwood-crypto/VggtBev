from __future__ import annotations

import json
import os
import pickle
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vggt_bev_method1.config import LabelValues

from .fov_targets import (
    cap_complete_and_visible_to_fov,
    fov_union_mask,
    load_world_from_bev_planar,
    relative_planar_pose_targets,
)
from .preprocess import RGBResizePad
from .void_coverage import FINAL_GT_VOID_FILTER, VoidCoverageIndex


SESSION_RECORD_CACHE_FORMAT = 1
_SESSION_RECORD_CACHE_MEMORY: dict[str, dict] = {}


@dataclass(frozen=True)
class SessionRecord:
    key: str
    path: Path
    dataset: str
    scene_id: str
    frame_count: int
    metadata: dict
    intrinsic: np.ndarray
    source_height: int
    source_width: int
    depth_suffix: str
    source_gt_depth_convention: str
    horizontal_fov_degrees: float
    floor_height_m: float
    world_from_bev_planar: np.ndarray

    @property
    def scene_key(self) -> str:
        return f"{self.dataset}:{self.scene_id}"


@dataclass(frozen=True)
class SampleRecord:
    session_index: int
    target_frame: int


def discover_sessions(root: str | Path) -> list[Path]:
    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {resolved}")
    def committed(marker: Path) -> bool:
        relative_parts = marker.parent.relative_to(resolved).parts
        return (
            marker.is_file()
            and marker.parent.name.startswith("session_")
            and not any(part.endswith(".partial") for part in relative_parts)
        )

    sessions = sorted(
        marker.parent
        for marker in resolved.rglob("COMPLETE")
        if committed(marker)
    )
    if not sessions:
        raise ValueError(f"no committed complete session directories found under {resolved}")
    return sessions


def _depth_suffix(path: Path, metadata: dict) -> str:
    configured = str(metadata.get("depth", {}).get("filename_pattern", "")).lower()
    if configured.endswith(".npz"):
        suffix = ".npz"
    elif configured.endswith(".npy"):
        suffix = ".npy"
    elif (path / "depth/frame_000000.npz").is_file():
        suffix = ".npz"
    elif (path / "depth/frame_000000.npy").is_file():
        suffix = ".npy"
    else:
        raise FileNotFoundError(f"session has no first metric-depth frame: {path}")
    return suffix


def _load_record(root: Path, path: Path) -> SessionRecord:
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError(f"session is not marked complete: {path}")
    if int(metadata.get("schema_version", 0)) < 4:
        raise ValueError(f"new P1B requires schema version 4 or newer: {path}")
    bev = metadata["bev"]
    if [float(value) for value in bev["extent_classes_m"]] != [6.5]:
        raise ValueError(f"session lacks the fixed 6.5 m BEV target: {path}")
    if int(bev["size"]) != 512:
        raise ValueError(f"session BEV raster size is not 512x512: {path}")
    if [float(value) for value in bev.get("merged_normalized_extents_m", [])] != [
        10.0
    ]:
        raise ValueError(f"session lacks the fixed 10 m merged target: {path}")
    if list(bev.get("merged_normalized_size", [])) != [512, 512]:
        raise ValueError(f"source merged target is not 512x512: {path}")
    if bev.get("masked_values") != {"occupied": 0, "unknown": 112, "free": 255}:
        raise ValueError(f"session BEV labels do not match the P1B contract: {path}")
    accepted_orientations = {
        "ego-centric in every output; latest robot centered and forward up",
        "ego-centric; latest robot centered and forward up",
    }
    if bev.get("merged_orientation") not in accepted_orientations:
        raise ValueError(f"session orientation is not latest-ego/forward-up: {path}")

    depth = metadata.get("depth", {})
    if str(depth.get("units", "")).lower() not in ("metres", "meters", "m"):
        raise ValueError(f"GT depth is not metric: {path}")
    depth_source = str(depth.get("source", "")).strip().lower()
    if depth_source == "habitat-sim pinhole depth sensor ground truth":
        source_gt_depth_convention = "camera_axis_z_depth_m"
    elif depth_source == "ai2-thor synchronized third-party metric depth ground truth":
        source_gt_depth_convention = "euclidean_camera_ray_distance_m"
    else:
        raise ValueError(f"GT depth source/convention is unsupported: {path}")

    camera = metadata.get("camera_intrinsics", {})
    intrinsic = np.asarray(camera.get("K"), dtype=np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"camera_intrinsics.K is missing or invalid: {path}")
    source_height = int(camera.get("height", 0))
    source_width = int(camera.get("width", 0))
    if source_height <= 0 or source_width <= 0:
        raise ValueError(f"source RGB/depth size is invalid: {path}")
    horizontal_fov_degrees = float(camera.get("horizontal_fov_degrees", 0.0))
    if horizontal_fov_degrees <= 0.0:
        fx = float(intrinsic[0, 0])
        horizontal_fov_degrees = float(
            np.degrees(2.0 * np.arctan(source_width / (2.0 * fx)))
        )
    if not 0.0 < horizontal_fov_degrees < 180.0:
        raise ValueError(f"horizontal camera FOV is invalid: {path}")
    frame_count = int(metadata["frame_count"])
    if frame_count < 1:
        raise ValueError(f"session frame_count must be positive: {path}")
    extrinsics_path = path / str(
        metadata.get("camera_extrinsics_file", "camera_extrinsics.jsonl")
    )
    if not extrinsics_path.is_file():
        raise FileNotFoundError(
            f"session lacks camera extrinsics required by FOV GT: {path}"
        )
    extrinsic_records = [
        json.loads(line)
        for line in extrinsics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    world_from_bev_planar = load_world_from_bev_planar(
        extrinsic_records,
        expected_frames=frame_count,
    )
    path_start = metadata.get("path", {}).get("start")
    floor_height_m = (
        float(path_start[1])
        if isinstance(path_start, list) and len(path_start) >= 2
        else float("nan")
    )
    return SessionRecord(
        key=path.relative_to(root).as_posix(),
        path=path,
        dataset=str(metadata["dataset"]),
        scene_id=str(metadata["scene_id"]),
        frame_count=frame_count,
        # The complete writer metadata can exceed 100 KiB per session and is
        # not consumed after the validated fields above have been extracted.
        # Retaining it for a full-data run multiplies tens of GiB across DDP
        # ranks without changing a sample, target, or runtime contract.
        metadata={},
        intrinsic=intrinsic,
        source_height=source_height,
        source_width=source_width,
        depth_suffix=_depth_suffix(path, metadata),
        source_gt_depth_convention=source_gt_depth_convention,
        horizontal_fov_degrees=horizontal_fov_degrees,
        floor_height_m=floor_height_m,
        world_from_bev_planar=world_from_bev_planar,
    )


def load_session_records(
    root: str | Path,
    session_keys: Sequence[str] | None = None,
) -> list[SessionRecord]:
    resolved = Path(root).expanduser().resolve()
    cache_text = os.environ.get("VGGT_BEV_SESSION_RECORD_CACHE", "").strip()
    if cache_text:
        cache_path = Path(cache_text).expanduser().resolve()
        cache_key = str(cache_path)
        payload = _SESSION_RECORD_CACHE_MEMORY.get(cache_key)
        if payload is None:
            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"configured session-record cache is missing: {cache_path}"
                )
            with cache_path.open("rb") as stream:
                payload = pickle.load(stream)
            if int(payload.get("format_version", 0)) != SESSION_RECORD_CACHE_FORMAT:
                raise ValueError("session-record cache format is unsupported")
            if Path(payload.get("dataset_root", "")).resolve() != resolved:
                raise ValueError("session-record cache dataset root mismatch")
            expected_manifest = os.environ.get(
                "VGGT_BEV_SESSION_RECORD_CACHE_MANIFEST_SHA256", ""
            ).strip()
            if expected_manifest and payload.get("manifest_content_sha256") != expected_manifest:
                raise ValueError("session-record cache manifest SHA-256 mismatch")
            records_by_key = payload.get("records_by_key")
            if not isinstance(records_by_key, dict) or not records_by_key:
                raise ValueError("session-record cache contains no records")
            _SESSION_RECORD_CACHE_MEMORY[cache_key] = payload
        records_by_key = payload["records_by_key"]
        if session_keys is None:
            return list(records_by_key.values())
        if len(set(session_keys)) != len(session_keys):
            raise ValueError("requested session keys contain duplicates")
        missing = [key for key in session_keys if key not in records_by_key]
        if missing:
            raise KeyError(f"session-record cache lacks manifest key: {missing[0]}")
        return [records_by_key[key] for key in session_keys]
    if session_keys is None:
        paths = discover_sessions(resolved)
    else:
        if len(set(session_keys)) != len(session_keys):
            raise ValueError("requested session keys contain duplicates")
        paths = []
        for key in session_keys:
            relative = Path(key)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"session key is not a safe relative path: {key}")
            if any(part.endswith(".partial") for part in relative.parts):
                raise ValueError(f"session key has a partial ancestor: {key}")
            path = (resolved / relative).resolve()
            if not path.is_relative_to(resolved):
                raise ValueError(f"session key escapes the dataset root: {key}")
            if not (path / "COMPLETE").is_file():
                raise FileNotFoundError(f"manifest session is not complete: {path}")
            paths.append(path)
    return [_load_record(resolved, path) for path in paths]


def split_sessions_by_scene(
    root: str | Path,
    *,
    validation_fraction: float = 0.1,
    seed: int = 17,
) -> tuple[list[str], list[str]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    records = load_session_records(root)
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record.scene_key].append(record.key)
    scene_keys = sorted(grouped)
    if len(scene_keys) < 2:
        raise ValueError("at least two scenes are required for a scene-grouped split")
    random.Random(seed).shuffle(scene_keys)
    validation_scene_count = max(1, round(len(scene_keys) * validation_fraction))
    validation_scene_count = min(validation_scene_count, len(scene_keys) - 1)
    validation_scenes = set(scene_keys[:validation_scene_count])
    train = sorted(
        key
        for scene, session_keys in grouped.items()
        if scene not in validation_scenes
        for key in session_keys
    )
    validation = sorted(
        key
        for scene, session_keys in grouped.items()
        if scene in validation_scenes
        for key in session_keys
    )
    return train, validation


class VGGNAVMethod1Dataset(Dataset[dict]):
    """RGB runtime input plus on-the-fly FOV-complete training targets.

    Collision-truth complete rasters are clipped by an unobstructed camera-FOV
    footprint during ``__getitem__``. Existing visibility-masked rasters split
    valid FOV cells into directly visible and occluded/inferred regions.
    Neither labels nor GT camera geometry enter the runtime model forward.
    """

    labels = LabelValues()
    single_bev_output_size = 512
    single_bev_extent_m = 6.5
    merged_bev_output_size = 800
    merged_bev_extent_m = 10.0

    def __init__(
        self,
        root: str | Path,
        *,
        supervision: str = "metric_fov_complete_evidential",
        preprocess: RGBResizePad | None = None,
        session_keys: Sequence[str] | None = None,
        sample_stride: int = 1,
        minimum_history: int = 1,
        maximum_history: int = 34,
        void_coverage_index: str | Path | None = None,
        expected_manifest_sha256: str | None = None,
        single_bev_extent_m: float = 6.5,
        single_bev_output_size: int = 512,
        merged_source_extent_m: float = 10.0,
        merged_source_image_size: int = 512,
        merged_complete_directory: str = "merged_complete_10m",
        merged_masked_directory: str = "merged_masked_10m",
        merged_bev_extent_m: float = 10.0,
        merged_bev_output_size: int = 800,
        include_single_targets: bool = True,
        include_latest_temporal_targets: bool = False,
    ) -> None:
        if supervision != "metric_fov_complete_evidential":
            raise ValueError(
                "P1B supervision must be metric_fov_complete_evidential"
            )
        if sample_stride <= 0 or minimum_history <= 0 or maximum_history <= 0:
            raise ValueError("history and stride settings must be positive")
        if minimum_history > maximum_history:
            raise ValueError("minimum_history cannot exceed maximum_history")
        if float(single_bev_extent_m) != 6.5 or int(single_bev_output_size) != 512:
            raise ValueError("Single BEV grid must remain 6.5 m at 512x512")
        if merged_source_extent_m <= 0.0:
            raise ValueError("Merged source extent must be positive")
        if merged_source_image_size <= 0:
            raise ValueError("Merged source image size must be positive")
        if not merged_complete_directory or not merged_masked_directory:
            raise ValueError("Merged source directories cannot be empty")
        if Path(merged_complete_directory).is_absolute() or Path(
            merged_masked_directory
        ).is_absolute():
            raise ValueError("Merged source directories must be session-relative")
        if not 0.0 < merged_bev_extent_m <= merged_source_extent_m:
            raise ValueError(
                "Merged output extent must be positive and no larger than its source"
            )
        if merged_bev_output_size <= 0:
            raise ValueError("Merged output size must be positive")
        self.root = Path(root).expanduser().resolve()
        self.supervision = supervision
        self.preprocess = preprocess or RGBResizePad()
        self.single_bev_extent_m = float(single_bev_extent_m)
        self.single_bev_output_size = int(single_bev_output_size)
        self.merged_source_extent_m = float(merged_source_extent_m)
        self.merged_source_image_size = int(merged_source_image_size)
        self.merged_complete_directory = str(merged_complete_directory)
        self.merged_masked_directory = str(merged_masked_directory)
        self.merged_bev_extent_m = float(merged_bev_extent_m)
        self.merged_bev_output_size = int(merged_bev_output_size)
        self.include_single_targets = bool(include_single_targets)
        self.include_latest_temporal_targets = bool(
            include_latest_temporal_targets
        )
        self.void_coverage = (
            VoidCoverageIndex(
                void_coverage_index,
                dataset_root=self.root,
                expected_manifest_sha256=expected_manifest_sha256,
                require_complete=True,
                verify_artifacts=True,
            )
            if void_coverage_index is not None
            else None
        )
        self.sessions = tuple(load_session_records(self.root, session_keys))
        if self.void_coverage is not None:
            missing_scenes = sorted(
                {session.scene_key for session in self.sessions}
                - set(self.void_coverage.scenes)
            )
            if missing_scenes:
                raise ValueError(
                    "Void coverage does not include a training scene: "
                    f"{missing_scenes[0]}"
                )
        self.samples: list[SampleRecord] = []
        for session_index, session in enumerate(self.sessions):
            final_target = min(session.frame_count, maximum_history) - 1
            for target in range(minimum_history - 1, final_target + 1, sample_stride):
                self.samples.append(SampleRecord(session_index, target))
        if not self.samples:
            raise ValueError("dataset settings produced no samples")

    @property
    def scene_keys(self) -> set[str]:
        return {session.scene_key for session in self.sessions}

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_depth(path: Path) -> np.ndarray:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if "depth" not in loaded:
                    raise ValueError(f"compressed depth file lacks 'depth': {path}")
                depth = loaded["depth"]
            finally:
                loaded.close()
        else:
            depth = loaded
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"GT depth must be HxW: {path}")
        return depth

    @staticmethod
    def _camera_ray_distance_to_z_depth(
        depth: np.ndarray,
        intrinsic: np.ndarray,
    ) -> np.ndarray:
        """Convert Euclidean camera-ray range to camera-axis z-depth."""
        height, width = depth.shape
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        x = (np.arange(width, dtype=np.float32) - cx) / fx
        y = (np.arange(height, dtype=np.float32) - cy) / fy
        ray_norm = np.sqrt(1.0 + y[:, None] ** 2 + x[None, :] ** 2)
        return np.asarray(depth / ray_norm, dtype=np.float32)

    @classmethod
    def _load_bev(
        cls,
        path: Path,
        *,
        source_extent_m: float,
        output_extent_m: float,
        output_size: int,
        source_image_size: int = 512,
    ) -> torch.Tensor:
        """Load a metric raster, optionally center-cropping before resampling.

        Source and output grids share the same latest-frame ego origin and
        orientation.  Cropping the 10 m Merged raster to 6.5 m therefore keeps
        only x,z in [-3.25, 3.25] m and never stretches the full 10 m map into
        the smaller metric extent.
        """

        if source_extent_m <= 0.0 or output_extent_m <= 0.0:
            raise ValueError("BEV metric extents must be positive")
        if output_extent_m > source_extent_m:
            raise ValueError("BEV output extent cannot exceed source extent")
        if output_size <= 0:
            raise ValueError("BEV output size must be positive")
        with Image.open(path) as image:
            expected = (source_image_size, source_image_size)
            if image.mode != "L" or image.size != expected:
                raise ValueError(
                    "metric BEV target must be "
                    f"{source_image_size}x{source_image_size} grayscale: {path}"
                )
            if output_extent_m < source_extent_m:
                source_size = image.width
                margin_px = (
                    0.5
                    * (source_extent_m - output_extent_m)
                    * source_size
                    / source_extent_m
                )
                image = image.transform(
                    (output_size, output_size),
                    Image.Transform.EXTENT,
                    (
                        margin_px,
                        margin_px,
                        source_size - margin_px,
                        source_size - margin_px,
                    ),
                    resample=Image.Resampling.NEAREST,
                )
            elif output_size != source_image_size:
                image = image.resize(
                    (output_size, output_size),
                    Image.Resampling.NEAREST,
                )
            labels = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
        values = {int(value) for value in torch.unique(labels)}
        allowed = {cls.labels.occupied, cls.labels.unknown, cls.labels.free}
        if not values <= allowed:
            raise ValueError(f"target contains invalid values {sorted(values - allowed)}")
        return labels

    @classmethod
    def _center_embed_bev(
        cls,
        value: torch.Tensor,
        *,
        source_extent_m: float,
        output_extent_m: float,
        output_size: int,
    ) -> torch.Tensor:
        """Embed a latest-ego raster in a larger latest-ego metric canvas."""

        if value.ndim != 2:
            raise ValueError("BEV embedding expects one HxW label raster")
        if not 0.0 < source_extent_m <= output_extent_m or output_size <= 0:
            raise ValueError("invalid centered BEV embedding contract")
        embedded_size = max(
            1,
            min(
                output_size,
                round(output_size * source_extent_m / output_extent_m),
            ),
        )
        resized = torch.nn.functional.interpolate(
            value[None, None].float(),
            size=(embedded_size, embedded_size),
            mode="nearest",
        )[0, 0].round().to(value.dtype)
        output = torch.full(
            (output_size, output_size),
            cls.labels.unknown,
            dtype=value.dtype,
        )
        top = (output_size - embedded_size) // 2
        left = (output_size - embedded_size) // 2
        output[top : top + embedded_size, left : left + embedded_size] = resized
        return output

    @classmethod
    def _validate_bev_pair(
        cls,
        complete: torch.Tensor,
        observed: torch.Tensor,
        *,
        name: str,
    ) -> None:
        if complete.shape != observed.shape:
            raise ValueError(f"{name} complete/observed shapes do not match")
        complete_valid = complete != cls.labels.unknown
        observed_valid = observed != cls.labels.unknown
        if bool((observed_valid & ~complete_valid).any()):
            raise ValueError(f"{name} observed cells lie outside complete GT")
        if bool((observed_valid & (observed != complete)).any()):
            raise ValueError(f"{name} observed labels disagree with complete GT")

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        session = self.sessions[sample.session_index]
        target_frame = sample.target_frame
        images: list[torch.Tensor] = []
        depths: list[torch.Tensor] = []
        depth_valid: list[torch.Tensor] = []
        for frame in range(target_frame + 1):
            rgb_path = session.path / "camera" / f"frame_{frame:06d}.png"
            with Image.open(rgb_path) as image:
                if (image.height, image.width) != (
                    session.source_height,
                    session.source_width,
                ):
                    raise ValueError(f"RGB dimensions disagree with K: {rgb_path}")
                images.append(self.preprocess(image))
            depth_path = (
                session.path
                / "depth"
                / f"frame_{frame:06d}{session.depth_suffix}"
            )
            source_depth = self._load_depth(depth_path)
            if session.source_gt_depth_convention == "euclidean_camera_ray_distance_m":
                source_depth = self._camera_ray_distance_to_z_depth(
                    source_depth,
                    session.intrinsic,
                )
            depth, valid = self.preprocess.depth(source_depth)
            depths.append(depth)
            depth_valid.append(valid)

        single_observed_path = (
            session.path
            / "bev_6p5m/masked"
            / f"frame_{target_frame:06d}.png"
        )
        single_complete_path = (
            session.path
            / "bev_6p5m/complete"
            / f"frame_{target_frame:06d}.png"
        )
        merged_observed_path = (
            session.path
            / "bev_6p5m"
            / self.merged_masked_directory
            / f"frame_{target_frame:06d}.png"
        )
        merged_complete_path = (
            session.path
            / "bev_6p5m"
            / self.merged_complete_directory
            / f"frame_{target_frame:06d}.png"
        )
        if self.include_single_targets:
            source_single_complete = self._load_bev(
                single_complete_path,
                source_extent_m=self.single_bev_extent_m,
                output_extent_m=self.single_bev_extent_m,
                output_size=self.single_bev_output_size,
            )
            if bool((source_single_complete == self.labels.unknown).any()):
                raise ValueError(
                    "single complete target contains unknown cells: "
                    f"{single_complete_path}"
                )
            source_single_visible = self._load_bev(
                single_observed_path,
                source_extent_m=self.single_bev_extent_m,
                output_extent_m=self.single_bev_extent_m,
                output_size=self.single_bev_output_size,
            )
        if self.include_latest_temporal_targets:
            latest_visible_6p5m = (
                source_single_visible
                if self.include_single_targets
                else self._load_bev(
                    single_observed_path,
                    source_extent_m=self.single_bev_extent_m,
                    output_extent_m=self.single_bev_extent_m,
                    output_size=self.single_bev_output_size,
                )
            )
            source_latest_visible = self._center_embed_bev(
                latest_visible_6p5m,
                source_extent_m=self.single_bev_extent_m,
                output_extent_m=self.merged_bev_extent_m,
                output_size=self.merged_bev_output_size,
            )
        source_merged_complete = self._load_bev(
            merged_complete_path,
            source_extent_m=self.merged_source_extent_m,
            output_extent_m=self.merged_bev_extent_m,
            output_size=self.merged_bev_output_size,
            source_image_size=self.merged_source_image_size,
        )
        source_merged_visible = self._load_bev(
            merged_observed_path,
            source_extent_m=self.merged_source_extent_m,
            output_extent_m=self.merged_bev_extent_m,
            output_size=self.merged_bev_output_size,
            source_image_size=self.merged_source_image_size,
        )
        if self.include_single_targets:
            self._validate_bev_pair(
                source_single_complete,
                source_single_visible,
                name="single",
            )
        self._validate_bev_pair(
            source_merged_complete,
            source_merged_visible,
            name="merged",
        )

        if self.include_single_targets:
            single_fov = fov_union_mask(
                session.world_from_bev_planar[target_frame : target_frame + 1],
                target_frame=0,
                horizontal_fov_degrees=session.horizontal_fov_degrees,
                output_size=self.single_bev_output_size,
                output_extent_m=self.single_bev_extent_m,
                source_extent_m=self.single_bev_extent_m,
            )
        merged_fov = fov_union_mask(
            session.world_from_bev_planar,
            target_frame=target_frame,
            horizontal_fov_degrees=session.horizontal_fov_degrees,
            output_size=self.merged_bev_output_size,
            output_extent_m=self.merged_bev_extent_m,
            source_extent_m=self.single_bev_extent_m,
        )
        if self.include_latest_temporal_targets:
            latest_fov = fov_union_mask(
                session.world_from_bev_planar[target_frame : target_frame + 1],
                target_frame=0,
                horizontal_fov_degrees=session.horizontal_fov_degrees,
                output_size=self.merged_bev_output_size,
                output_extent_m=self.merged_bev_extent_m,
                source_extent_m=self.single_bev_extent_m,
            )
            latest_fov_support = latest_fov & (
                source_merged_complete != self.labels.unknown
            )
            latest_visible_known = latest_fov_support & (
                source_latest_visible != self.labels.unknown
            )
            latest_disagreement = latest_visible_known & (
                source_latest_visible != source_merged_complete
            )
            latest_observed_free = latest_visible_known & (
                source_latest_visible == self.labels.free
            ) & (source_merged_complete == self.labels.free)
        if self.include_single_targets:
            (
                single_fov_complete,
                single_visible,
                single_fov_support,
            ) = cap_complete_and_visible_to_fov(
                source_single_complete,
                source_single_visible,
                single_fov,
                labels=self.labels,
            )
        (
            merged_fov_complete,
            merged_visible,
            merged_fov_support,
        ) = cap_complete_and_visible_to_fov(
            source_merged_complete,
            source_merged_visible,
            merged_fov,
            labels=self.labels,
        )
        if self.include_single_targets:
            self._validate_bev_pair(
                single_fov_complete,
                single_visible,
                name="single FOV-complete",
            )
        self._validate_bev_pair(
            merged_fov_complete,
            merged_visible,
            name="merged FOV-complete",
        )
        if self.void_coverage is None:
            if self.include_single_targets:
                single_gt_valid = torch.ones_like(
                    single_fov_support, dtype=torch.bool
                )
            merged_gt_valid = torch.ones_like(
                merged_fov_support, dtype=torch.bool
            )
            void_index_sha256 = None
        else:
            if not np.isfinite(session.floor_height_m):
                raise ValueError(
                    f"Void filtering requires a valid floor height: {session.path}"
                )
            reference_pose = session.world_from_bev_planar[target_frame]
            # GT validity comes only from the repaired global scene geometry
            # and the output-grid pose. FOV and masked visibility are not inputs.
            if self.include_single_targets:
                single_gt_valid = self.void_coverage.render_valid_mask(
                    scene_key=session.scene_key,
                    floor_height_m=session.floor_height_m,
                    world_from_bev_planar=reference_pose,
                    output_size=self.single_bev_output_size,
                    output_extent_m=self.single_bev_extent_m,
                )
            merged_gt_valid = self.void_coverage.render_valid_mask(
                scene_key=session.scene_key,
                floor_height_m=session.floor_height_m,
                world_from_bev_planar=reference_pose,
                output_size=self.merged_bev_output_size,
                output_extent_m=self.merged_bev_extent_m,
            )
            void_index_sha256 = self.void_coverage.content_sha256
        intrinsic = self.preprocess.intrinsics(
            session.intrinsic,
            source_height=session.source_height,
            source_width=session.source_width,
        )
        frame_ids = list(range(target_frame + 1))
        relative_pose_target = relative_planar_pose_targets(
            session.world_from_bev_planar,
            target_frame=target_frame,
        )
        output = {
            "images": torch.stack(images),
            "relative_pose_target": relative_pose_target,
            "scale_gt_depth_m": torch.stack(depths),
            "scale_gt_valid_mask": torch.stack(depth_valid),
            "scale_gt_intrinsics": intrinsic.unsqueeze(0).expand(
                target_frame + 1, -1, -1
            ).clone(),
            "merged_fov_complete_target": merged_fov_complete,
            "merged_visible_target": merged_visible,
            "merged_fov_support_target": merged_fov_support,
            "merged_gt_valid_mask": merged_gt_valid,
            "metadata": {
                "sample_id": f"{session.key}:frame_{target_frame:06d}",
                "session_key": session.key,
                "session_id": session.path.name,
                "dataset": session.dataset,
                "scene_id": session.scene_id,
                "scene_key": session.scene_key,
                "reference_frame_id": target_frame,
                "source_frame_ids": frame_ids,
                "history_frame_count": target_frame + 1,
                "merged_source_complete_path": str(merged_complete_path),
                "merged_source_visible_path": str(merged_observed_path),
                "fov_target_generation": "on_the_fly_unobstructed_horizontal_frustum_v1",
                "horizontal_fov_degrees": session.horizontal_fov_degrees,
                "gt_depth_convention": "camera_axis_z_depth_m",
                "source_gt_depth_convention": session.source_gt_depth_convention,
                "preprocessing_version": self.preprocess.version,
                "coordinate_mode": "p1b_fixed_metric",
                "merged_bev_extent_m": self.merged_bev_extent_m,
                "merged_bev_output_size": self.merged_bev_output_size,
                "merged_bev_cell_size_m": (
                    self.merged_bev_extent_m / self.merged_bev_output_size
                ),
                "merged_bev_bounds_m": [
                    -self.merged_bev_extent_m / 2.0,
                    self.merged_bev_extent_m / 2.0,
                    -self.merged_bev_extent_m / 2.0,
                    self.merged_bev_extent_m / 2.0,
                ],
                "merged_source_extent_m": self.merged_source_extent_m,
                "merged_source_image_size": self.merged_source_image_size,
                "merged_complete_directory": self.merged_complete_directory,
                "merged_masked_directory": self.merged_masked_directory,
                "merged_target_transform": (
                    "identity"
                    if self.merged_bev_extent_m == self.merged_source_extent_m
                    else "latest_ego_metric_center_crop_then_nearest_resample"
                ),
                "orientation": "latest ego centered; forward is image-up",
                "runtime_model_inputs": ["rgb_window"],
                "bev_content_supervision": (
                    "complete_collision_truth_inside_camera_fov_union"
                ),
                "bev_confidence_supervision": (
                    "visible_masked_vs_fov_complete_occluded_relationship"
                ),
                "outside_fov_semantics": "unknown",
                "gt_void_filter": (
                    FINAL_GT_VOID_FILTER
                    if self.void_coverage is not None
                    else "disabled"
                ),
                "gt_void_scope": "all_bev_losses_independent_of_fov_and_mask",
                "complete_gt_contract": (
                    "semantic_target_plus_independent_gt_valid_mask"
                ),
                "void_training_semantics": (
                    "gt_valid_mask_false_hard_ignores_every_bev_loss"
                ),
                "scene_coverage_algorithm": (
                    self.void_coverage.algorithm
                    if self.void_coverage is not None
                    else "disabled"
                ),
                "void_coverage_index_sha256": void_index_sha256,
            },
        }
        if self.include_latest_temporal_targets:
            output.update(
                {
                    "latest_observed_free_target": latest_observed_free,
                    "latest_fov_support_target": latest_fov_support,
                }
            )
            output["metadata"].update(
                {
                    "latest_temporal_target_source": (
                        "latest 6.5m masked GT centered in Merged metric grid"
                    ),
                    "history_region_contract": (
                        "merged union minus latest-frame support/observed-free"
                    ),
                    "latest_merged_label_disagreement_fraction": float(
                        latest_disagreement.float().mean()
                    ),
                }
            )
        if self.include_single_targets:
            output.update(
                {
                    "single_fov_complete_target": single_fov_complete,
                    "single_visible_target": single_visible,
                    "single_fov_support_target": single_fov_support,
                    "single_gt_valid_mask": single_gt_valid,
                }
            )
            output["metadata"].update(
                {
                    "single_source_complete_path": str(single_complete_path),
                    "single_source_visible_path": str(single_observed_path),
                    "single_bev_extent_m": self.single_bev_extent_m,
                    "single_bev_output_size": self.single_bev_output_size,
                    "single_bev_cell_size_m": (
                        self.single_bev_extent_m / self.single_bev_output_size
                    ),
                    "single_bev_bounds_m": [-3.25, 3.25, -3.25, 3.25],
                }
            )
        return output
