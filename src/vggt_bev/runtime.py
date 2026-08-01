from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vggt_bev.config import LabelValues
from vggt_bev.data import CalibrationAwareResize
from vggt_bev.models import (
    FrozenVGGTAdapter,
    Method2BEVHead,
    Method2System,
    render_observed_bev,
)

MODEL_FILENAMES = {
    "3p5m": "hm3d_600x3_method2_dual_bev_3p5m.pt",
    "5m": "hm3d_600x3_method2_dual_bev_5m.pt",
    "6p5m": "hm3d_600x3_method2_dual_bev_6p5m.pt",
}
EXTENT_KEY_PATTERN = re.compile(r"^bev_(\d+(?:p\d+)?)m$")


@dataclass(frozen=True)
class RuntimeSequence:
    """RGB history for VGGT-predicted geometry runtime inference."""

    images: torch.Tensor
    image_valid: torch.Tensor
    camera_height_m: torch.Tensor | None
    frame_ids: tuple[int, ...]
    session_path: Path
    target_frame_id: int

    def __post_init__(self) -> None:
        frame_count = self.images.shape[0]
        if self.images.ndim != 4 or self.images.shape[1] != 3:
            raise ValueError("images must have shape [N, 3, H, W]")
        if frame_count < 1:
            raise ValueError("runtime sequence must contain at least one frame")
        if self.image_valid.shape != (
            frame_count,
            self.images.shape[-2],
            self.images.shape[-1],
        ):
            raise ValueError("image_valid shape must match the image history")
        if self.camera_height_m is not None:
            if self.camera_height_m.numel() != 1:
                raise ValueError("camera_height_m must be scalar")
            if (
                not torch.isfinite(self.camera_height_m).all()
                or self.camera_height_m <= 0
            ):
                raise ValueError("camera_height_m must be finite and positive")
        if len(self.frame_ids) != frame_count:
            raise ValueError("frame_ids must align with the image history")


@dataclass(frozen=True)
class RuntimePrediction:
    single_labels: torch.Tensor
    merged_labels: torch.Tensor
    geometry_single_labels: torch.Tensor
    geometry_merged_labels: torch.Tensor
    single_occupancy_probability: torch.Tensor
    single_observed_probability: torch.Tensor
    merged_occupancy_probability: torch.Tensor
    merged_observed_probability: torch.Tensor
    depth_scale: float
    single_extent_m: float
    merged_extent_m: float
    output_size: int
    checkpoint_epoch: int
    checkpoint_global_step: int
    geometry_estimated_intrinsic: torch.Tensor
    metric_scale_mode: str = "learned_global"
    coordinate_mode: str = "metric"
    merged_output_size: int | None = None
    reference_scale_vggt: float | None = None
    normalized_units_per_output_pixel: float | None = None


@dataclass(frozen=True)
class RuntimeSinglePrediction:
    labels: torch.Tensor
    occupancy_probability: torch.Tensor
    observed_probability: torch.Tensor
    depth_scale: float
    extent_m: float
    output_size: int
    checkpoint_epoch: int
    checkpoint_global_step: int
    metric_scale_mode: str = "learned_global"
    coordinate_mode: str = "metric"
    reference_scale_vggt: float | None = None
    normalized_units_per_output_pixel: float | None = None


def extent_key_to_meters(extent_key: str) -> float:
    match = EXTENT_KEY_PATTERN.fullmatch(extent_key)
    if match is None:
        raise ValueError(f"invalid extent key: {extent_key!r}")
    return float(match.group(1).replace("p", "."))


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_path(model_key: str, project_root: Path | None = None) -> Path:
    try:
        filename = MODEL_FILENAMES[model_key]
    except KeyError as error:
        raise ValueError(
            f"unknown model key {model_key!r}; choose one of {sorted(MODEL_FILENAMES)}"
        ) from error
    root = project_root or default_project_root()
    return root / "models" / filename


def default_vggt_paths(project_root: Path | None = None) -> tuple[Path, Path]:
    root = project_root or default_project_root()
    candidates = (root.parent / "vggt", root.parent)
    for source in candidates:
        checkpoint = source / "checkpoints" / "VGGT-Omega-1B-512" / "model.pt"
        if (source / "vggt_omega").is_dir() and checkpoint.is_file():
            return source, checkpoint
    source = candidates[0]
    return source, source / "checkpoints" / "VGGT-Omega-1B-512" / "model.pt"


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Method II checkpoint not found: {resolved}")
    state = torch.load(resolved, map_location="cpu", weights_only=False)
    required = {
        "config",
        "epoch",
        "global_step",
        "head",
        "merged_head",
        "log_depth_scale",
    }
    missing = required - state.keys()
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {sorted(missing)}")
    if state["merged_head"] is None:
        raise ValueError("runtime requires a dual-output checkpoint with a merged head")
    if state["config"]["data"].get("target_mode") != "both":
        raise ValueError("checkpoint was not trained with target_mode='both'")
    if (
        state["config"]["model"].get("metric_scale_mode")
        == "vggt_normalized"
        and state.get("normalizer") is None
    ):
        raise ValueError("normalized checkpoint is missing normalizer state")
    return resolved, state


def _build_head(
    model_config: dict[str, Any],
    device: torch.device,
    *,
    output_size: int,
) -> Method2BEVHead:
    return Method2BEVHead(
        feature_dim=int(model_config.get("feature_dim", 2048)),
        hidden_dim=int(model_config.get("hidden_dim", 96)),
        output_size=output_size,
        filter_by_height=bool(model_config.get("filter_by_height", True)),
        ray_steps=int(model_config.get("ray_steps", 32)),
    ).to(device)


class Method2Runtime:
    """Load one trained dual-output head and run label-free Method II inference."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        vggt_source: str | Path | None = None,
        vggt_checkpoint: str | Path | None = None,
        device: str | torch.device = "cuda",
        merged_extent_m: float | None = None,
        shared_adapter: FrozenVGGTAdapter | None = None,
    ) -> None:
        if merged_extent_m is not None and merged_extent_m <= 0:
            raise ValueError("merged_extent_m must be positive")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        self.checkpoint_path, state = _load_checkpoint(checkpoint)
        self.config = state["config"]
        data_config = self.config["data"]
        model_config = self.config["model"]
        self.extent_key = str(data_config["extent_key"])
        self.single_extent_m = extent_key_to_meters(self.extent_key)
        self.merged_extent_m = float(
            merged_extent_m
            if merged_extent_m is not None
            else data_config.get("expected_merged_extent_m", 8.0)
        )
        self.output_size = int(
            model_config.get(
                "single_output_size",
                model_config.get("output_size", 512),
            )
        )
        self.merged_output_size = int(
            model_config.get(
                "merged_output_size",
                model_config.get("output_size", 512),
            )
        )
        self.metric_scale_mode = str(
            model_config.get("metric_scale_mode", "learned_global")
        )
        self.coordinate_mode = str(
            data_config.get("coordinate_mode", "metric")
        )
        self.image_height = int(data_config.get("image_height", 384))
        self.image_width = int(data_config.get("image_width", 512))
        self.checkpoint_epoch = int(state["epoch"])
        self.checkpoint_global_step = int(state["global_step"])

        patch_size = int(model_config.get("patch_size", 16))
        if shared_adapter is None:
            default_source, default_checkpoint = default_vggt_paths()
            source_path = Path(vggt_source or default_source).expanduser().resolve()
            backbone_path = Path(
                vggt_checkpoint or default_checkpoint
            ).expanduser().resolve()
            self.adapter = FrozenVGGTAdapter(
                source_path,
                backbone_path,
                device=self.device,
                patch_size=patch_size,
            )
        else:
            if shared_adapter.patch_size != patch_size:
                raise ValueError("shared VGGT adapter patch size is incompatible")
            self.adapter = shared_adapter
        single_head = _build_head(
            model_config,
            self.device,
            output_size=self.output_size,
        )
        merged_head = _build_head(
            model_config,
            self.device,
            output_size=self.merged_output_size,
        )
        self.model = Method2System(
            self.adapter,
            single_head,
            merged_head=merged_head,
            bev_feature_size=int(
                model_config.get(
                    "single_bev_feature_size",
                    model_config.get("bev_feature_size", 128),
                )
            ),
            merged_bev_feature_size=int(
                model_config.get(
                    "merged_bev_feature_size",
                    model_config.get("bev_feature_size", 128),
                )
            ),
            initial_depth_scale=float(model_config.get("initial_depth_scale", 1.0)),
            learn_depth_scale=bool(model_config.get("learn_depth_scale", True)),
            metric_scale_mode=model_config.get(
                "metric_scale_mode",
                "learned_global",
            ),
            normalizer_hidden_dim=int(
                model_config.get("normalizer_hidden_dim", 96)
            ),
            initial_normalized_single_span=float(
                model_config.get("initial_normalized_single_span", 2.0)
            ),
            stabilize_intrinsics=bool(
                model_config.get("stabilize_intrinsics", False)
            ),
            minimum_confidence=float(
                model_config.get("minimum_confidence", 0.05)
            ),
            minimum_depth_m=float(
                model_config.get("minimum_depth_m", 0.05)
            ),
            maximum_depth_m=float(
                model_config.get("maximum_depth_m", 20.0)
            ),
            ground_minimum_points=int(
                model_config.get("ground_minimum_points", 48)
            ),
            ground_candidate_quantile=float(
                model_config.get("ground_candidate_quantile", 0.55)
            ),
            ground_maximum_candidate_quantile=float(
                model_config.get(
                    "ground_maximum_candidate_quantile",
                    0.995,
                )
            ),
            ground_irls_iterations=int(
                model_config.get("ground_irls_iterations", 5)
            ),
            ground_huber_delta=float(
                model_config.get("ground_huber_delta", 2.5)
            ),
            ground_maximum_tilt_degrees=float(
                model_config.get("ground_maximum_tilt_degrees", 35.0)
            ),
        ).to(self.device)
        self.model.head.load_state_dict(state["head"], strict=True)
        assert self.model.merged_head is not None
        self.model.merged_head.load_state_dict(state["merged_head"], strict=True)
        if self.metric_scale_mode == "vggt_normalized":
            self.model.normalizer.load_state_dict(
                state["normalizer"],
                strict=True,
            )
        self.model.log_depth_scale.data.copy_(
            state["log_depth_scale"].to(self.model.log_depth_scale.device)
        )
        self.model.eval()

    def predict(
        self,
        sequence: RuntimeSequence,
        *,
        threshold: float = 0.5,
    ) -> RuntimePrediction:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between zero and one")
        if sequence.images.shape[-2:] != (self.image_height, self.image_width):
            raise ValueError(
                "runtime image shape does not match training: "
                f"got {tuple(sequence.images.shape[-2:])}, "
                f"expected {(self.image_height, self.image_width)}"
            )
        frame_count = sequence.images.shape[0]
        batch = {
            "images": sequence.images[None].to(self.device, non_blocking=True),
            "image_valid": sequence.image_valid[None].to(self.device, non_blocking=True),
            "frame_valid": torch.ones(
                (1, frame_count), dtype=torch.bool, device=self.device
            ),
            "return_geometry": True,
            "use_predicted_geometry": True,
        }
        if self.metric_scale_mode != "vggt_normalized":
            batch["single_target_extent_m"] = torch.tensor(
                [self.single_extent_m],
                dtype=torch.float32,
                device=self.device,
            )
            batch["merged_target_extent_m"] = torch.tensor(
                [self.merged_extent_m],
                dtype=torch.float32,
                device=self.device,
            )
        if self.metric_scale_mode not in ("vggt_raw", "vggt_normalized"):
            if sequence.camera_height_m is None:
                raise ValueError(
                    "this metric checkpoint requires camera_height_m"
                )
            batch["camera_height_m"] = sequence.camera_height_m.reshape(1).to(
                self.device,
                non_blocking=True,
            )
        with torch.inference_mode():
            raw = self.model(batch)
            single = raw["single"]
            merged = raw["merged"]
            single_labels = render_observed_bev(
                single["occupancy_logit"],
                single["observed_logit"],
                threshold=threshold,
            )
            merged_labels = render_observed_bev(
                merged["occupancy_logit"],
                merged["observed_logit"],
                threshold=threshold,
            )
            geometry = raw["geometry"]
            normalization = raw.get("normalization")

        if normalization is None:
            single_span = self.single_extent_m
            merged_span = self.merged_extent_m
            reference_scale_vggt = None
            normalized_units_per_output_pixel = None
        else:
            single_span = float(
                normalization["single_span_normalized"][0].cpu()
            )
            merged_span = float(
                normalization["merged_span_normalized"][0].cpu()
            )
            reference_scale_vggt = float(
                normalization["reference_scale_vggt"][0].cpu()
            )
            normalized_units_per_output_pixel = float(
                normalization[
                    "normalized_units_per_output_pixel"
                ][0].cpu()
            )

        return RuntimePrediction(
            single_labels=single_labels[0].cpu(),
            merged_labels=merged_labels[0].cpu(),
            geometry_single_labels=geometry["single_labels"][0].cpu(),
            geometry_merged_labels=geometry["merged_labels"][0].cpu(),
            single_occupancy_probability=single["occupancy_logit"][0].sigmoid().cpu(),
            single_observed_probability=single["observed_logit"][0].sigmoid().cpu(),
            merged_occupancy_probability=merged["occupancy_logit"][0].sigmoid().cpu(),
            merged_observed_probability=merged["observed_logit"][0].sigmoid().cpu(),
            depth_scale=float(raw["depth_scale"].detach().cpu()),
            single_extent_m=single_span,
            merged_extent_m=merged_span,
            output_size=self.output_size,
            checkpoint_epoch=self.checkpoint_epoch,
            checkpoint_global_step=self.checkpoint_global_step,
            geometry_estimated_intrinsic=geometry["estimated_intrinsics"][
                0,
                -1,
            ].cpu(),
            metric_scale_mode=self.metric_scale_mode,
            coordinate_mode=self.coordinate_mode,
            merged_output_size=self.merged_output_size,
            reference_scale_vggt=reference_scale_vggt,
            normalized_units_per_output_pixel=(
                normalized_units_per_output_pixel
            ),
        )

    def predict_single(
        self,
        sequence: RuntimeSequence,
        *,
        threshold: float = 0.5,
        vggt_extraction: dict[str, torch.Tensor] | None = None,
    ) -> RuntimeSinglePrediction:
        """Run only this checkpoint's current-frame head.

        ``vggt_extraction`` allows several independently trained heads to reuse
        exactly one frozen VGGT feature/geometry extraction.
        """

        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between zero and one")
        if sequence.images.shape[-2:] != (self.image_height, self.image_width):
            raise ValueError(
                "runtime image shape does not match training: "
                f"got {tuple(sequence.images.shape[-2:])}, "
                f"expected {(self.image_height, self.image_width)}"
            )
        frame_count = sequence.images.shape[0]
        batch = {
            "images": sequence.images[None].to(self.device, non_blocking=True),
            "image_valid": sequence.image_valid[None].to(
                self.device,
                non_blocking=True,
            ),
            "frame_valid": torch.ones(
                (1, frame_count), dtype=torch.bool, device=self.device
            ),
            "return_geometry": False,
            "use_predicted_geometry": True,
            "single_only": True,
        }
        if self.metric_scale_mode != "vggt_normalized":
            batch["single_target_extent_m"] = torch.tensor(
                [self.single_extent_m],
                dtype=torch.float32,
                device=self.device,
            )
            batch["merged_target_extent_m"] = torch.tensor(
                [self.merged_extent_m],
                dtype=torch.float32,
                device=self.device,
            )
        if self.metric_scale_mode not in ("vggt_raw", "vggt_normalized"):
            if sequence.camera_height_m is None:
                raise ValueError(
                    "this metric checkpoint requires camera_height_m"
                )
            batch["camera_height_m"] = sequence.camera_height_m.reshape(1).to(
                self.device,
                non_blocking=True,
            )
        if vggt_extraction is not None:
            batch["vggt_extraction"] = vggt_extraction
        with torch.inference_mode():
            raw = self.model(batch)
            single = raw["single"]
            labels = render_observed_bev(
                single["occupancy_logit"],
                single["observed_logit"],
                threshold=threshold,
            )
            normalization = raw.get("normalization")
        if normalization is None:
            extent = self.single_extent_m
            reference_scale_vggt = None
            normalized_units_per_output_pixel = None
        else:
            extent = float(
                normalization["single_span_normalized"][0].cpu()
            )
            reference_scale_vggt = float(
                normalization["reference_scale_vggt"][0].cpu()
            )
            normalized_units_per_output_pixel = float(
                normalization[
                    "normalized_units_per_output_pixel"
                ][0].cpu()
            )
        return RuntimeSinglePrediction(
            labels=labels[0].cpu(),
            occupancy_probability=single["occupancy_logit"][0].sigmoid().cpu(),
            observed_probability=single["observed_logit"][0].sigmoid().cpu(),
            depth_scale=float(raw["depth_scale"].detach().cpu()),
            extent_m=extent,
            output_size=self.output_size,
            checkpoint_epoch=self.checkpoint_epoch,
            checkpoint_global_step=self.checkpoint_global_step,
            metric_scale_mode=self.metric_scale_mode,
            coordinate_mode=self.coordinate_mode,
            reference_scale_vggt=reference_scale_vggt,
            normalized_units_per_output_pixel=(
                normalized_units_per_output_pixel
            ),
        )


class MultiMethod2Runtime:
    """Three Method II heads sharing one frozen VGGT backbone instance."""

    def __init__(
        self,
        checkpoints: dict[str, str | Path],
        *,
        vggt_source: str | Path | None = None,
        vggt_checkpoint: str | Path | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        if not checkpoints:
            raise ValueError("at least one Method II checkpoint is required")
        self.device = torch.device(device)
        self.runtimes: dict[str, Method2Runtime] = {}
        shared_adapter: FrozenVGGTAdapter | None = None
        for model_key, checkpoint in checkpoints.items():
            runtime = Method2Runtime(
                checkpoint,
                vggt_source=vggt_source,
                vggt_checkpoint=vggt_checkpoint,
                device=self.device,
                shared_adapter=shared_adapter,
            )
            if shared_adapter is None:
                shared_adapter = runtime.adapter
            self.runtimes[model_key] = runtime
        assert shared_adapter is not None
        self.adapter = shared_adapter
        first = next(iter(self.runtimes.values()))
        self.image_height = first.image_height
        self.image_width = first.image_width
        self.output_size = first.output_size
        for runtime in self.runtimes.values():
            if (
                runtime.image_height,
                runtime.image_width,
                runtime.output_size,
            ) != (self.image_height, self.image_width, self.output_size):
                raise ValueError(
                    "all shared-backbone checkpoints must use the same image "
                    "and output dimensions"
                )

    def metadata(self) -> dict[str, dict[str, Any]]:
        return {
            model_key: {
                "extent_key": runtime.extent_key,
                **(
                    {
                        "range_source": "learned_per_sequence",
                        "single_output_size": runtime.output_size,
                    }
                    if runtime.coordinate_mode == "vggt_normalized"
                    else (
                    {
                        "single_extent_vggt_units": (
                            runtime.single_extent_m
                        )
                    }
                    if runtime.coordinate_mode == "vggt_raw"
                    else {"single_extent_m": runtime.single_extent_m}
                    )
                ),
                "output_size": runtime.output_size,
                "checkpoint_epoch": runtime.checkpoint_epoch,
                "checkpoint_global_step": runtime.checkpoint_global_step,
                "coordinate_mode": runtime.coordinate_mode,
            }
            for model_key, runtime in self.runtimes.items()
        }

    def predict(
        self,
        sequence: RuntimeSequence,
        *,
        threshold: float = 0.5,
    ) -> dict[str, RuntimeSinglePrediction]:
        if sequence.images.shape[-2:] != (self.image_height, self.image_width):
            raise ValueError(
                "runtime image shape does not match shared VGGT input: "
                f"got {tuple(sequence.images.shape[-2:])}, "
                f"expected {(self.image_height, self.image_width)}"
            )
        images = sequence.images[None].to(self.device, non_blocking=True)
        with torch.inference_mode():
            extraction = self.adapter(images, include_geometry=True)
            return {
                model_key: runtime.predict_single(
                    sequence,
                    threshold=threshold,
                    vggt_extraction=extraction,
                )
                for model_key, runtime in self.runtimes.items()
            }


def load_vggnav_runtime_sequence(
    session_path: str | Path,
    *,
    target_frame: int = -1,
    image_height: int = 384,
    image_width: int = 512,
    camera_height_m: float | None = None,
) -> RuntimeSequence:
    """Load only camera frames 0..t; do not read GT calibration or trajectory."""

    session = Path(session_path).expanduser().resolve()
    camera_paths = sorted((session / "camera").glob("frame_*.png"))
    if not camera_paths:
        raise FileNotFoundError(f"session contains no camera/frame_*.png files: {session}")
    frame_ids: list[int] = []
    for path in camera_paths:
        match = re.fullmatch(r"frame_(\d+)\.png", path.name)
        if match is None:
            raise ValueError(f"invalid camera frame name: {path.name}")
        frame_ids.append(int(match.group(1)))
    if frame_ids != list(range(len(frame_ids))):
        raise ValueError("runtime requires contiguous camera frame ids starting at zero")
    resolved_target = (
        target_frame if target_frame >= 0 else len(camera_paths) + target_frame
    )
    if not 0 <= resolved_target < len(camera_paths):
        raise IndexError(
            f"target frame {target_frame} is outside a {len(camera_paths)}-frame session"
        )

    preprocess = CalibrationAwareResize(image_height, image_width)
    images: list[torch.Tensor] = []
    image_valid: list[torch.Tensor] = []
    dummy_intrinsic = torch.eye(3, dtype=torch.float32)
    for camera_path in camera_paths[: resolved_target + 1]:
        with Image.open(camera_path) as image:
            image_tensor, _, valid = preprocess(image, dummy_intrinsic)
        images.append(image_tensor)
        image_valid.append(valid)

    return RuntimeSequence(
        images=torch.stack(images),
        image_valid=torch.stack(image_valid),
        camera_height_m=(
            torch.tensor(camera_height_m, dtype=torch.float32)
            if camera_height_m is not None
            else None
        ),
        frame_ids=tuple(frame_ids[: resolved_target + 1]),
        session_path=session,
        target_frame_id=frame_ids[resolved_target],
    )


def save_runtime_prediction(
    prediction: RuntimePrediction,
    output_dir: str | Path,
    *,
    sequence: RuntimeSequence,
    checkpoint_path: str | Path,
    threshold: float,
) -> dict[str, str]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    single_path = output / "single_masked.png"
    merged_path = output / "merged_masked.png"
    geometry_single_path = output / "geometry_single.png"
    geometry_merged_path = output / "geometry_merged.png"
    probabilities_path = output / "probabilities.npz"
    metadata_path = output / "runtime.json"

    Image.fromarray(prediction.single_labels.numpy(), mode="L").save(single_path)
    Image.fromarray(prediction.merged_labels.numpy(), mode="L").save(merged_path)
    Image.fromarray(
        prediction.geometry_single_labels.numpy(),
        mode="L",
    ).save(geometry_single_path)
    Image.fromarray(
        prediction.geometry_merged_labels.numpy(),
        mode="L",
    ).save(geometry_merged_path)
    np.savez_compressed(
        probabilities_path,
        single_occupancy=prediction.single_occupancy_probability.numpy(),
        single_observed=prediction.single_observed_probability.numpy(),
        merged_occupancy=prediction.merged_occupancy_probability.numpy(),
        merged_observed=prediction.merged_observed_probability.numpy(),
    )
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    labels = LabelValues()
    merged_output_size = (
        prediction.merged_output_size or prediction.output_size
    )
    if prediction.coordinate_mode == "vggt_normalized":
        single_extent = {
            "span_normalized_vggt": prediction.single_extent_m,
            "normalized_units_per_pixel": (
                prediction.normalized_units_per_output_pixel
            ),
        }
        merged_extent = {
            "span_normalized_vggt": prediction.merged_extent_m,
            "normalized_units_per_pixel": (
                prediction.normalized_units_per_output_pixel
            ),
        }
    elif prediction.coordinate_mode == "vggt_raw":
        single_extent = {
            "extent_vggt_units": prediction.single_extent_m,
            "vggt_units_per_pixel": (
                prediction.single_extent_m / prediction.output_size
            ),
        }
        merged_extent = {
            "extent_vggt_units": prediction.merged_extent_m,
            "vggt_units_per_pixel": (
                prediction.merged_extent_m / prediction.output_size
            ),
        }
    else:
        single_extent = {
            "extent_m": prediction.single_extent_m,
            "meters_per_pixel": (
                prediction.single_extent_m / prediction.output_size
            ),
        }
        merged_extent = {
            "extent_m": prediction.merged_extent_m,
            "meters_per_pixel": (
                prediction.merged_extent_m / prediction.output_size
            ),
        }
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "checkpoint_epoch": prediction.checkpoint_epoch,
        "checkpoint_global_step": prediction.checkpoint_global_step,
        "session": str(sequence.session_path),
        "target_frame_id": sequence.target_frame_id,
        "source_frame_ids": list(sequence.frame_ids),
        "history_frame_count": len(sequence.frame_ids),
        "threshold": threshold,
        "depth_scale": prediction.depth_scale,
        "reference_scale_vggt": prediction.reference_scale_vggt,
        "metric_scale_mode": prediction.metric_scale_mode,
        "coordinate_mode": prediction.coordinate_mode,
        "runtime_input_contract": {
            "camera_rgb": True,
            "camera_height_m": (
                float(sequence.camera_height_m)
                if sequence.camera_height_m is not None
                else None
            ),
            "external_intrinsics": False,
            "external_camera_poses": False,
            "ground_truth_trajectory": False,
            "geometry_source": "VGGT-estimated intrinsics and relative poses",
        },
        "single": {
            "path": str(single_path),
            "shape": [prediction.output_size, prediction.output_size],
            **single_extent,
        },
        "merged": {
            "path": str(merged_path),
            "shape": [merged_output_size, merged_output_size],
            **merged_extent,
        },
        "geometry_projection": {
            "uses_bev_head": False,
            "camera_source": "VGGT-estimated intrinsics and relative poses",
            "depth_source": (
                "VGGT dense depth with per-window camera-height metric scale"
                if prediction.metric_scale_mode == "camera_height"
                else (
                    "VGGT dense depth in raw VGGT reconstruction scale"
                    if prediction.metric_scale_mode == "vggt_raw"
                    else (
                        "VGGT geometry with a learned per-sequence "
                        "normalized raster range"
                        if prediction.metric_scale_mode == "vggt_normalized"
                        else "VGGT dense depth with learned global metric scale"
                    )
                )
            ),
            "estimated_intrinsic_latest": (
                prediction.geometry_estimated_intrinsic.tolist()
            ),
            "single_path": str(geometry_single_path),
            "merged_path": str(geometry_merged_path),
        },
        "probabilities": str(probabilities_path),
        "label_values": {
            "occupied": labels.occupied,
            "unknown": labels.unknown,
            "free": labels.free,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "single": str(single_path),
        "merged": str(merged_path),
        "geometry_single": str(geometry_single_path),
        "geometry_merged": str(geometry_merged_path),
        "probabilities": str(probabilities_path),
        "metadata": str(metadata_path),
    }
