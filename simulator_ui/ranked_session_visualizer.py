#!/usr/bin/env python3
"""Lazy, SSH-tunnelled browser for the ranked 60K P1B session sweep.

Only the sweep CSV is indexed at startup. RGB frames and BEV targets are read
from the remote dataset, and RGB-only model inference is run, only after the
user selects a session. The service binds to loopback by default.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

from p1b_runtime_server_two_expert import build_system, render_confidence, render_semantic

from vggt_bev_method1.config import LabelValues
from vggt_bev_method1.data.fov_targets import (
    cap_complete_and_visible_to_fov,
    fov_union_mask,
)
from vggt_bev_method1.data.preprocess import RGBResizePad
from vggt_bev_method1.p1b_metrics import (
    finalize_p1b_metrics,
    p1b_metric_totals,
)


ROOT = Path(__file__).resolve().parent
HTML_FILE = ROOT / "web_ui/ranked_sessions.html"
DEFAULT_DATA_ROOT = Path("/home/user/data/BEV")
DEFAULT_SWEEP = Path(
    "/home/user/VGGT/method1_train_p1b_nll/"
    "audits/p2b_nll_epoch8_all60k_session_sweep.csv"
)
DEFAULT_SUMMARY = Path(
    "/home/user/VGGT/method1_train_p1b_nll/"
    "runs/p2b_nll_epoch8_runtime_audit_all60k_20260802_1740/summary.json"
)
DEFAULT_BACKBONE_SOURCE = Path(
    "/home/user/VGGT/method1_train_p1b_nll/vendor/backbone"
)
DEFAULT_BACKBONE_CHECKPOINT = Path(
    "/home/user/VGGT/method1_train_p1b_nll/checkpoints/model.pt"
)
DEFAULT_SORT = "guessed_selection_score"
IDENTITY_COLUMNS = {
    "checkpoint",
    "runtime_inputs",
    "sample_id",
    "scene_key",
    "schema",
    "session_key",
    "source_split",
}
BOOLEAN_COLUMNS = {"scale_target_valid", "selected_for_retraining"}
INTEGER_COLUMNS = {
    "checkpoint_epoch",
    "checkpoint_global_step",
    "guessed_rank_within_source_split",
    "history_frame_count",
    "rank",
    "reference_frame_id",
}
LIST_COLUMNS = {
    "history_frame_count",
    "guessed_rank_within_source_split",
    "guessed_selection_score",
    "loss_bev_loss",
    "loss_loss",
    "loss_single_bev_guessed_objective",
    "loss_single_bev_guessed_pixel_loss",
    "loss_single_bev_observed_gate_pixel_bce",
    "loss_single_bev_support_objective",
    "scale_scale_relative_error",
}


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


@dataclass(frozen=True, slots=True)
class SessionEntry:
    session_key: str
    scene_key: str
    source_split: str
    selected_for_retraining: bool
    sample_id: str
    checkpoint: str
    reference_frame_id: int
    history_frame_count: int
    metrics: dict[str, float]

    def searchable(self) -> str:
        return f"{self.session_key} {self.scene_key} {self.source_split}".lower()


class SweepIndex:
    """Memory index over ranking metadata, never over RGB/GT payloads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        entries: list[SessionEntry] = []
        metric_names: set[str] = set()
        seen: set[str] = set()
        checkpoints: set[str] = set()
        with self.path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError("sweep CSV has no header")
            required = {
                "session_key",
                "scene_key",
                "source_split",
                "selected_for_retraining",
                "sample_id",
                "checkpoint",
                "reference_frame_id",
                "history_frame_count",
                DEFAULT_SORT,
            }
            missing = required.difference(reader.fieldnames)
            if missing:
                raise ValueError(f"sweep CSV lacks fields: {sorted(missing)}")
            numeric_columns = [
                name
                for name in reader.fieldnames
                if name not in IDENTITY_COLUMNS and name not in BOOLEAN_COLUMNS
            ]
            for row_number, row in enumerate(reader, start=2):
                key = str(row["session_key"])
                if not key or key in seen:
                    raise ValueError(
                        f"empty or duplicate session key at CSV row {row_number}: {key}"
                    )
                seen.add(key)
                metrics: dict[str, float] = {}
                for name in numeric_columns:
                    raw = str(row.get(name, "")).strip()
                    if not raw:
                        continue
                    try:
                        value = float(raw)
                    except ValueError:
                        continue
                    metrics[name] = value
                    metric_names.add(name)
                checkpoint = str(row["checkpoint"])
                checkpoints.add(str(Path(checkpoint).expanduser().resolve()))
                entries.append(
                    SessionEntry(
                        session_key=key,
                        scene_key=str(row["scene_key"]),
                        source_split=str(row["source_split"]),
                        selected_for_retraining=_parse_bool(
                            str(row["selected_for_retraining"])
                        ),
                        sample_id=str(row["sample_id"]),
                        checkpoint=checkpoint,
                        reference_frame_id=int(row["reference_frame_id"]),
                        history_frame_count=int(row["history_frame_count"]),
                        metrics=metrics,
                    )
                )
        if not entries:
            raise ValueError("sweep CSV is empty")
        if len(checkpoints) != 1:
            raise ValueError("sweep rows do not share one checkpoint")
        self.entries = tuple(entries)
        self.by_key = {entry.session_key: entry for entry in entries}
        self.checkpoint = next(iter(checkpoints))
        self.sortable_fields = tuple(sorted(metric_names))
        self._sort_cache: OrderedDict[tuple[str, str], tuple[int, ...]] = OrderedDict()
        self._sort_lock = threading.Lock()

    def _ordered_indices(self, sort_key: str, direction: str) -> tuple[int, ...]:
        if sort_key not in self.sortable_fields:
            raise ValueError(f"unsupported sort field: {sort_key}")
        if direction not in {"asc", "desc"}:
            raise ValueError("direction must be asc or desc")
        cache_key = (sort_key, direction)
        with self._sort_lock:
            cached = self._sort_cache.get(cache_key)
            if cached is not None:
                self._sort_cache.move_to_end(cache_key)
                return cached
        valid = [
            index
            for index, entry in enumerate(self.entries)
            if math.isfinite(entry.metrics.get(sort_key, math.nan))
        ]
        invalid = [
            index
            for index, entry in enumerate(self.entries)
            if not math.isfinite(entry.metrics.get(sort_key, math.nan))
        ]
        valid.sort(
            key=lambda index: (
                self.entries[index].metrics[sort_key],
                self.entries[index].session_key,
            ),
            reverse=direction == "desc",
        )
        ordered = tuple(valid + invalid)
        with self._sort_lock:
            self._sort_cache[cache_key] = ordered
            self._sort_cache.move_to_end(cache_key)
            while len(self._sort_cache) > 8:
                self._sort_cache.popitem(last=False)
        return ordered

    @staticmethod
    def _matches(entry: SessionEntry, query: str, subset: str) -> bool:
        if query and query not in entry.searchable():
            return False
        if subset == "selected" and not entry.selected_for_retraining:
            return False
        if subset == "unselected" and entry.selected_for_retraining:
            return False
        if subset in {"train", "validation"} and entry.source_split != subset:
            return False
        if subset not in {"all", "selected", "unselected", "train", "validation"}:
            raise ValueError(f"unsupported subset: {subset}")
        return True

    def page(
        self,
        *,
        sort_key: str,
        direction: str,
        page: int,
        page_size: int,
        query: str = "",
        subset: str = "all",
    ) -> dict[str, Any]:
        if page < 1 or not 1 <= page_size <= 200:
            raise ValueError("page must be positive and page_size in [1,200]")
        query = query.strip().lower()
        ordered = self._ordered_indices(sort_key, direction)
        ranked_matches = [
            (global_rank, self.entries[index])
            for global_rank, index in enumerate(ordered, start=1)
            if self._matches(self.entries[index], query, subset)
        ]
        start = (page - 1) * page_size
        stop = start + page_size
        rows = []
        for global_rank, entry in ranked_matches[start:stop]:
            rows.append(
                {
                    "rank": global_rank,
                    "session_key": entry.session_key,
                    "scene_key": entry.scene_key,
                    "source_split": entry.source_split,
                    "selected_for_retraining": entry.selected_for_retraining,
                    "reference_frame_id": entry.reference_frame_id,
                    "history_frame_count": entry.history_frame_count,
                    "sort_value": entry.metrics.get(sort_key),
                    "metrics": {
                        name: entry.metrics.get(name)
                        for name in LIST_COLUMNS
                        if name in entry.metrics
                    },
                }
            )
        return {
            "sort_key": sort_key,
            "direction": direction,
            "page": page,
            "page_size": page_size,
            "total_sessions": len(self.entries),
            "total_filtered": len(ranked_matches),
            "rows": rows,
        }


def _encode_image(image: Image.Image, *, format_name: str, **save_args: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format_name, **save_args)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _encode_png(array: np.ndarray) -> str:
    return _encode_image(Image.fromarray(np.asarray(array, dtype=np.uint8)), format_name="PNG")


def _encode_preview(image: Image.Image) -> str:
    preview = image.convert("RGB").copy()
    preview.thumbnail((480, 360), Image.Resampling.LANCZOS)
    return _encode_image(preview, format_name="JPEG", quality=82, optimize=True)


def _probability_heatmap(probability: torch.Tensor) -> np.ndarray:
    values = probability.detach().float().cpu().numpy().clip(0.0, 1.0)
    low = np.asarray((25.0, 54.0, 104.0), dtype=np.float32)
    high = np.asarray((255.0, 197.0, 61.0), dtype=np.float32)
    return (low + values[..., None] * (high - low)).clip(0, 255).astype(np.uint8)


class LazySessionRuntime:
    def __init__(
        self,
        *,
        index: SweepIndex,
        data_root: str | Path,
        checkpoint: str | Path,
        backbone_source: str | Path,
        backbone_checkpoint: str | Path,
        device: str,
        cache_sessions: int,
        expected_checkpoint_sha256: str | None,
        enforce_sweep_checkpoint: bool = True,
    ) -> None:
        self.index = index
        self.data_root = Path(data_root).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if enforce_sweep_checkpoint and str(self.checkpoint) != self.index.checkpoint:
            raise ValueError(
                "visualizer checkpoint must exactly match the checkpoint used by the sweep"
            )
        digest = hashlib.sha256()
        with self.checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        self.checkpoint_sha256 = digest.hexdigest()
        if (
            enforce_sweep_checkpoint
            and self.checkpoint_sha256 != expected_checkpoint_sha256
        ):
            raise ValueError(
                "visualizer checkpoint SHA-256 does not match the completed sweep"
            )
        self.sweep_checkpoint_compatible = (
            str(self.checkpoint) == self.index.checkpoint
            and self.checkpoint_sha256 == expected_checkpoint_sha256
        )
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.checkpoint_state = torch.load(
            self.checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        self.system = build_system(
            self.checkpoint_state,
            backbone_source=Path(backbone_source).expanduser().resolve(),
            backbone_checkpoint=Path(backbone_checkpoint).expanduser().resolve(),
            device=self.device,
        )
        data = self.checkpoint_state["config"]["data"]
        self.preprocess = RGBResizePad(
            int(data["image_height"]), int(data["image_width"])
        )
        model = self.checkpoint_state["config"]["model"]
        if int(model["single_bev_output_size"]) != 512:
            raise ValueError("ranked visualizer requires the native 512 single output")
        if float(model["single_bev_extent_m"]) != 6.5:
            raise ValueError("ranked visualizer requires the fixed 6.5 m single grid")
        if cache_sessions < 1:
            raise ValueError("cache_sessions must be positive")
        self.cache_sessions = int(cache_sessions)
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "ready": True,
            "sessions": len(self.index.entries),
            "checkpoint": str(self.checkpoint),
            "checkpoint_epoch": int(self.checkpoint_state.get("epoch", -1)),
            "checkpoint_global_step": int(
                self.checkpoint_state.get("global_step", -1)
            ),
            "checkpoint_sha256": self.checkpoint_sha256,
            "sweep_checkpoint_compatible": self.sweep_checkpoint_compatible,
            "runtime_inputs": ["rgb_window"],
            "lazy_session_loading": True,
            "cached_sessions": len(self.cache),
            "device": str(self.device),
            "single_grid": {"size": 512, "extent_m": 6.5},
        }

    def _session_path(self, session_key: str) -> Path:
        if session_key not in self.index.by_key:
            raise KeyError(f"unknown sweep session: {session_key}")
        relative = Path(session_key)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe session key")
        path = (self.data_root / relative).resolve()
        if not path.is_relative_to(self.data_root):
            raise ValueError("session escapes dataset root")
        if not (path / "COMPLETE").is_file():
            raise FileNotFoundError(f"session is not complete: {path}")
        return path

    @staticmethod
    def _read_bev(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            if image.mode != "L" or image.size != (512, 512):
                raise ValueError(f"BEV is not 512x512 grayscale: {path}")
            return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())

    def _load_inputs(
        self, entry: SessionEntry
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict, list[dict]]:
        path = self._session_path(entry.session_key)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        reference = entry.reference_frame_id
        if not 0 <= reference < min(int(metadata["frame_count"]), 10):
            raise ValueError("sweep reference frame violates the max-10 runtime window")
        if entry.history_frame_count != reference + 1:
            raise ValueError("sweep history count and reference frame disagree")
        camera = metadata["camera_intrinsics"]
        source_size = (int(camera["width"]), int(camera["height"]))
        tensors = []
        previews = []
        for frame_id in range(reference + 1):
            frame_path = path / "camera" / f"frame_{frame_id:06d}.png"
            with Image.open(frame_path) as image:
                rgb = image.convert("RGB")
                if rgb.size != source_size:
                    raise ValueError(f"RGB size disagrees with metadata: {frame_path}")
                tensors.append(self.preprocess(rgb))
                previews.append(
                    {
                        "frame_id": frame_id,
                        "jpeg_base64": _encode_preview(rgb),
                    }
                )
        complete = self._read_bev(
            path / "bev_6p5m/complete" / f"frame_{reference:06d}.png"
        )
        visible = self._read_bev(
            path / "bev_6p5m/masked" / f"frame_{reference:06d}.png"
        )
        support = fov_union_mask(
            np.eye(3, dtype=np.float64)[None],
            target_frame=0,
            horizontal_fov_degrees=float(camera["horizontal_fov_degrees"]),
            output_size=512,
            output_extent_m=6.5,
            source_extent_m=6.5,
        )
        complete, visible, support = cap_complete_and_visible_to_fov(
            complete,
            visible,
            support,
            labels=LabelValues(),
        )
        return (
            torch.stack(tensors)[None],
            complete[None],
            visible[None],
            support[None],
            metadata,
            previews,
        )

    def _predict_uncached(self, entry: SessionEntry) -> dict[str, Any]:
        images, complete, visible, support, metadata, previews = self._load_inputs(
            entry
        )
        started = time.monotonic()
        images = images.to(self.device, non_blocking=True)
        use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            enabled=self.device.type == "cuda",
        ):
            extraction = self.system.extract(images)
            prediction = self.system.forward_head(
                extraction,
                enabled_bev_branches=("single",),
                include_scale=True,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_seconds = time.monotonic() - started
        branch = prediction["single_bev"]
        metric_totals = p1b_metric_totals(
            branch,
            complete.to(self.device),
            visible.to(self.device),
            support.to(self.device),
        )
        live_metrics = finalize_p1b_metrics(
            {name: float(value.detach().cpu()) for name, value in metric_totals.items()}
        )
        return {
            "session_key": entry.session_key,
            "scene_key": entry.scene_key,
            "source_split": entry.source_split,
            "selected_for_retraining": entry.selected_for_retraining,
            "sample_id": entry.sample_id,
            "reference_frame_id": entry.reference_frame_id,
            "history_frame_count": entry.history_frame_count,
            "camera": {
                "horizontal_fov_degrees": float(
                    metadata["camera_intrinsics"]["horizontal_fov_degrees"]
                ),
                "camera_height_m": float(
                    metadata["random_parameters"]["camera_height_m"]
                ),
            },
            "grid": {
                "extent_m": 6.5,
                "size": 512,
                "meters_per_pixel": 6.5 / 512,
                "bounds_m": [-3.25, 3.25, -3.25, 3.25],
                "orientation": "latest ego centered; forward is image-up",
            },
            "checkpoint": {
                "epoch": int(self.checkpoint_state.get("epoch", -1)),
                "global_step": int(self.checkpoint_state.get("global_step", -1)),
            },
            "sweep_metrics": entry.metrics,
            "live_metrics": live_metrics,
            "inference_seconds": inference_seconds,
            "lambda_m_per_vggt": float(
                prediction["scale"]["lambda_m_per_vggt"][0].detach().cpu()
            ),
            "rgb_frames": previews,
            "images": {
                "gt_fov_complete_png_base64": _encode_png(complete[0].numpy()),
                "gt_masked_png_base64": _encode_png(visible[0].numpy()),
                "model_semantic_png_base64": _encode_png(
                    render_semantic(branch, 0.5)
                ),
                "model_confidence_png_base64": _encode_png(
                    render_confidence(branch, 0.5)
                ),
                "observed_gate_png_base64": _encode_png(
                    _probability_heatmap(branch["observed_gate_probability"][0])
                ),
                "guessed_occupancy_png_base64": _encode_png(
                    _probability_heatmap(
                        branch["guessed"]["occupancy_probability"][0]
                    )
                ),
            },
        }

    def session(self, session_key: str) -> dict[str, Any]:
        with self.lock:
            cached = self.cache.get(session_key)
            if cached is not None:
                self.cache.move_to_end(session_key)
                return cached
            entry = self.index.by_key.get(session_key)
            if entry is None:
                raise KeyError(f"unknown session: {session_key}")
            result = self._predict_uncached(entry)
            self.cache[session_key] = result
            self.cache.move_to_end(session_key)
            while len(self.cache) > self.cache_sessions:
                self.cache.popitem(last=False)
            return result


def _field_label(name: str) -> str:
    replacements = (
        ("loss_single_bev_", "BEV · "),
        ("loss_", "Loss · "),
        ("scale_", "Scale · "),
        ("guessed_", "Guessed · "),
    )
    label = name
    for prefix, replacement in replacements:
        if label.startswith(prefix):
            label = replacement + label[len(prefix) :]
            break
    return label.replace("_", " ")


def make_handler(
    index: SweepIndex,
    runtime: LazySessionRuntime,
    html: bytes,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "P1BRankedSessions/1.0"

        def _send_json(
            self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(html)
                    return
                if parsed.path == "/health":
                    self._send_json(runtime.health())
                    return
                if parsed.path == "/api/meta":
                    self._send_json(
                        {
                            **runtime.health(),
                            "default_sort": DEFAULT_SORT,
                            "sort_fields": [
                                {"name": name, "label": _field_label(name)}
                                for name in index.sortable_fields
                            ],
                            "subsets": [
                                "all",
                                "selected",
                                "unselected",
                                "train",
                                "validation",
                            ],
                        }
                    )
                    return
                query = parse_qs(parsed.query)
                if parsed.path == "/api/sessions":
                    payload = index.page(
                        sort_key=query.get("sort", [DEFAULT_SORT])[0],
                        direction=query.get("direction", ["asc"])[0],
                        page=int(query.get("page", ["1"])[0]),
                        page_size=int(query.get("page_size", ["50"])[0]),
                        query=query.get("query", [""])[0],
                        subset=query.get("subset", ["all"])[0],
                    )
                    self._send_json(payload)
                    return
                if parsed.path == "/api/session":
                    session_key = query.get("session_key", [""])[0]
                    if not session_key:
                        raise ValueError("session_key is required")
                    self._send_json(runtime.session(session_key))
                    return
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except KeyError as error:
                self._send_json({"error": str(error)}, HTTPStatus.NOT_FOUND)
            except Exception as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format_string: str, *args: object) -> None:
            if "/health" not in str(args[0]):
                super().log_message(format_string, *args)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sweep-csv", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--backbone-source", type=Path, default=DEFAULT_BACKBONE_SOURCE
    )
    parser.add_argument(
        "--backbone-checkpoint", type=Path, default=DEFAULT_BACKBONE_CHECKPOINT
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8892)
    parser.add_argument("--cache-sessions", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not HTML_FILE.is_file():
        raise FileNotFoundError(f"ranked visualizer HTML is missing: {HTML_FILE}")
    summary = json.loads(args.summary.expanduser().resolve().read_text(encoding="utf-8"))
    if not bool(summary.get("complete")) or int(summary.get("session_count", 0)) != 60000:
        raise ValueError("ranked visualizer requires the complete 60K sweep")
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else Path(summary["checkpoint"]).expanduser().resolve()
    )
    index = SweepIndex(args.sweep_csv)
    if len(index.entries) != int(summary["session_count"]):
        raise ValueError("sweep CSV and summary session counts disagree")
    runtime = LazySessionRuntime(
        index=index,
        data_root=args.data_root,
        checkpoint=checkpoint,
        backbone_source=args.backbone_source,
        backbone_checkpoint=args.backbone_checkpoint,
        device=args.device,
        cache_sessions=args.cache_sessions,
        expected_checkpoint_sha256=str(summary["checkpoint_sha256"]),
        # An explicit checkpoint is a deliberate live-model override. Sweep
        # ranks remain historical, while every rendered prediction and live
        # metric comes from the supplied checkpoint. The default path retains
        # strict sweep/checkpoint identity checks.
        enforce_sweep_checkpoint=args.checkpoint is None,
    )
    html = HTML_FILE.read_bytes()
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(index, runtime, html)
    )
    server.daemon_threads = True
    print(
        json.dumps(
            {
                "ready": True,
                "url": f"http://{args.host}:{args.port}",
                "sessions": len(index.entries),
                "checkpoint_epoch": runtime.health()["checkpoint_epoch"],
                "checkpoint_global_step": runtime.health()[
                    "checkpoint_global_step"
                ],
                "lazy_session_loading": True,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
