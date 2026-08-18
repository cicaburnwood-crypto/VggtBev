"""Run the accepted Single baseline with GT-targeted realtime A* planning."""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import random
import shutil
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from vggt_bev_method1.cli_train_p1b import build_model
from vggt_bev_method1.data import RGBResizePad
from vggt_bev_method1.navigation import (
    GTPlanningTrial,
    PredictedPlan,
    TargetSelectionError,
    plan_on_prediction,
    select_gt_target,
)


@dataclass(frozen=True)
class FrameCandidate:
    key: tuple[str, str, str, int]
    session: Path
    rgb: Path
    complete_gt: Path
    masked_gt: Path | None
    fov_degrees: float


@dataclass(frozen=True)
class FrozenTrial:
    candidate: FrameCandidate
    planning: GTPlanningTrial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RGB-only Single-BEV inference with targets sampled only "
            "from GT-free, GT-reachable cells inside the true 6.5 m FOV"
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backbone-source", required=True, type=Path)
    parser.add_argument("--backbone-checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--planning-size", type=int, default=512)
    parser.add_argument("--extent-m", type=float, default=6.5)
    parser.add_argument("--planning-inflation-radius-m", type=float, default=0.10)
    parser.add_argument("--gt-clearance-radius-m", type=float, default=0.05)
    parser.add_argument(
        "--known-ego-pose-clearance-radius-m", type=float, default=0.10
    )
    parser.add_argument("--confidence-cost-weight", type=float, default=1.0)
    parser.add_argument("--minimum-target-distance-m", type=float, default=0.75)
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--support-threshold", type=float, default=0.5)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_index(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid frame filename: {path.name}") from error


def discover_candidates(roots: list[Path], *, extent_m: float) -> list[FrameCandidate]:
    """Find unique 512px RGB/GT pairs, preferring the first root's copy."""

    candidates: dict[tuple[str, str, str, int], FrameCandidate] = {}
    for unresolved_root in roots:
        root = unresolved_root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"data root does not exist: {root}")
        for metadata_path in root.rglob("metadata.json"):
            session = metadata_path.parent
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("status") != "complete":
                continue
            validation = metadata.get("validation")
            if isinstance(validation, dict) and validation.get("passed") is False:
                continue
            bev = metadata.get("bev", {})
            if int(bev.get("size", 0)) != 512:
                continue
            extents = [float(value) for value in bev.get("extent_classes_m", ())]
            if not any(abs(value - extent_m) < 1e-6 for value in extents):
                continue
            parameters = metadata.get("random_parameters", {})
            try:
                fov = float(parameters["horizontal_fov_degrees"])
            except (KeyError, TypeError, ValueError):
                continue
            dataset = str(metadata.get("dataset", ""))
            scene_id = str(metadata.get("scene_id", ""))
            session_id = str(metadata.get("session_id", session.name))
            for rgb in sorted((session / "camera").glob("frame_*.png")):
                complete = session / "bev_6p5m" / "complete" / rgb.name
                if not complete.is_file():
                    continue
                try:
                    with Image.open(complete) as image:
                        if image.size != (512, 512):
                            continue
                except OSError:
                    continue
                masked = session / "bev_6p5m" / "masked" / rgb.name
                frame = _frame_index(rgb)
                key = (dataset, scene_id, session_id, frame)
                candidates.setdefault(
                    key,
                    FrameCandidate(
                        key=key,
                        session=session,
                        rgb=rgb,
                        complete_gt=complete,
                        masked_gt=masked if masked.is_file() else None,
                        fov_degrees=fov,
                    ),
                )
    return sorted(candidates.values(), key=lambda item: item.key)


def freeze_gt_trials(
    candidates: list[FrameCandidate],
    *,
    count: int,
    seed: int,
    planning_size: int,
    extent_m: float,
    gt_clearance_radius_m: float,
    minimum_target_distance_m: float,
) -> tuple[list[FrozenTrial], list[dict[str, str]]]:
    """Freeze the entire evaluation set before any model prediction exists."""

    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    trials: list[FrozenTrial] = []
    rejected: list[dict[str, str]] = []
    for candidate in shuffled:
        with Image.open(candidate.complete_gt) as image:
            complete = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        try:
            planning = select_gt_target(
                complete,
                horizontal_fov_degrees=candidate.fov_degrees,
                rng=rng,
                planning_size=planning_size,
                extent_m=extent_m,
                robot_radius_m=gt_clearance_radius_m,
                minimum_target_distance_m=minimum_target_distance_m,
            )
        except TargetSelectionError as error:
            rejected.append({"frame": str(candidate.rgb), "reason": str(error)})
            continue
        trials.append(FrozenTrial(candidate=candidate, planning=planning))
        if len(trials) == count:
            return trials, rejected
    raise RuntimeError(
        f"only {len(trials)} GT-valid unique frames are available; requested {count}"
    )


def _trial_manifest(
    trials: list[FrozenTrial],
    rejected: list[dict[str, str]],
    *,
    gt_clearance_radius_m: float,
) -> dict:
    return {
        "target_selection_source": "GT complete occupancy + true geometric FOV only",
        "prediction_used_for_target_sampling": False,
        "gt_clearance_radius_m": gt_clearance_radius_m,
        "sample_count": len(trials),
        "rejected_before_inference": rejected,
        "samples": [
            {
                "index": index,
                "key": list(trial.candidate.key),
                "rgb": str(trial.candidate.rgb),
                "complete_gt": str(trial.candidate.complete_gt),
                "horizontal_fov_degrees": trial.candidate.fov_degrees,
                "goal_grid_row_column": list(trial.planning.goal),
                "goal_native_pixel_row_column": list(
                    trial.planning.goal_native_pixel
                ),
                "goal_x_z_m": [trial.planning.goal_x_m, trial.planning.goal_z_m],
                "gt_astar_path_length_m": trial.planning.gt_path_length_m,
            }
            for index, trial in enumerate(trials)
        ],
    }


def _load_model(
    checkpoint: Path,
    backbone_source: Path,
    backbone_checkpoint: Path,
    device: torch.device,
):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("single_output") != [512, 512, 6.5]:
        raise ValueError("checkpoint is not the accepted 512x512 / 6.5m Single model")
    config = copy.deepcopy(state["config"])
    config["model"]["vggt_source"] = str(backbone_source)
    config["model"]["checkpoint"] = str(backbone_checkpoint)
    model = build_model(config, device)
    model.unwrapped_head().load_state_dict(state["head"], strict=True)
    return model.eval(), state, config


def _grid_image(blocked: np.ndarray, fov: np.ndarray | None = None) -> Image.Image:
    output = np.full((*blocked.shape, 3), 255, dtype=np.uint8)
    output[blocked] = (0, 0, 0)
    if fov is not None:
        output[~fov] = (135, 135, 135)
    return Image.fromarray(output, mode="RGB").resize((512, 512), Image.Resampling.NEAREST)


def _overlay_plan(
    image: Image.Image,
    trial: GTPlanningTrial,
    path: tuple[tuple[int, int], ...],
    *,
    success: bool,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    scale = output.width / trial.gt_blocked.shape[0]

    def point(cell: tuple[int, int]) -> tuple[float, float]:
        return ((cell[1] + 0.5) * scale, (cell[0] + 0.5) * scale)

    if path:
        draw.line([point(cell) for cell in path], fill=(255, 145, 0), width=4)
    start = point(trial.start)
    goal = point(trial.goal)
    radius = 7
    draw.ellipse(
        (start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius),
        fill=(0, 170, 255),
        outline=(255, 255, 255),
        width=2,
    )
    goal_color = (30, 210, 80) if success else (255, 45, 45)
    draw.ellipse(
        (goal[0] - radius, goal[1] - radius, goal[0] + radius, goal[1] + radius),
        fill=goal_color,
        outline=(255, 255, 255),
        width=2,
    )
    return output


def _predicted_semantic(
    occupancy: np.ndarray,
    support: np.ndarray,
    *,
    occupancy_threshold: float,
    support_threshold: float,
) -> Image.Image:
    output = np.full((*occupancy.shape, 3), 135, dtype=np.uint8)
    known = support >= support_threshold
    output[known & (occupancy < occupancy_threshold)] = (255, 255, 255)
    output[known & (occupancy >= occupancy_threshold)] = (0, 0, 0)
    return Image.fromarray(output, mode="RGB")


def _save_visuals(
    output: Path,
    trial: FrozenTrial,
    prediction: PredictedPlan,
    occupancy: np.ndarray,
    support: np.ndarray,
    *,
    occupancy_threshold: float,
    support_threshold: float,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(trial.candidate.rgb, output / "rgb.png")
    gt = _overlay_plan(
        _grid_image(trial.planning.gt_blocked, trial.planning.geometric_fov),
        trial.planning,
        trial.planning.gt_path,
        success=True,
    )
    gt.save(output / "gt_target_and_path.png")
    predicted = _overlay_plan(
        _predicted_semantic(
            occupancy,
            support,
            occupancy_threshold=occupancy_threshold,
            support_threshold=support_threshold,
        ),
        trial.planning,
        prediction.path,
        success=prediction.success,
    )
    predicted.save(output / "predicted_astar.png")
    with Image.open(trial.candidate.rgb) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((512, 512), Image.Resampling.LANCZOS)
        rgb_panel = Image.new("RGB", (512, 512), (25, 25, 25))
        rgb_panel.paste(rgb, ((512 - rgb.width) // 2, (512 - rgb.height) // 2))
    composite = Image.new("RGB", (1536, 512), (25, 25, 25))
    composite.paste(rgb_panel, (0, 0))
    composite.paste(gt, (512, 0))
    composite.paste(predicted, (1024, 0))
    composite.save(output / "composite.jpg", quality=90)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _html(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    cards = []
    for record in records:
        status = "PASS" if record["success"] else html.escape(record["failure_reason"])
        cards.append(
            f"""<article class={'pass' if record['success'] else 'fail'}>
<h3>#{record['index']:03d} · {status}</h3>
<img loading=lazy src="{record['directory']}/composite.jpg">
<p>inference {record['inference_ms']:.1f} ms · A* {record['astar_ms']:.2f} ms · "
f"goal occ {record['goal_occupancy_probability']:.3f} · support "
f"{record['goal_support_probability']:.3f} · navigation confidence "
f"{record['goal_navigation_confidence']:.3f}</p>
<p>{html.escape(record['dataset'])} / {html.escape(record['scene_id'])} / "
f"frame {record['frame_index']}</p></article>"""
        )
    payload = html.escape(json.dumps(summary, indent=2))
    return f"""<!doctype html><meta charset=utf-8>
<title>Single baseline GT-target A*</title>
<style>
body{{margin:0;background:#10141a;color:#e9eef5;font:14px system-ui}}
header{{position:sticky;top:0;background:#151b24;padding:16px;z-index:2}}
main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:12px;padding:12px}}
article{{background:#1b222d;border:1px solid #344052;border-radius:10px;padding:10px}}
article.pass{{border-color:#24794a}} article.fail{{border-color:#9d3941}}
img{{width:100%;height:auto;background:#222}} h1,h3,p{{margin:5px 0}} pre{{white-space:pre-wrap}}
</style><header><h1>Single baseline · GT-only targets · realtime A*</h1>
<details><summary>summary.json</summary><pre>{payload}</pre></details></header>
<main>{''.join(cards)}</main>"""


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.warmup_iterations < 0:
        raise ValueError("samples must be positive and warmup non-negative")
    if abs(args.extent_m - 6.5) > 1e-6:
        raise ValueError("the accepted Single baseline has a fixed 6.5m extent")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    candidates = discover_candidates(args.data_root, extent_m=args.extent_m)
    trials, rejected = freeze_gt_trials(
        candidates,
        count=args.samples,
        seed=args.seed,
        planning_size=args.planning_size,
        extent_m=args.extent_m,
        gt_clearance_radius_m=args.gt_clearance_radius_m,
        minimum_target_distance_m=args.minimum_target_distance_m,
    )
    manifest = _trial_manifest(
        trials,
        rejected,
        gt_clearance_radius_m=args.gt_clearance_radius_m,
    )
    (output / "gt_frozen_trials.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this local baseline run requires an available CUDA GPU")
    checkpoint = args.checkpoint.expanduser().resolve()
    backbone_source = args.backbone_source.expanduser().resolve()
    backbone_checkpoint = args.backbone_checkpoint.expanduser().resolve()
    for path in (checkpoint, backbone_source, backbone_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    model_load_started = time.perf_counter()
    model, state, config = _load_model(
        checkpoint, backbone_source, backbone_checkpoint, device
    )
    model_load_seconds = time.perf_counter() - model_load_started
    preprocess = RGBResizePad(
        int(config["data"]["image_height"]), int(config["data"]["image_width"])
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def infer(rgb: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        with Image.open(rgb) as image:
            tensor = preprocess(image)[None, None].to(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            extraction = model.extract(tensor)
            prediction = model.forward_head(
                extraction,
                enabled_bev_branches=("single",),
                include_scale=False,
            )["single_bev"]
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        occupancy = prediction["occupancy_probability"][0].float().cpu().numpy()
        support = prediction["fov_support_probability"][0].float().cpu().numpy()
        confidence = prediction["navigation_confidence"][0].float().cpu().numpy()
        return occupancy, support, confidence, elapsed

    for _ in range(args.warmup_iterations):
        infer(trials[0].candidate.rgb)
    torch.cuda.reset_peak_memory_stats(device)

    records: list[dict[str, Any]] = []
    jsonl = output / "results.jsonl"
    with jsonl.open("x", encoding="utf-8") as stream:
        for index, trial in enumerate(trials):
            occupancy, support, confidence, inference_seconds = infer(
                trial.candidate.rgb
            )
            astar_started = time.perf_counter()
            planned = plan_on_prediction(
                occupancy,
                support,
                confidence,
                trial.planning,
                occupancy_threshold=args.occupancy_threshold,
                support_threshold=args.support_threshold,
                planning_inflation_radius_m=(
                    args.planning_inflation_radius_m
                ),
                known_ego_pose_clearance_radius_m=(
                    args.known_ego_pose_clearance_radius_m
                ),
                confidence_cost_weight=args.confidence_cost_weight,
            )
            astar_seconds = time.perf_counter() - astar_started
            directory = f"sample_{index:03d}"
            _save_visuals(
                output / directory,
                trial,
                planned,
                occupancy,
                support,
                occupancy_threshold=args.occupancy_threshold,
                support_threshold=args.support_threshold,
            )
            dataset, scene_id, session_id, frame = trial.candidate.key
            record = {
                "index": index,
                "directory": directory,
                "dataset": dataset,
                "scene_id": scene_id,
                "session_id": session_id,
                "frame_index": frame,
                "rgb": str(trial.candidate.rgb),
                "target_selection_source": "GT_ONLY",
                "prediction_used_for_target_sampling": False,
                "goal_grid_row_column": list(trial.planning.goal),
                "goal_x_z_m": [trial.planning.goal_x_m, trial.planning.goal_z_m],
                "goal_occupancy_probability": planned.raw_goal_occupancy_probability,
                "goal_support_probability": planned.raw_goal_support_probability,
                "goal_navigation_confidence": (
                    planned.raw_goal_navigation_confidence
                ),
                "success": planned.success,
                "failure_reason": planned.failure_reason,
                "gt_path_length_m": trial.planning.gt_path_length_m,
                "predicted_path_length_m": planned.predicted_path_length_m,
                "predicted_path_cost_m": planned.predicted_path_cost_m,
                "mean_path_safe_confidence": planned.mean_path_safe_confidence,
                "path_length_ratio": planned.path_length_ratio,
                "gt_collision_cells_on_predicted_path": planned.colliding_path_cells,
                "robot_origin_blocked_before_known_pose_clearance": (
                    planned.robot_origin_blocked_before_known_pose_clearance
                ),
                "inference_ms": inference_seconds * 1000.0,
                "astar_ms": astar_seconds * 1000.0,
            }
            records.append(record)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            print(
                f"[{index + 1:03d}/{len(trials):03d}] "
                f"{'PASS' if planned.success else planned.failure_reason} "
                f"inference={record['inference_ms']:.1f}ms "
                f"astar={record['astar_ms']:.2f}ms",
                flush=True,
            )

    inference = [float(record["inference_ms"]) for record in records]
    astar = [float(record["astar_ms"]) for record in records]
    ratios = [
        float(record["path_length_ratio"])
        for record in records
        if record["path_length_ratio"] is not None
    ]
    weighted_costs = [
        float(record["predicted_path_cost_m"])
        for record in records
        if record["predicted_path_cost_m"] is not None
    ]
    failures = Counter(
        str(record["failure_reason"])
        for record in records
        if not record["success"]
    )
    successes = sum(bool(record["success"]) for record in records)
    summary = {
        "contract": {
            "runtime_model_input": ["one RGB image"],
            "target_selection": (
                "GT-free + true geometric FOV + GT footprint-reachable; "
                "frozen before inference"
            ),
            "prediction_used_for_target_sampling": False,
            "planner_input": (
                "predicted Single occupancy/support/navigation confidence"
            ),
            "failure_if_predicted_target_occupied": True,
            "failure_if_predicted_path_hits_gt_obstacle": True,
            "single_bev_extent_m": args.extent_m,
            "single_bev_native_size": 512,
            "planning_size": args.planning_size,
            "planning_cell_size_m": args.extent_m / args.planning_size,
            "predicted_obstacle_inflation_radius_m": (
                args.planning_inflation_radius_m
            ),
            "gt_success_clearance_radius_m": args.gt_clearance_radius_m,
            "known_ego_pose_clearance_radius_m": (
                args.known_ego_pose_clearance_radius_m
            ),
            "confidence_cost": {
                "safe_confidence": (
                    "(1-p_occupied) * navigation_confidence"
                ),
                "navigation_confidence_source": (
                    "accepted P1B fused output: support * classification "
                    "confidence * distribution confidence"
                ),
                "cell_multiplier": (
                    "1 + lambda * -ln(clamp(safe_confidence, 1e-4, 1))"
                ),
                "edge_cost": (
                    "distance_m * mean(endpoint cell multipliers)"
                ),
                "lambda": args.confidence_cost_weight,
            },
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_schema": state.get("checkpoint_schema"),
        "checkpoint_epoch": int(state["epoch"]),
        "checkpoint_global_step": int(state["global_step"]),
        "backbone_checkpoint": str(backbone_checkpoint),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "model_load_seconds": model_load_seconds,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "candidate_unique_frames": len(candidates),
        "gt_prefilter": {
            "required_for_counting": True,
            "attempted_before_100_qualified": len(trials) + len(rejected),
            "qualified_and_counted": len(trials),
            "rejected_before_inference": len(rejected),
            "all_counted_samples_have_gt_astar_path": all(
                bool(trial.planning.gt_path) for trial in trials
            ),
        },
        "evaluated_samples": len(records),
        "successes": successes,
        "failures": len(records) - successes,
        "success_rate": successes / len(records),
        "failure_reasons": dict(sorted(failures.items())),
        "inference_ms": {
            "mean": statistics.fmean(inference),
            "median": statistics.median(inference),
            "p95": _percentile(inference, 0.95),
            "maximum": max(inference),
        },
        "astar_ms": {
            "mean": statistics.fmean(astar),
            "median": statistics.median(astar),
            "p95": _percentile(astar, 0.95),
            "maximum": max(astar),
        },
        "successful_path_length_ratio": {
            "mean": statistics.fmean(ratios) if ratios else None,
            "median": statistics.median(ratios) if ratios else None,
            "count": len(ratios),
        },
        "confidence_weighted_path_cost_m": {
            "mean": statistics.fmean(weighted_costs) if weighted_costs else None,
            "median": statistics.median(weighted_costs) if weighted_costs else None,
            "count": len(weighted_costs),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "index.html").write_text(
        _html(summary, records), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
