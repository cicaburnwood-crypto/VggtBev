from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vggt_bev_method1.config import LabelValues
from vggt_bev_method1.data.fov_targets import (
    cap_complete_and_visible_to_fov,
    fov_union_mask,
    load_world_from_bev_planar,
)
from vggt_bev_method1.data.preprocess import RGBResizePad
from vggt_bev_method1.metrics import fov_complete_evidential_metrics
from vggt_bev_method1.models import LiveVGGTOmegaAdapter, Method1System


CLASS_VALUES = np.asarray((255, 0), dtype=np.uint8)
CLASS_COLORS = np.asarray(
    (
        (73, 206, 122),  # free
        (242, 78, 78),  # occupied
    ),
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a held-out P1B v5 RGB-only evaluation window"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-source", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prefixes",
        default="1,4,8,12,16,20,24,28,32",
        help="comma-separated RGB history lengths",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def heldout_contract(manifest_path: Path, metadata: dict) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_key = f"{metadata['dataset']}:{metadata['scene_id']}"
    train_scenes = {entry["scene_key"] for entry in manifest["train"]}
    validation_scenes = {entry["scene_key"] for entry in manifest["validation"]}
    if scene_key in train_scenes or scene_key not in validation_scenes:
        raise ValueError(f"scene is not validation-only: {scene_key}")

    # The frozen training split is 90/10. Select exactly round(5%) of all
    # scenes from its validation-only pool with the original split seed.
    five_percent_count = round(int(manifest["scene_count"]) * 0.05)
    rng = random.Random(int(manifest["split_seed"]))
    five_percent_scenes = set(
        rng.sample(sorted(validation_scenes), five_percent_count)
    )
    if scene_key not in five_percent_scenes:
        raise ValueError(
            f"scene {scene_key} is held out but not in the deterministic 5% subset"
        )

    matching = [
        entry
        for entry in manifest["validation"]
        if entry["scene_key"] == scene_key
        and entry["key"].endswith(metadata["session_id"])
    ]
    if len(matching) != 1:
        raise ValueError("session does not resolve uniquely in validation manifest")
    return {
        "scene_key": scene_key,
        "session_key": matching[0]["key"],
        "train_scene_overlap": 0,
        "frozen_validation_fraction": float(manifest["validation_fraction"]),
        "evaluation_subset_scene_count": five_percent_count,
        "total_scene_count": int(manifest["scene_count"]),
        "evaluation_subset_fraction": five_percent_count
        / int(manifest["scene_count"]),
        "split_seed": int(manifest["split_seed"]),
        "manifest_content_sha256": manifest["content_sha256"],
    }


def build_system(
    state: dict,
    *,
    backbone_source: Path,
    backbone_checkpoint: Path,
    device: torch.device,
) -> Method1System:
    if (
        state.get("checkpoint_schema")
        != "p1b-fixed-metric-fov-complete-evidential-v6"
    ):
        raise ValueError(
            "visualizer requires an FOV-complete P1B v6 checkpoint"
        )
    model = state["config"]["model"]
    layers = tuple(int(value) for value in model["cached_layers"])
    adapter = LiveVGGTOmegaAdapter(
        backbone_source,
        backbone_checkpoint,
        device=device,
        patch_size=int(model["patch_size"]),
        cached_layers=layers,
    )
    system = Method1System(
        adapter,
        cached_layers=layers,
        spatial_scales=tuple(float(value) for value in model["spatial_scales"]),
        vggt_token_dim=int(model["vggt_token_dim"]),
        hidden_dim=int(model["hidden_dim"]),
        heads=int(model["attention_heads"]),
        decoder_layers=int(model["decoder_layers"]),
        scale_decoder_layers=int(model["scale_decoder_layers"]),
        self_attention_mode=str(model["self_attention_mode"]),
        cross_attention_mode=str(model["cross_attention_mode"]),
        deformable_samples=int(model["deformable_samples"]),
        cross_query_chunk_size=int(model["cross_query_chunk_size"]),
        single_latent_bev_size=int(model["single_latent_bev_size"]),
        merged_latent_bev_size=int(model["merged_latent_bev_size"]),
        single_output_size=int(model["single_bev_output_size"]),
        merged_output_size=int(model["merged_bev_output_size"]),
        single_bev_extent_m=float(model["single_bev_extent_m"]),
        merged_bev_extent_m=float(model["merged_bev_extent_m"]),
        predict_scale_uncertainty=bool(
            model.get("predict_scale_uncertainty", True)
        ),
    ).to(device)

    system.head.load_state_dict(state["head_state_dict"], strict=True)
    return system.eval()


def load_target(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as image:
        gray = image.convert("L")
        if gray.size != (size, size):
            gray = gray.resize((size, size), Image.Resampling.NEAREST)
        array = np.asarray(gray, dtype=np.uint8).copy()
    return torch.from_numpy(array)


def save_semantic(
    class_index: torch.Tensor,
    support: torch.Tensor,
    path: Path,
) -> None:
    classes = class_index.detach().cpu().numpy().astype(np.int64)
    known = support.detach().cpu().numpy().astype(bool)
    rendered = np.full(classes.shape, 112, dtype=np.uint8)
    rendered[known] = CLASS_VALUES[classes[known]]
    Image.fromarray(rendered, mode="L").save(path)


def save_confidence(
    class_index: torch.Tensor,
    confidence: torch.Tensor,
    support_probability: torch.Tensor,
    path: Path,
) -> None:
    classes = class_index.detach().cpu().numpy().astype(np.int64)
    strength = confidence.detach().cpu().numpy()[..., None].astype(np.float32)
    color = CLASS_COLORS[classes]
    background = np.full_like(color, 28.0)
    rendered = background * (1.0 - strength) + color * strength
    support = support_probability.detach().cpu().numpy()[..., None]
    rendered = 112.0 * (1.0 - support) + rendered * support
    Image.fromarray(rendered.clip(0, 255).astype(np.uint8), mode="RGB").save(path)


def tensor_metrics(
    prediction: dict[str, torch.Tensor],
    fov_complete_target: torch.Tensor,
    visible_target: torch.Tensor,
    fov_support_target: torch.Tensor,
    extent_m: float,
) -> dict[str, float]:
    device = prediction["occupancy_probability"].device
    values = fov_complete_evidential_metrics(
        prediction,
        fov_complete_target[None].to(device),
        visible_target[None].to(device),
        fov_support_target[None].to(device),
        target_extent_m=torch.tensor(
            [extent_m],
            device=device,
        ),
    )
    return {key: float(value.detach().cpu()) for key, value in values.items()}


def html_document(contract: dict) -> str:
    payload = json.dumps(contract, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P1B held-out 5% visualization</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin:0; background:#0d1117; color:#e6edf3; }}
header {{ padding:20px 28px; border-bottom:1px solid #30363d; }}
h1 {{ margin:0 0 8px; font-size:22px; }}
.sub {{ color:#9da7b3; font-size:13px; }}
.bar {{ display:flex; gap:16px; align-items:center; padding:14px 28px;
        position:sticky; top:0; background:#0d1117ee; z-index:2;
        border-bottom:1px solid #30363d; }}
input[type=range] {{ width:min(520px,60vw); }}
.pill {{ background:#1f6feb22; border:1px solid #388bfd88;
         padding:6px 10px; border-radius:999px; }}
.grid {{ display:grid; grid-template-columns:repeat(3,minmax(260px,1fr));
         gap:12px; padding:18px 28px 30px; }}
.card {{ background:#161b22; border:1px solid #30363d; border-radius:10px;
         overflow:hidden; }}
.card h2 {{ font-size:14px; margin:0; padding:10px 12px;
            border-bottom:1px solid #30363d; }}
.card img {{ width:100%; aspect-ratio:1/1; object-fit:contain; display:block;
             background:#808080; }}
.camera img {{ aspect-ratio:4/3; object-fit:cover; }}
.stats {{ padding:12px; line-height:1.55; color:#b7c0cb; font-size:13px; }}
.legend {{ display:flex; gap:14px; margin-top:8px; }}
.dot {{ width:10px; height:10px; display:inline-block; margin-right:5px; }}
@media(max-width:900px) {{ .grid {{grid-template-columns:1fr;}} }}
</style>
</head>
<body>
<header>
  <h1>P1B — held-out 5% scene visualization</h1>
  <div class="sub">{contract['scene_key']} · zero train-scene overlap ·
  RGB-only model input · checkpoint step {contract['checkpoint_global_step']}</div>
</header>
<div class="bar">
  <label for="prefix">History</label>
  <input id="prefix" type="range" min="0" value="0" step="1">
  <span id="history" class="pill"></span>
  <span id="latency"></span>
</div>
<main class="grid">
  <section class="card camera"><h2>Latest RGB frame</h2><img id="camera"></section>
  <section class="card"><h2>GT FOV-complete single · 6.5 m</h2><img id="gtSingle"></section>
  <section class="card"><h2>Predicted FOV single · 6.5 m</h2><img id="predSingle"></section>
  <section class="card"><h2>Single confidence</h2><img id="confSingle"></section>
  <section class="card"><h2>GT FOV-union merged · 10 m</h2><img id="gtMerged"></section>
  <section class="card"><h2>Predicted FOV-union merged · 10 m</h2><img id="predMerged"></section>
  <section class="card"><h2>Merged confidence</h2><img id="confMerged"></section>
  <section class="card">
    <h2>Metrics and scale</h2>
    <div id="stats" class="stats"></div>
  </section>
  <section class="card">
    <h2>Semantic legend</h2>
    <div class="stats">
      Semantic maps use the training label contract.
      Maps show complete occupied/free predictions inside predicted FOV support.
      Gray is unknown/outside-FOV. Brightness is Beta-evidence confidence;
      occluded inference should remain less confident than direct observation.
      <div class="legend">
        <span><i class="dot" style="background:#49ce7a"></i>free</span>
        <span><i class="dot" style="background:#f24e4e"></i>occupied</span>
      </div>
    </div>
  </section>
</main>
<script>
const data = {payload};
  const slider = document.querySelector("#prefix");
  slider.max = data.records.length - 1;
  const fmt = v => (100*v).toFixed(2) + "%";
  function draw() {{
    const r = data.records[Number(slider.value)];
    document.querySelector("#history").textContent = r.history_frames + " RGB frames";
    document.querySelector("#latency").textContent = r.inference_seconds.toFixed(2) + " s";
    for (const [id,key] of Object.entries({{
      camera:"camera", gtSingle:"gt_single", predSingle:"pred_single",
      confSingle:"confidence_single", gtMerged:"gt_merged",
      predMerged:"pred_merged", confMerged:"confidence_merged"
    }})) document.querySelector("#"+id).src = r[key];
    document.querySelector("#stats").innerHTML =
      `<b>λ:</b> ${{r.lambda_m_per_vggt.toFixed(4)}} m/VGGT-unit<br>` +
      `<b>scale σ:</b> ${{r.scale_std_m_per_vggt.toFixed(4)}}<br>` +
      `<b>single FOV IoU:</b> ${{fmt(r.single_metrics.support_iou)}}<br>` +
      `<b>single occupied IoU:</b> ${{fmt(r.single_metrics.occupied_iou)}}<br>` +
      `<b>single free IoU:</b> ${{fmt(r.single_metrics.free_iou)}}<br>` +
      `<b>merged FOV IoU:</b> ${{fmt(r.merged_metrics.support_iou)}}<br>` +
      `<b>merged occupied IoU:</b> ${{fmt(r.merged_metrics.occupied_iou)}}<br>` +
      `<b>merged free IoU:</b> ${{fmt(r.merged_metrics.free_iou)}}`;
  }}
  slider.addEventListener("input", draw); slider.value = data.records.length-1; draw();
</script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    session = args.session.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
    contract = heldout_contract(args.manifest.expanduser().resolve(), metadata)
    frame_count = int(metadata["frame_count"])
    extrinsic_records = [
        json.loads(line)
        for line in (
            session
            / str(
                metadata.get(
                    "camera_extrinsics_file",
                    "camera_extrinsics.jsonl",
                )
            )
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    world_from_bev_planar = load_world_from_bev_planar(
        extrinsic_records,
        expected_frames=frame_count,
    )
    horizontal_fov_degrees = float(
        metadata["camera_intrinsics"]["horizontal_fov_degrees"]
    )
    labels = LabelValues()
    prefixes = sorted(
        {
            int(value)
            for value in args.prefixes.split(",")
            if value.strip()
        }
    )
    if not prefixes or prefixes[0] < 1 or prefixes[-1] > frame_count:
        raise ValueError(f"prefixes must stay inside 1..{frame_count}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is unavailable")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    system = build_system(
        state,
        backbone_source=args.backbone_source.expanduser().resolve(),
        backbone_checkpoint=args.backbone_checkpoint.expanduser().resolve(),
        device=device,
    )

    preprocess = RGBResizePad(
        int(state["config"]["data"]["image_height"]),
        int(state["config"]["data"]["image_width"]),
    )
    images = []
    for frame in range(frame_count):
        with Image.open(session / "camera" / f"frame_{frame:06d}.png") as image:
            images.append(preprocess(image))

    output.mkdir(parents=True, exist_ok=True)
    records = []
    for prefix in prefixes:
        frame = prefix - 1
        frame_dir = output / f"history_{prefix:02d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            session / "camera" / f"frame_{frame:06d}.png",
            frame_dir / "camera.png",
        )
        source_merged_complete = load_target(
            session
            / "bev_6p5m/merged_complete_10m"
            / f"frame_{frame:06d}.png",
            800,
        )
        source_single_complete = load_target(
            session / "bev_6p5m/complete" / f"frame_{frame:06d}.png",
            512,
        )
        source_single_visible = load_target(
            session / "bev_6p5m/masked" / f"frame_{frame:06d}.png",
            512,
        )
        source_merged_visible = load_target(
            session
            / "bev_6p5m/merged_masked_10m"
            / f"frame_{frame:06d}.png",
            800,
        )
        single_fov = fov_union_mask(
            world_from_bev_planar[frame : frame + 1],
            target_frame=0,
            horizontal_fov_degrees=horizontal_fov_degrees,
            output_size=512,
            output_extent_m=6.5,
        )
        merged_fov = fov_union_mask(
            world_from_bev_planar,
            target_frame=frame,
            horizontal_fov_degrees=horizontal_fov_degrees,
            output_size=800,
            output_extent_m=10.0,
        )
        (
            single_target,
            single_visible,
            single_support_target,
        ) = cap_complete_and_visible_to_fov(
            source_single_complete,
            source_single_visible,
            single_fov,
            labels=labels,
        )
        (
            merged_target,
            merged_visible,
            merged_support_target,
        ) = cap_complete_and_visible_to_fov(
            source_merged_complete,
            source_merged_visible,
            merged_fov,
            labels=labels,
        )
        Image.fromarray(single_target.numpy(), mode="L").save(
            frame_dir / "gt_single.png"
        )
        Image.fromarray(merged_target.numpy(), mode="L").save(
            frame_dir / "gt_merged.png"
        )

        model_input = torch.stack(images[:prefix])[None].to(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = system(model_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.monotonic() - started

        single_probability = prediction["single_bev"]["occupancy_probability"]
        merged_probability = prediction["merged_bev"]["occupancy_probability"]
        single_confidence = prediction["single_bev"]["evidence_confidence"]
        merged_confidence = prediction["merged_bev"]["evidence_confidence"]
        single_support = prediction["single_bev"]["fov_support_probability"]
        merged_support = prediction["merged_bev"]["fov_support_probability"]
        single_class = (single_probability >= 0.5).long()
        merged_class = (merged_probability >= 0.5).long()
        save_semantic(
            single_class[0],
            single_support[0] >= 0.5,
            frame_dir / "pred_single.png",
        )
        save_semantic(
            merged_class[0],
            merged_support[0] >= 0.5,
            frame_dir / "pred_merged.png",
        )
        save_confidence(
            single_class[0],
            single_confidence[0],
            single_support[0],
            frame_dir / "confidence_single.png",
        )
        save_confidence(
            merged_class[0],
            merged_confidence[0],
            merged_support[0],
            frame_dir / "confidence_merged.png",
        )
        single_metrics = tensor_metrics(
            prediction["single_bev"],
            single_target,
            single_visible,
            single_support_target,
            6.5,
        )
        merged_metrics = tensor_metrics(
            prediction["merged_bev"],
            merged_target,
            merged_visible,
            merged_support_target,
            10.0,
        )
        scale = prediction["scale"]
        relative = frame_dir.relative_to(output)
        records.append(
            {
                "history_frames": prefix,
                "frame_index": frame,
                "inference_seconds": inference_seconds,
                "peak_vram_mib": (
                    torch.cuda.max_memory_allocated(device) / 2**20
                    if device.type == "cuda"
                    else 0.0
                ),
                "lambda_m_per_vggt": float(
                    scale["lambda_m_per_vggt"][0].detach().cpu()
                ),
                "scale_std_m_per_vggt": float(
                    scale["scale_std_m_per_vggt"][0].detach().cpu()
                ),
                "single_metrics": single_metrics,
                "merged_metrics": merged_metrics,
                **{
                    name: str(relative / filename)
                    for name, filename in {
                        "camera": "camera.png",
                        "gt_single": "gt_single.png",
                        "pred_single": "pred_single.png",
                        "confidence_single": "confidence_single.png",
                        "gt_merged": "gt_merged.png",
                        "pred_merged": "pred_merged.png",
                        "confidence_merged": "confidence_merged.png",
                    }.items()
                },
            }
        )
        del model_input, prediction
        if device.type == "cuda":
            torch.cuda.empty_cache()

    contract.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_schema": state["checkpoint_schema"],
            "checkpoint_global_step": int(state["global_step"]),
            "checkpoint_epoch": int(state["epoch"]),
            "session": str(session),
            "frame_count": frame_count,
            "runtime_inputs": ["RGB window"],
            "gt_usage": "visual comparison and metrics only; never model input",
            "device": str(device),
            "records": records,
        }
    )
    (output / "data.json").write_text(
        json.dumps(contract, indent=2),
        encoding="utf-8",
    )
    (output / "index.html").write_text(
        html_document(contract),
        encoding="utf-8",
    )
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
