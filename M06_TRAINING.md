# M06 training snapshot

## Identity and provenance

Release branch: `M06`. Source: the deployed A100 `m05_plus_train` code on
2026-09-10. This release renames the GitHub snapshot, not the tensor schema.
Runtime pipeline: `M05PLUS-PER-QUERY-TEMPORAL-ATTENTION-LARGE-METRIC-NLL`.
Checkpoint schema: `m05plus-dpt-role-token-per-query-temporal-fixed-metric-v3`.

Active configuration: `configs/m05_plus_a100_8gpu_10e_256_frozen.toml`.
Entry point: `python -m vggt_bev_method1.cli_train_m05_plus`.
Architecture details: [M05_PLUS_IMPLEMENTATION.md](M05_PLUS_IMPLEMENTATION.md).
Where older prose differs, the active 256-resolution config is authoritative.

## Input, output and training

Chronological RGB windows (1–10 frames) are resized/padded to 384 x 512.
VGGT internally anchors the latest frame. The head produces an evidential
256 x 256 BEV over a fixed 10 x 10 metre extent, with occupancy, support and
confidence. The independent scale branch does not condition BEV output.

BEV supervision includes observed gate, guessed occupied/free evidence,
visible surface, FOV support, and wrong-evidence regularization. Scale uses
Student-t NLL. The latest-frame auxiliary objective has multiplier 0.1 and
interval 4; the early full-frequency fraction is 0.2.

Eight ranks, local batch 1, global batch 8; 10 epochs; learning rate 1e-4,
minimum 1e-5; warmup fraction 0.03; weight decay 0.02; gradient clip 1.0.
History caps by epoch: `[3,5,8,10,10,10,10,10,10,10]`.
Fused optimizer, channels-last, cuDNN benchmark, no head compilation and
no attention activation checkpointing. Checkpoints every 5,000 steps.
See the config for the full loss weights and execution settings.

Frozen manifest content SHA-256:
`7cce63f1592d696266aadc2f81a9c97637c169c943dafb305156e1c98cf482b3`.
535,307 training sessions / 1,022 scenes; 26,226 validation sessions / 54 scenes;
scene-key intersection is zero. Incomplete sessions are skipped using the
existing incomplete-session index. Manifest and index are external artifacts.

## Resume

Install the existing dependencies and provide the dataset, frozen VGGT weights,
manifest, session cache and incomplete-session index expected by the config.
The checked-in A100 config contains site-specific paths that must be adapted
on another server. Preserve the cache/manifest hash binding when relocating.

```bash
export PYTHONPATH="$PWD/src"
export VGGT_BEV_SESSION_RECORD_CACHE=/path/to/session_records.pkl
export VGGT_BEV_SESSION_RECORD_CACHE_MANIFEST_SHA256=7cce63f1592d696266aadc2f81a9c97637c169c943dafb305156e1c98cf482b3
torchrun --standalone --nproc_per_node=8 \
  -m vggt_bev_method1.cli_train_m05_plus \
  --config configs/m05_plus_a100_8gpu_10e_256_frozen.toml \
  --resume /path/to/m05_plus_step_00290000.pt --skip-validation
```

Latest checkpoint at publication: `m05_plus_step_00290000.pt`, saved 2026-09-10
08:15 HKT. Training then stopped as requested; the planned 10 epochs are not
complete. Historical operational launchers and watchdog scripts are retained
as source snapshots and contain dated paths/checkpoints. Do not run them
blindly: use the intended checkpoint and obtain GPU availability first.

No training was started as part of publishing this branch. No credentials,
model weights, datasets, full manifests, logs or runtime caches are published.
