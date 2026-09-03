# M05 Latest-Anchored Reverse-Gated Merged BEV

M05 is the active RGB-window to Merged-BEV-and-Scale pipeline. It restores a
full native 512 x 512 query table, anchors the map with the latest frame, and
then applies one shared gated deformable update to history frames from newest
to oldest. A training-only latest-frame auxiliary prediction uses the proven
Single-Baseline map objective; runtime still returns only Merged BEV and Scale.

It reuses the existing 512 x 512, 10 x 10 m Merged supervision and requires no
new data collection. See `M05_IMPLEMENTATION.md` for the complete architecture,
loss, data, and runtime contracts.

```bash
pytest -q tests/test_m05.py tests/test_m04.py tests/test_dataset.py
python -m vggt_bev_method1.cli_train_m05 \
  --config configs/m05_reverse_gated_10m_template.toml \
  --data-only
scripts/train_m05.sh configs/m05_reverse_gated_10m_template.toml
```

## Legacy P1D Direct Merged BEV

P1D is the successor to P1C/WTBD. It is one differentiable, single-forward
runtime path:

```text
RGB window -> frozen VGGT aggregator -> shared temporal tokens
           -> one parallel P1D head -> Merged BEV + confidence + scale
```

The head jointly predicts FOV support, Observed Gate, Guessed occupied/free
Beta evidence, and metre-per-VGGT-unit Scale. Learned bounded frame
reliability modulates internal Cross-Attention. There is no Single BEV input,
extrinsic/pose input, camera-height input, per-frame warp, overwrite, semantic
fusion, or runtime morphology.

Training additionally uses the existing latest-frame masked GT to emphasize
history-only regions, the same evidential NLL for bounded hard-pixel mining,
continuous Gate/FOV boundary weights, and history-frame token dropout. These
targets and sampling rules disappear at runtime.

See `P1D_DIRECT_PIPELINE.md` for the exact I/O, loss and scale contract.

## P1D quick start

```bash
python -m pip install --no-deps -e .
pytest -q tests/test_p1d.py tests/test_p1b.py
python -m vggt_bev_method1.cli_train_p1d \
  --config configs/p1d_direct_merged_scale_template.toml \
  --data-only
GPU_COUNT=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  scripts/train_p1d.sh configs/p1d_direct_merged_scale_template.toml
```

The former WTBD commands remain below for reproducibility; P1D does not delete
or rewrite them.

## Legacy WTBD Merge-Scale

Standalone training pipeline for a frozen-VGGT extension with two outputs:

1. a temporal Merged evidential BEV in VGGT runtime units;
2. a parallel metric Scale Token in metre per VGGT runtime unit.

There is no Single BEV branch, explicit extrinsic input, or learned
relative-pose head. A head-local multi-frame token trunk learns geometry
implicitly and Merged queries cross-attend to all ordered frame tokens.
Predicted scale never conditions Merged.

The full architecture, target conversion, losses and runtime coordinate
contract are documented in `WTBD_MERGE_SCALE_PIPELINE.md`.

## Install

```bash
python -m pip install --no-deps -e .
```

## Validate the standalone contract

```bash
pytest -q tests/test_wtbd_merge_scale.py
```

## Inspect data contract

```bash
python -m vggt_bev_method1.cli_train_wtbd \
  --config configs/wtbd_merge_scale_template.toml \
  --data-only
```

## Train

Set `required_cuda_devices` to the number of visible GPUs, then use:

```bash
GPU_COUNT=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  scripts/train_wtbd.sh configs/wtbd_merge_scale_template.toml
```

Do not launch the full run until `merged_bev_extent_vggt` has been audited and
frozen from the training-set scale distribution.

Run that Stage-0 audit with:

```bash
python -m vggt_bev_method1.cli_audit_wtbd_scale \
  --config configs/wtbd_merge_scale_template.toml \
  --max-samples 2000
```
