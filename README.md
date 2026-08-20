# WTBD Merge-Scale

Standalone training pipeline for a frozen-VGGT extension with two outputs:

1. a temporal Merged evidential BEV in VGGT runtime units;
2. a parallel metric Scale Token in metre per VGGT runtime unit.

There is no Single BEV branch and no learned relative-pose head. Original VGGT
extrinsics condition the Merged branch; predicted scale never does.

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
