# M04: Parallel Anchor/History Merged BEV + Scale

M04 is a fresh, end-to-end BEV-and-scale head on top of one frozen VGGT
aggregation pass.  Its runtime input remains an ordered RGB window only.  It
does not consume GT BEV, depth, camera extrinsics, camera height, navigation
targets, a Single-BEV prediction, or an externally fused map.

## Existing supervision contract

M04 deliberately reuses the existing per-frame files:

- `bev_6p5m/merged_complete_10m/frame_*.png`
- `bev_6p5m/merged_masked_10m/frame_*.png`

They remain 512 x 512 rasters spanning 10 x 10 metric metres.  No intermediate
resize, data collection, or relabelling is required. During training, the GT
metric scale `lambda_gt` is used only to inverse-sample those native metric
labels onto the 512 x 512, 6.5-VGGT-unit output grid:

```text
(x_m, z_m) = lambda_gt * (x_vggt, z_vggt)
```

If the native grid extends beyond the existing 10 x 10 m source raster, those
cells are hard-ignored by every BEV loss.  They are never silently labelled as
free or unknown. Training separately logs geometric source coverage, source
coverage over valid-scale samples, and the final effective-supervision
fraction after scale/source/void validity. Evaluation records the same
geometric and effective quantities per sample.

## Architecture

The frozen VGGT token pyramid is decoded by two parallel branches:

1. The latest-anchor branch uses deformable cross-attention over only the
   latest frame.
2. The history branch uses reliability-weighted linear attention over the
   preceding frames, enriched by the shared implicit-geometry context trunk.

Both branches query the final 512 x 512 grid directly. Query content is
factorized into row and column embeddings plus continuous Fourier-coordinate
features, rather than storing one learned vector per pixel.  A learned
residual gate fuses history into the latest anchor, followed by shared spatial
refinement and shared evidential/routing heads.  With a one-frame input, the
history path is structurally skipped and contributes an exact zero residual.

The scale branch is parallel to the BEV decoder. It has its own token
projector, reliability head, and decoder, so Scale and BEV losses share only
frozen VGGT features and cannot update each other's trainable path. It
predicts metres per VGGT runtime unit and uncertainty from the same frozen
VGGT extraction. Both independent reliability distributions are logged rather
than presenting the BEV reliability as if it also controlled Scale.

CUDA runtime inference uses the same AMP token precision as training and
evaluation. Teacher-only VGGT aggregation/image references and their temporary
FP32 token copies are released before the native 512-grid head is decoded.

Because the history branch is truly absent from the `N=1` graph and active
for `N>1`, distributed training uses dynamic DDP with unused-parameter
discovery. Static-graph DDP is rejected by the M04 config validator rather
than risking a reduction failure at the first one-frame batch.

## Objective

The map objective is one hierarchical probabilistic likelihood:

- balanced Bernoulli NLL for temporal-FOV support;
- balanced Bernoulli NLL for observed-free versus inferred content;
- balanced expected Beta NLL for inferred occupied/free content;
- a small uniform-Beta evidence prior.

There are no Dice, boundary, hard-pixel, temporal-region, curriculum,
planner, or navigation losses.  Scale uses a robust Student-t likelihood of
dense log depth ratios.  The total joint objective is the sum of the two
normalized likelihoods.

## Train and evaluate

Copy the template and change only environment-specific paths such as the data
root, split manifest, VGGT source/checkpoint, and output directory:

```bash
cp configs/m04_parallel_anchor_history_10m_template.toml configs/m04.toml
python -m vggt_bev_method1.cli_train_m04 \
  --config configs/m04.toml \
  --data-only

torchrun --standalone --nproc-per-node=1 \
  -m vggt_bev_method1.cli_train_m04 \
  --config configs/m04.toml

python -m vggt_bev_method1.cli_eval_m04 \
  --config configs/m04.toml \
  --checkpoint runs/m04/m04_latest.pt \
  --output-dir runs/m04/eval
```

`--data-only` verifies the manifest/dataset contract without loading VGGT or
starting training.  Resume is fail-closed: the M04 schema, manifest hash, VGGT
checkpoint hash, grid contract, and 10 m source contract must all match.
