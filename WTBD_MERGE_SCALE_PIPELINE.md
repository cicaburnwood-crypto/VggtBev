# FULL-WTBD v2: implicit multi-view Merged BEV + Metric Scale

Version: `wtbd-merge-only-implicit-geometry-cross-attention-scale-v2`

## Architecture contract

```text
RGB window (10 ordered frames)
                    |
                    v
          Frozen VGGT Aggregator (one pass)
                    |
       +------------+-------------------+
       |                                |
patch tokens at 4 levels       camera/register tokens
       |                                |
       |                    4 head-local attention blocks
       |                                |
       +------ per-frame implicit geometry context
       |                                |
       v                                v
Merged BEV queries -- global cross-attention over every frame token
       |
       +--> Merged evidential BEV in VGGT units

The same frozen patch-token memory independently feeds:

Scale Token --> lambda_hat [metre / VGGT runtime unit]
```

The Merged path never receives depth, intrinsics, extrinsics, reconstructed
points, camera height, Single BEV or predicted Scale. It does not wait for any
VGGT task head. Its only runtime inputs are ordered RGB frames; latest-frame
and frame-age embeddings make the output reference frame unambiguous.

The implicit geometry trunk follows the useful structure of VGGT's CameraHead:
final camera/register tokens from all frames are mixed by four small
head-local attention blocks. Unlike CameraHead, it stops at latent per-frame
contexts and has no pose output or pose loss. Merged queries then use global
linear cross-attention to access patch tokens from all frames. Linear attention
keeps all-query/all-token connectivity without the quadratic 800x800 cost.

There is no Single head, learned relative-pose head or path head. VGGT remains
frozen. Merged and Scale have independent projectors and output branches.

## Training-only targets

Existing data is reused. Stored Merged GT remains a 10 m, latest-ego metric
raster. Frozen VGGT depth and metric GT depth produce the robust training
label `lambda* = metre / VGGT unit`. This teacher geometry is used only to
construct targets and validate scale; it is never passed to the Merged head.

For canonical target coordinate `(x_V,z_V)`, target lookup is:

```text
x_m = lambda* x_V
z_m = lambda* z_V
```

Complete, visible, FOV-support and Void-valid masks share the same nearest-
neighbour inverse lookup. Out-of-source cells, invalid scale fits and Void
cells are ignored. Predicted Scale never participates in target construction.

## Loss and gradient contract

Merged retains the accepted evidential P1B objective: balanced pixel Observed
Gate loss; guessed free/visible-surface/hidden-occupied NLL; gradual wrong-
evidence regulation; FOV BCE + Dice; and hard ignoring of invalid cells.

Scale uses quality-weighted log-SmoothL1, dense metric-depth consistency and
optional uncertainty NLL. Merged and Scale parameters are disjoint, so neither
branch's loss can train the other branch.

## Runtime contract

```text
input:  ordered 10-frame RGB sliding window
output: Merged occupancy/evidence/confidence in VGGT-unit grid
        lambda_hat in metre/VGGT-unit
```

Metric restoration is external coordinate metadata, not an image resize:

```text
extent_m = lambda_hat * extent_VGGT
cell_m   = extent_m / output_pixels
point_m  = lambda_hat * point_VGGT
```

The deployable runtime executes only the frozen Aggregator and the parallel
Merged/Scale head. Calling VGGT camera or depth heads is neither required nor
permitted by the runtime forward contract.

## Checkpoint rules

Only schema `wtbd-merge-only-implicit-geometry-cross-attention-scale-v2` can
resume this pipeline. The explicit-extrinsic v1 and historical P1B/P1C/Single
checkpoints are intentionally incompatible and remain preserved as history.
