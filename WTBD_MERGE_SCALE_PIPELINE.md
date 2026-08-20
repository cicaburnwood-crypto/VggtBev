# FULL-WTBD: Merge-only VGGT-unit BEV + Metric Scale

Version: `wtbd-merge-only-native-geometry-vggt-unit-scale-v1`

## Fixed architecture contract

```text
RGB window (1..10 frames)
        |
        v
Frozen VGGT-Omega -- one aggregator pass
        |                         |
        | tokens                  | native camera/depth heads
        |                         |
        |              native extrinsics + temporal order
        |                         |
        +--------> Merged branch <+
        |              |
        |              +--> Merged evidential BEV in VGGT units
        |
        +--------> independent Scale branch
                       |
                       +--> lambda_hat [metre / VGGT unit]
```

There is no Single head, Single target, Single checkpoint, learned relative
pose head, path head, camera-height input or Scale-to-Merged connection.
VGGT remains frozen. Both trainable branches start from random initialization.

The Merged branch consumes the frozen patch tokens and a frame embedding made
from the same window's original VGGT camera-from-world matrices. It does not
consume GT geometry at runtime. The Scale Token reads a separate token
projector and is predicted in parallel.

## Training-only target conversion

Existing data is reused. The stored Merged source remains a 10 m, latest-ego,
metric raster. For each window, frozen VGGT depth and metric GT depth produce a
robust label:

```text
lambda* = metre / VGGT runtime unit
```

For a canonical target cell `(x_V,z_V)`, the source lookup is:

```text
x_m = lambda* x_V
z_m = lambda* z_V
```

Complete, masked-visible, FOV support and Void-validity targets all use the
same nearest-neighbour inverse lookup. Cells outside the 10 m source or from a
window with an invalid scale fit are ignored by every BEV loss. This is not a
resize and does not feed predicted scale into the Merged network.

The configured canonical extent is a number in VGGT units. `6.5` is only an
initial audit value. Before a full run, choose and freeze it from the training
distribution of `lambda*` and the logged source-coverage fraction. The 10 m
source limits a sample's complete VGGT-unit coverage to `10/lambda*`.

## Loss and gradient contract

Merged keeps the accepted P1B evidential objective:

- balanced pixel Observed Gate loss;
- guessed free / visible surface / hidden occupied evidential NLL;
- loss-only visible-surface emphasis;
- gradually enabled wrong-evidence regularization;
- FOV support BCE + Dice;
- Void, out-of-source, and invalid-scale cells hard ignored.

Scale uses quality-weighted log-SmoothL1 plus dense metric-depth consistency,
with optional predicted uncertainty. The Merged and Scale branches have
different projectors and output decoders. Therefore Merged loss has no
trainable path to Scale parameters, and Scale loss has no trainable path to
Merged parameters.

## Runtime contract

External input is RGB only. Frozen VGGT internally generates tokens, depth,
intrinsics and extrinsics from that same window. The deployable outputs are:

```text
Merged occupancy/evidence/confidence in a fixed VGGT-unit grid
lambda_hat in metre / VGGT runtime unit
```

Metric restoration is external metadata, not a bitmap resize:

```text
extent_m = lambda_hat * extent_VGGT
cell_m   = extent_m / output_pixels
point_m  = lambda_hat * point_VGGT
```

Camera height may later calibrate or validate `lambda_hat`, but is not a model
input and is not required by this training pipeline.

## Checkpoint rules

Only checkpoints with schema
`wtbd-merge-only-native-geometry-vggt-unit-scale-v1` may resume this pipeline.
Historical P1B/P1C, Single, routing-only and learned-pose checkpoints are
incompatible by design and remain preserved separately.
