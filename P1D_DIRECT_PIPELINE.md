# P1D direct Merged-BEV pipeline

## Runtime contract

```text
ordered multi-frame RGB
  -> frozen VGGT aggregator (one invocation)
  -> patch tokens + camera/register tokens
  -> P1D head
       |-- learned bounded frame reliability
       |-- implicit temporal token trunk
       |-- FOV Support
       |-- Observed Gate
       |-- Guessed occupied/free Beta evidence
       `-- Scale Token
  -> Merged BEV + confidence + metre/VGGT-unit scale
```

External runtime input is exactly `rgb_window`. The model does not decode or
consume VGGT depth, intrinsics or extrinsics at runtime. It has no Single BEV,
relative-pose output, camera-height input, warp, last-frame overwrite,
morphological repair, hand FOV mask, confidence threshold router, or external
semantic fusion.

The frozen backbone means the precise paper wording is **direct end-to-end
prediction with a frozen pretrained VGGT backbone**, rather than full-backbone
joint end-to-end training.

## Parallel outputs

- `fov_support_probability`: whether a cell belongs to the multi-frame FOV
  union.
- `observed_gate_probability`: directly observed free versus Guessed routing.
- `alpha_occupied`, `beta_free`: Guessed Expert Beta evidence.
- `occupancy_probability`, `navigation_confidence`: assembled from the same
  parallel outputs, without an extra network or sequential inference pass.
- `lambda_m_per_vggt`: scalar metre per current-window VGGT runtime unit, plus
  optional uncertainty.

The Merged grid remains in VGGT-native units. Scale is predicted in parallel
and never conditions the Merged decoder. External metric restoration is
`x_m = lambda_hat * x_vggt`.

## Learned frame reliability

For frame prefix tokens `F_i`, P1D predicts

```text
r_i = 0.25 + 1.50 * sigmoid(MLP(mean_prefix(F_i)))
```

and normalizes the window mean to one. Fresh initialization produces exactly
uniform reliability. The weights enter the key contribution of global linear
Cross-Attention and the attention bias of the small exact-attention temporal
trunk/Scale Token. They do not select or discard frames at runtime.

During training only, historical frame-token groups are independently dropped
with probability 0.15; the latest reference frame is always retained. Kept
weights are renormalized. This augmentation has no runtime branch.

## Training labels without recollection

Existing data already contains everything P1D needs:

- RGB history;
- 10 m Merged complete and masked GT;
- latest 6.5 m masked GT;
- GT depth for training-only Scale labels;
- GT pose/FOV metadata for online labels only;
- optional Void-valid mask.

The latest masked raster is centered in the 10 m latest-ego Merged source
grid. The same inverse VGGT-unit lookup used for Merged GT is then applied to
the latest targets. Consequently history masks and full targets share exactly
the same scale, grid, orientation and validity mask.

`history_support = merged_support AND NOT latest_support` and
`history_observed = merged_observed_free AND NOT latest_observed_free`.
Neither is a runtime input.

## Loss

The original P1C/WTBD objectives remain intact:

- class-balanced Observed Gate BCE + region Dice;
- continuous Gaussian Gate/FOV boundary weighting;
- class-balanced FOV BCE + Dice;
- pixelwise Guessed evidential NLL for free, visible surface, and hidden
  occupied groups;
- wrong-evidence regularization and its existing curriculum;
- Scale log loss, dense depth-scale consistency and uncertainty.

P1D adds only losses on existing outputs:

```text
L_P1D = L_P1C-base
      + w_hard * hard_topK(the same Guessed evidential NLL)
      + w_HG * Gate-BCE(history-only region)
      + w_HF * positive-FOV-BCE(history-only support)
      + w_HN * Guessed-NLL(history-only region)
      + scale losses
```

Hard mining is restricted to in-FOV, non-Void Guessed cells and performed per
semantic group with a bounded top-K budget. It introduces no second loss
definition and cannot move the Observed Gate target.

## Validation

Validation reports full and history-only Gate/FOV/Guessed precision, recall
and IoU; Gate/FOV boundary precision, recall and Boundary IoU at 1, 2, 4 and 8
pixels; Guessed calibration/NLL/Brier/ECE; Scale relative/log error; and
temporal-gain recall in regions supplied only by history.

## Checkpoint identity

- pipeline: `P1D-DIRECT-MERGED-SCALE-NLL`
- schema: `p1d-direct-merged-temporal-reliability-scale-v1`
- initialization: fresh P1D head
- runtime passes: one
- frozen VGGT invocations per training batch: one
