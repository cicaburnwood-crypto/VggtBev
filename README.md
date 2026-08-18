# OdinEye P1B — Pixel Routing + Guessed Completion + Scale Token

The active implementation is a from-scratch replacement for the former P1B
occupancy objective. Frozen VGGT aggregation runs exactly once per RGB window.
Independent Guessed and Routing paths consume the same frozen token memory.
There is no learned Observed occupancy decoder.

Two strict variants are supported:

- `P1B-NLL`: Beta evidence, pixelwise Evidential NLL and annealed wrong-evidence
  KL for guessed completion;
- `P1B-BCE`: Bernoulli logits and pixelwise BCE, with classification certainty
  but no claim of epistemic confidence.

Both variants retain the same RGB-only runtime contract, 512x512/6.5 m single
BEV, 800x800/10 m merged BEV and metric Scale Token. The Routing decoder has
one Observed-free Gate and no Surface Gate. Masked free pixels supervise the
observed-free state. Every other valid FOV pixel, including masked occupied
boundary pixels, is handled by the Guessed Expert as guessed-free or
guessed-occupied. The three final states are `observed_free`, `guessed_free`,
and `guessed_occupied`. Outside the FOV remains unknown.

Training validation reports `observed_gate_precision`,
`observed_gate_recall`, `observed_gate_f1`, and `observed_gate_iou` by
thresholding the Observed Gate at 0.5 and comparing it directly with the
masked-GT observed-free region. These metrics are not inferred from final
three-state argmax routing.

No ray bank, contour, dilation, morphology, hole or continuity loss is used.
The masked GT defines one loss-only surface set exactly as
`visible_target == occupied`; it is not a new routing class or output head.
Each sample is split into four disjoint supervision subsets:

```text
O  = observed free
S  = exact visible occupied surface
Hf = hidden guessed free
Ho = hidden guessed occupied
```

The Observed Gate retains its original per-sample 1:1 BCE between `O` and all
Guessed cells; surface grouping cannot change its class weights. The Guessed
Expert NLL averages `Hf/S/Ho` with weights `.35/.40/.25`. Missing groups are
skipped and the present weights are renormalized for that sample. An additional
surface term supervises `-log(P_guess-occupied)` on `S` and sends gradients only
to the Guessed Expert. Hidden occupied supervision is zero for the first 10%
of updates, ramps during 10%-25%, then reaches full weight. Wrong-evidence KL
uses coefficient `.005`, starts after 20%, and ramps during 20%-30%.

The new training entrypoint is:

```bash
python -m vggt_bev_method1.cli_train_p1b --config CONFIG.toml
```

### Current Merged routing training

`configs/p1b_merged6p5_role_contour_2epoch.toml` reproduces the current
Merged-only training path without embedding machine-specific paths. It freezes
VGGT, Single, Scale and Guessed parameters and trains only the independent
Merged routing projector/decoder for FOV Support and Observed Gate.

The 6.5 m Merged target is a metric centre crop of the existing 10 m Merged
raster, never a resize. Both latent and output grids are native 512x512. FOV
Support uses independently normalized interior/edge BCE plus contour and
region Dice weights `0.25/0.45/0.20/0.10`; Observed Gate uses
`0.20/0.50/0.20/0.10`. The first 20% of the two-epoch schedule trains FOV
Support only, after which both routing outputs train together.

Execution-only acceleration uses bounded 65,536-query deformable-attention
chunks, head-only `torch.compile`, fused AdamW and activation checkpointing.
These settings do not change model parameters or the loss contract.

Checkpoint schemas are intentionally incompatible:

```text
P1B-NLL: p1b-three-region-evidential-v6
P1B-BCE: p1b-three-region-bce-v6
```

## Standalone GT Void audit

P1B training has been rewound to the Stage-1 baseline supervision contract.
Scene-geometry Void masks are not dataset fields, model inputs, loss masks,
metric masks, checkpoint fields, or output classes. Every semantic/FOV pixel is
handled exactly as it was by the 36K Surface-NLL baseline.

Missing-scene review now lives in the independent `p1b_void_audit` package.
It reads the immutable geometry coverage sidecars and existing GT rasters, but
P1B never imports it. The audit is read-only and writes:

- `void_fraction_per_gt_bev.csv`: one row per corresponding Single/Merged,
  Complete/Observed GT BEV;
- `void_fraction_summary.json`: aggregate statistics for each GT type;
- `void_fraction_of_grid`: Void pixels divided by all BEV pixels;
- `void_fraction_of_gt_known`: Void pixels intersecting known semantic GT,
  divided by known semantic pixels.

Run it independently with:

```bash
odineye-p1b-void-audit \
  --dataset-root DATASET \
  --manifest MANIFEST.json \
  --void-index COVERAGE_INDEX.json \
  --output-dir AUDIT_OUTPUT
```

The default audits the final prefix used by one-prefix-per-session training.
Use `--frame-mode all` to report every frame up to the ten-frame window.

Legacy P1B modules remain in the repository only for historical comparison and
old checkpoint inspection. They are not called by `cli_train_p1b`.
Pre-rename `P2B-*` configs are quarantined under `configs/legacy_p2b/`; they
are historical records, not valid active launch configurations.

## Legacy P1B reference

This project implements the revised non-cascaded P1B. A frozen VGGT-Ω
aggregator runs once per RGB window. One trainable extension reads its shared
tokens and produces two parallel outputs:

```text
RGB window
   │
   ▼
frozen VGGT aggregator
   │ shared token memory
   ├────────────────────────────┐
   ▼                            ▼
FOV-complete evidential BEVs   Scale Token
├─ single 512×512, 6.5 m      λ [metre / VGGT unit]
└─ merged 800×800, 10 m
   ├─ learned FOV support
   ├─ occupied/free probability
   └─ evidence confidence/uncertainty
```

There is no canonical scale, ground token, path head, or camera-height input
in this version.

## Runtime contract

The only external runtime input is an RGB window. `Method1System.forward()`
runs the frozen aggregator and the P1B head; it does not run or consume VGGT
depth, intrinsics, extrinsics, confidence, or a point cloud.

The optional interactive verifier is the sole exception at the serving layer:
it decodes the frozen VGGT camera head after shared aggregation so that the
closed-loop controller can estimate inter-frame motion. It still takes RGB
only, does not execute the depth head, and does not feed camera poses into the
P1B BEV predictor.

The output is:

- per-BEV `routing_probability`: `[B,3,H,W]` ordered as observed-free,
  guessed-free, guessed-occupied;
- per-BEV `guessed`: occupied/free Beta evidence used only for completion;
- `single_bev.fov_support_probability`: `[B,512,512]`;
- `merged_bev.fov_support_probability`: `[B,800,800]`;
- per-BEV `occupancy_probability = P(guessed) × P(occupied|guessed)`;
  outside predicted support is unknown;
- per-BEV `fov_complete_semantic` assembled as occupied `0`, unknown `112`,
  and free `255`;
- per-BEV routing entropy, mixture variance and navigation confidence;
- `scale.lambda_m_per_vggt`: one positive scalar per window;
- optional `scale.log_variance` and `scale_std_m_per_vggt`.

There is no separate confidence head. Completion uncertainty comes from Beta
evidence, while routing uncertainty comes from the three-state distribution;
the final mixture variance includes both sources.

The existing simulator labels define a 6.5 m × 6.5 m latest-ego-centred grid:

```text
x ∈ [-3.25, 3.25] m
z ∈ [-3.25, 3.25] m
forward = image-up
cell size = 6.5 / 512 m
```

The merged output is a 10 m × 10 m latest-ego-centred grid:

```text
x,z ∈ [-5,5] m
forward = image-up
cell size = 10 / 800 m
```

The single branch reads latest-frame VGGT tokens. The merged branch directly
reads all RGB-window VGGT tokens; it does not consume or merge predicted
single-frame BEVs.

The forward-only `[0,6.5] m` layout in the design note was an example. This
implementation deliberately preserves the dataset's centred grid so the GT,
training, validation, runtime, and planner conventions remain identical
without recollecting data.

## Interactive A* verifier

Start the local Habitat verifier with:

```bash
./run_interactive_verifier.sh
```

Click a GT-free target on Complete GT and select one of two modes:

- **Open loop:** run exact-grid A* once on the synchronized predicted BEV,
  then execute those metric waypoints without replanning.
- **Closed loop:** on every predicted-BEV update, obtain the relative camera
  transform from VGGT world-to-camera extrinsics, multiply only its
  translation by the Scale Token λ, express the fixed target in the new ego
  frame, and rerun exact-grid A*.

Both modes plan on the native 512×512, 6.5 m semantic raster with zero
inflation. Occupied and unknown cells are blocked. A closed-loop missing pose,
broken history chain, newly blocked target, or disconnected route is surfaced
as an explicit failure and stops execution.

## Training targets and teacher path

Training first predicts BEV and scale from frozen aggregator tokens. It then
decodes VGGT depth/confidence only to construct the scale label:

```text
GT metric z-depth / VGGT z-depth
   → log-ratio median
   → residual rejection
   → confidence-weighted Huber IRLS
   → λ_gt and quality q
```

GT depth, GT poses/intrinsics, and BEV labels never enter the model forward.
At dataset access time, camera horizontal FOV and GT planar poses generate an
unobstructed FOV footprint. The single target uses the latest camera footprint;
the merged target unions all historical footprints after transforming them
into the latest ego frame. This is a geometric frustum mask, not a
depth/occlusion mask, so complete collision truth behind obstacles remains
supervised. Complete BEVs are capped to that footprint. Existing masked BEVs
only partition its known cells into directly visible and occluded/inferred
regions.

The content loss treats direct observation and completion as different tasks:

```text
observed-free    = visible known cells whose complete label is free
observed-surface = visible known cells whose complete label is occupied
guessed          = visible unknown cells whose FOV-complete label is known

Ldirect = task-balanced(
    expected Beta NLL(observed-free outside the surface band),
    minimum occupied Beta NLL within the surface tolerance
)

Lguess = class-weighted expected Beta NLL(guessed)
         + β mean(Dice(guessed occupied), Dice(guessed free))

Lbev = fixed-region-weight(Ldirect, g(step) Lguess)
       + balanced FOV-support BCE + support Dice
       + annealed wrong-evidence/calibration/relation losses

Ltotal = wB Lbev
+ wS quality-weighted SmoothL1(log λ)
+ wD dense robust metric-depth consistency
+ wU optional scale uncertainty NLL
```

Observed occupied labels are ray-hit obstacle surfaces and can be only one
cell thick. They therefore never receive Dice or another area-overlap loss.
At the current 64/80 latent grids, a hit is accepted within half one latent
cell: 4 output pixels for single BEV and 5 for merged BEV. The same narrow band
is removed from observed-free supervision, so the decoder is not asked to
predict both occupied and free at one representational location. This
tolerance is computed from output/latent resolution; increasing latent
resolution automatically shrinks it.

The surface NLL is averaged as an independent task, so a one-percent surface
set is not drowned by the much larger observed-free set. Completion overlap is
the macro average of occupied and free Dice and applies only to the
occluded/guessed region; this prevents an all-occupied completion from being
rewarded as a useful solution. `g(step)` holds completion supervision at zero
for the first 5% of optimizer steps and ramps it over the next 10%, without
changing the direct-observation coefficient.

When direct and completion occupancy gradients conflict, direct-priority
PCGrad removes only the component of the completion gradient that points
against the direct gradient. It uses the same head forward and the same one
frozen-VGGT aggregation; it does not add a model branch or a second VGGT pass.
In DDP, both task gradients are averaged before projection and the final
gradient is synchronized exactly once.

Masked unknown cells inside the FOV are never interpreted as unknown content.
If complete GT contains a label there, it is an occluded-content training cell.
There is no fixed confidence ceiling: confidence is calibrated to correctness,
while a relative margin teaches visible predictions to be more confident than
occluded inferences. Outside-FOV cells are unknown and are handled only by the
support branch; they do not receive meaningless occupied/free evidence loss.

Validation reports `observed_free_false_occupied_rate`, the free-space error
outside the tolerance band, exact and tolerance-aware surface recall, and
`observed_direct_balanced_accuracy` separately from
`guessed_occupied_iou/precision/recall/f1`. It also reports the all-occupied
guessed-IoU baseline and the model's gain over that baseline. Global occupied
IoU is retained only as a summary diagnostic because it mixes surface
detection and area completion.

The present small-scale validation decoder still predicts at 64×64 and 80×80
before bilinear output resizing. Its effective detail is therefore about
10–12.5 cm, even though the output rasters are 512×512 and 800×800. That is a
current experiment setting, not a permanent pipeline limit: the planned
scale-up increases both dataset size and latent resolution, and the
resolution-derived tolerance follows it automatically.

Curriculum stages are selected with `training.stage`:

- `scale_only`
- `bev_only`
- `joint`

VGGT remains frozen in all stages.

## Frozen-teacher cache

`teacher_cache.mode` supports:

- `live`: exact frozen VGGT pass every time (default);
- `write_through`: use existing entries and atomically cache misses;
- `read`: require a complete cache.

Each entry contains selected VGGT tokens, depth, confidence, K/E, sample and
frame IDs, checkpoint hash, and preprocessing version. The cache is exact per
window. A full prefix cache can be very large, so it is never built
automatically.

## Metric reprojection

For a metric point `P_m`, convert it to the current window's VGGT units with:

```text
P_vggt = P_m / λ_hat
```

Metric translations are divided by the same scalar; rotations are unchanged.
The resulting point can be projected with same-window VGGT K/E. This use of
K/E is downstream and does not make BEV inference cascaded.

## Checkpoint contract

New checkpoints use:

```text
format_version = 17
pipeline_id = P1B
checkpoint_schema = p1b-fixed-metric-fov-complete-evidential-v6
```

They contain only the trainable P1B head state plus optimizer/scheduler and
provenance. The frozen VGGT checkpoint remains a private dependency inside
each deployment. Old v5 checkpoints and v6 checkpoints made with the former
`surface-hit-plus-guessed-completion-v1` loss are supervision-incompatible and
cannot be resumed. Start a fresh run for this loss revision.

## Commands

```bash
python -m pip install --no-deps -e .
odineye-p1b-freeze-split --config configs/p1b_nll_local_smoke.toml
odineye-p1b-train --config configs/p1b_nll_local_smoke.toml --data-only
odineye-p1b-train --config configs/p1b_nll_local_smoke.toml \
  --smoke-first-sample --max-train-steps 1 --skip-validation
```

These former-architecture configs and launchers are retained only under
`configs/legacy_p1b_previous_architecture/` and
`scripts/legacy_p1b_previous_architecture/`. They are not valid inputs to the
current `cli_train_p1b` entrypoint.
