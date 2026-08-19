# P1C: Geometry-Aware Merged BEV

P1C branches from P1B and changes only the multi-frame reasoning path. Frozen
VGGT still runs once per RGB window. Runtime remains RGB-only and does not
consume GT pose, depth, intrinsics, extrinsics, or camera height.

## Motivation

P1B Merged supervision uses GT planar camera poses to warp historical FOVs
into the latest ego frame, while its decoder receives only VGGT patch tokens.
P1C restores VGGT's camera-motion representation by exposing the frozen final
camera/register prefix tokens and supervising a small relative-pose head.

## Runtime graph

```text
RGB window
  -> frozen VGGT aggregator (one pass)
     -> patch tokens -----------------------------> Merged projector
     -> camera + register tokens -> SE(2) head ---^  + pose embedding
                                                     -> Merged decoder
                                                        -> FOV support
                                                        -> Observed Gate
                                                        -> later Guessed NLL
```

The pose head predicts, for every selected frame, the transform into the
latest ego frame:

```text
[tx_m, tz_m, sin(yaw), cos(yaw)]
```

The latest frame is constrained to `[0, 0, 0, 1]`. Three residual refinements
are composed as SE(2) transforms rather than added component-wise.

## Training-only pose target

Existing session metadata is sufficient; no data regeneration is required:

```text
T_latest_from_i = inverse(T_world_from_latest) @ T_world_from_i
```

GT pose enters only the loss. The model forward receives frozen VGGT tokens.
All refinement stages receive metric translation SmoothL1 and yaw cosine
supervision, with later refinements weighted more strongly.

## Stages

1. `pose_only`: train only the relative SE(2) head. Report translation and yaw
   MAE overall and by history length. It must beat the zero-motion baseline.
2. `fov_support_and_observed_gate`: train the pose head, pose embedding,
   Merged routing projector, and Merged routing decoder together. Single,
   Scale, Guessed, and VGGT remain frozen. This stage must warm-start only the
   verified Stage-1 pose head. Gate strength ramps in gradually; edge and
   contour weights also ramp from zero while their mass initially trains the
   filled interior.
3. After geometry is verified, train Guessed Evidential NLL as a separate
   stage. P1C does not alter the existing Guessed loss contract yet.

P1B checkpoints remain P1B artifacts. P1C uses a distinct checkpoint schema
and cannot be mistaken for a P1B Merged model.

Portable starting configs are `configs/p1c_pose_only_v1.toml` and
`configs/p1c_pose_fov_gate_v1.toml`.

## Diagnostic contract

Validation reports Support and Observed Gate precision/recall/IoU separately
for history lengths 1 through 10. P1C additionally reports metric translation
MAE and yaw MAE for every history bucket. A monotonic P1B degradation with
history length is evidence for cross-frame alignment as the bottleneck; it is
not assumed in advance.
