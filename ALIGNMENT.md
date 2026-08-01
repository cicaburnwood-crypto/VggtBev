# P1B FOV-complete alignment invariants

1. RGB windows are the only runtime inputs.
2. Frozen VGGT aggregator tokens are the only P1B-head inputs.
3. BEV and Scale Token execute in parallel and never read or wait for
   VGGT K/E/depth/confidence.
4. Single BEV is fixed 6.5 m × 6.5 m at 512 × 512; merged BEV is fixed
   10 m × 10 m at 800 × 800. Both are latest-ego-centred metric grids.
5. The merged decoder reads all RGB-window tokens directly; it does not take
   predicted single BEVs as input.
6. The head predicts FOV support plus binary `occupied/free` content inside
   that support. Outside predicted support is `unknown`. Confidence and
   epistemic uncertainty are derived from occupied/free Beta evidence.
7. Scale is exactly `lambda = metre / VGGT runtime unit` and is isotropic.
8. There is no canonical scale, learned extent, ground token, camera-height
   input, or path head.
9. GT metric z-depth and live/cached frozen-VGGT depth construct scale labels
   only after the parallel head prediction.
10. The loader generates GT FOV support online from horizontal camera FOV and
    GT planar poses. It never uses depth visibility, so occluded cells inside
    the geometric FOV remain in the complete target.
11. Single support is the latest-frame footprint. Merged support is the union
    of every window footprint transformed into the latest ego frame.
12. Complete BEVs provide content and masked BEVs only partition directly
    visible versus occluded/inferred cells inside support.
13. Directly observed free ray space and sparse obstacle surface hits are
    independent pointwise tasks with task-level balancing. Observed surface
    hits never receive occupied-area Dice. Surface matching uses half one
    latent cell of tolerance and observed-free supervision excludes the same
    band; the pixel radius is derived from output/latent resolution.
14. Macro occupied/free Dice and class-weighted completion NLL apply only to
    occluded/inferred cells. Their optimizer-step curriculum ramps
    independently without reducing the fixed direct-observation coefficient.
    Direct-priority PCGrad projects only conflicting completion gradients and
    never changes the forward graph or runs VGGT twice.
15. Scene splits are sequence/scene grouped. Manifest format 6 also
    fingerprints camera intrinsics and per-frame planar extrinsics because
    they generate the online target.
16. Runtime checkpoints use
    `p1b-fixed-metric-fov-complete-evidential-v6`, format 17, with loss schema
    `tolerant-surface-macro-completion-pcgrad-v2`. v5 and former v6 loss-v1
    checkpoints cannot resume.
