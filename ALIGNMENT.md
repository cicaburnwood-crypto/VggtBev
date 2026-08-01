# P1A alignment invariants

1. External runtime inputs are RGB history and calibrated camera height in
   metres.
2. Depth, confidence, intrinsics and extrinsics come from the same live VGGT
   forward that produces the shared tokens.
3. The latest camera is the BEV reference; image-up means ego forward.
4. Camera height divided by the predicted camera-to-ground distance produces
   one isotropic metre/VGGT-native scale; x/y/z never receive separate scales.
5. The output grid is fixed: single is 512×512 over 6.5×6.5 and merged is
   800×800 over 10×10 anchored normalized-scale units.
6. One anchored normalized-scale unit equals one metre. The decoder receives
   the fixed extents and cannot predict or override them.
7. GT trajectory is a training-only scale-validation oracle. It does not enter
   the model, set an extent or resample a label.
8. Runtime uses no GT and reports scale/ground quality plus a validity flag.
9. Training BEV losses are weighted by the same runtime geometry-quality
   signal; GT trajectory is not part of that runtime quality input.
10. Both grids use the latest ego as centre and forward as image-up.
11. Checkpoints record `pipeline_id=P1A`,
    `checkpoint_schema=p1a-fixed-anchor-v2`, and format 13.
