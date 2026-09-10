# Adapter v6 repair and user parameter update — 2026-09-09

## Current settings

- Body: **0.20×0.20×0.50 m**; camera height0.50 m.
- Algorithm safety margin: **0.05 m per side**, distinct from physical collision.
- Translation ceiling1 m/s; angular ceiling **90°/s**; reverse and in-place turns.
- BIT* body-circle plus margin radius: hypot(.10,.10)+.05 = **0.191421 m**.
- GeNie native footprint is computed in metric units after its240px resizing:
  padded width0.30 m, rounded outward to native footprint18px for this projection.
- Native velocity models do not gain an added external path follower. All use
  the same actual collision body. Weights and success radius0.20 m unchanged.

## Integration repairs carried forward

Reverse-aware exact XY execution avoids huge turns for centimetre reversals.
LiMo SE(2) yaw is retained as a tangent-branch reference (XY exact, yaw projected
when inconsistent). BIT* uses the exact metric origin on even grids, restored
predicted-only connected local targets and bounded search retry, and at most
0.20 m unknown travel before new observation. NoMaD calibration0.05 m is separate
from speed ceilings; its outer v/w limiter preserves curvature. MBRA's native
internal clipping remains. No GT obstacle/route correction enters local inference.

## Newly found FOV-boundary defect

Saved-RGB replay showed9 near-origin occupied pixels along the predicted FOV
edge in each of two cases. Their occupied probabilities were only0.50–0.75.
They were present in Single before fusion, then treated as hard obstacles and
expanded around a free origin. The old seam protection affected displayed GATE
but not the numerical occupancy/support used by BIT*.

At the failing frame, nearest predicted occupied cells were0.165 m (ProcTHOR)
and0.184 m (HM3D). The corresponding GT body-height-band boundaries, read ONLY
for diagnosis, were0.724 m and0.238 m. Different height semantics/model error
remain possible; this is not proof every model obstacle is false.

v6 applies the existing2-pixel predicted-support inset consistently to numerical
occupancy/confidence/ownership as well:
- newest trusted interior overwrites;
- a new untrusted border does not destroy older trusted evidence;
- border without older trusted evidence stays **unknown**, not free;
- interior obstacles retain hard occupancy; no GT-based carving or threshold tuning.

This is explicit confidence-domain preprocessing in our SHM adapter, not a
change to model weights or a claim of exact model predictions. With the saved
Single outputs, nearest trusted occupied cells become0.694 m /0.742 m, without
using GT to select or relabel them. Unknown still carries cost and bounded travel.

## Verification and limitations

87 CPU tests passed. New-body live execution:10/10 routes (5 ProcTHOR +5 Habitat)
completed with actual camera pose/yaw/height and changing RGB independently
verified. These GT instruction routes test the executor, NOT model success.

Seven-model live interface check completed:44 real RGB requests across two
backends. Our Model returned17/20 nonempty plans (HM3D10/10, ProcTHOR7/10);
the remaining3 were BIT* search-budget failures, retained as failures. None
returned the former origin-inflation error. The other six methods each returned
their required native output on all4 probes. No contact occurred during these
bounded interface movements; this is NOT full-route success/performance.
See repair_evidence/interfaces and verification_report/REPORT.md.
All qualification services exited and GPU0 was released; no owned v4/v5/v6
worker/model processes remain. No500×5 formal run restarted. Heartbeat remains
paused; foreign GPU6–7 jobs untouched.

The previous30 cm v5 reports remain in the sibling directory
`../shm_adapter_v5_20260909`; never quote them as20 cm results.
Reduced body/safety limits and support handling change the experimental protocol.
Full-route success rate and rankings must be measured anew; no claim all model
failures are fixed. Keep GPU isolation, Xid circuit breakers, locks and
qualification signatures intact.
