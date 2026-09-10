# Exact bounded camera executor v6

Limits: 1 m/s, **90°/s**, camera 0.50 m, oriented 0.20×0.20×0.50 m body,
20 ms maximum tick. This is an ideal bounded executor, not a dynamics model.

External execution is ONLY for Our Model, LiMo-TEL/AUG and GeNie-SAMTP.
OmniVLA/MBRA/NoMaD use native commands and analytic constant-v/w arc integration.

## Planar path execution

Input metric [right, forward] is anchored at stationary RGB exposure. All native
XY vertices are preserved, including centimetre-scale reversals. The connector
from the origin is traversed, not teleported. No smoothing/downsampling/GT repair.

v5 allows reverse. LiMo native left-positive SE(2) yaw selects the forward/reverse
tangent branch. XY-only paths use two-state minimum-total-turn gear selection;
no map, goal or GT input to this operation. At a corner stop translation, turn
at ≤90°/s, then follow the exact segment at ≤1 m/s. No lateral translation.
Learned XY/yaw may be inconsistent: **XY is exact, intermediate yaw is projected**,
not claimed to follow arbitrary native SE(2) poses exactly.

Prefix ≤0.50 m arc length before fresh RGB/replanning. Our Model additionally
caps predicted-unknown travel at 0.20 m. Inference occurs while stationary and
counts as time. Empty external plans allow up to six bounded fresh-view turns,
using only the current local target: declared adapter, not paper-native recovery.

## Collision, timing, verification

Only the currently executed cuboid sweep is collision-tested, including reverse
and in-place turning. No future-GT-path veto. Physical body and 0.05 m planning
margin are distinct. Signed velocity records reverse; distance is positive arc.
Unmeasured tracking error is null, never an invented zero.

`test_adapter_v5.py` covers reverse jitter, SE(2) reference, rear collision,
90°/s bounds, exact metric origin and prediction-only navigation.
`replay_adapter_audit.py` compares stored XY prefixes at 30°/s, 90°/s, and
90°/s+reverse. It is not a simulator/model rerun or a success-rate measurement.

`executor_scene_smoke.py` executes five GT instruction routes per backend,
independently checks actual renderer position/yaw, changing RGB, camera height,
endpoint and cross-track error. These trials NEVER enter model results.
`build_executor_report.py` rejects obsolete robot-contract evidence.
