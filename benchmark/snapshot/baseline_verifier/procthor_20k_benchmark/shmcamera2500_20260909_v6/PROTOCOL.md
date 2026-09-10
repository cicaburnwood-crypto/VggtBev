# SHM native-or-external benchmark — adapter v6

Status: v4 stopped; v6 repairs and bounded qualification only.
`protocol.json` is authoritative. Formal restart requires current executor and
model-interface signatures. Never pool geometry-fast-forward, Single/v3, SHM/v4
and v5/v6 results.

## Models and information boundary

Our Model = accepted Single Baseline + SHM + BIT*, NOT M03.
Others: LiMo-TEL, LiMo-AUG, OmniVLA, MBRA/LoGoNav, NoMaD, GeNie-SAMTP.
Existing full checkpoints unchanged.

Actual rendered RGB and current metric local goal from a shared GT upper guide
are supplied. Shared localization transforms the goal and anchors execution;
local inference receives no future global route, GT BEV/depth/obstacles or GT
trajectory. Known fixed camera calibration is supplied when required.
NoMaD uses the current subgoal ImageGoal: explicitly not strict equal-input
PointGoal. GeNie is the released SAM-TP planner, not its unreleased robot stack.

SHM uses up to10 fresh planning RGB frames, route-local FIFO/latest-wins fusion,
predicted VGGT relative extrinsics, existing camera-height metric scale chain.
No GT pose enters fusion. BIT* consumes fused numeric occupancy/confidence/
support, never Single aliases.

## Sampling

500 groups: 125 each HM3D/HSSD/MP3D/stored ProcTHOR. Each scene initialization
provides five identical frozen routes to every method (2,500 routes/method).
Random with replacement, largest same-floor component, route length10–25 m,
GT upper subgoals every2 m, initial yaw jitter±20°. Max200 scene attempts/group,
2,000 path attempts/scene. No held-out claim without training-overlap audit.
All-model failures stay in the unconditional denominator; conditional rates
reported separately. No success-based sample deletion.

## Physical contract

| Parameter | Value |
|---|---|
| Linear speed ceiling | 1 m/s, reverse allowed, not forced minimum |
| Angular speed ceiling | **90°/s = π/2 rad/s** |
| Camera height / pitch | 0.50 m / 0° |
| RGB / horizontal FOV | 640×480 / 90° |
| Oriented collision cuboid | 0.20×0.20×0.50 m |
| Tick / planning safety margin | ≤20 ms / 0.05 m |
| Goal acceptance | 0.20 m, evaluator-only short wall-barrier check |

External executor ONLY: Our Model, LiMo-TEL/AUG, GeNie.
Native commands ONLY: OmniVLA, MBRA, NoMaD. Preserve native XY vertices;
no smoothing, corner skipping, teleportation or GT path correction.
LiMo native yaw selects forward/reverse tangent branch: XY exact, yaw projected
where nonholonomically inconsistent. XY-only gear choice minimizes total turns.
External prefix ≤0.50 m, then fresh RGB and replan; stationary inference counts.
Empty external plans allow ≤6 bounded fresh-view turns using the current target,
not GT free-space. This is a disclosed wrapper, not paper-native recovery.

Native controllers/actuator adapters are detailed in NATIVE_RUNTIME_NOTES.md.
v5 NoMaD/MBRA saturation preserves v/w curvature; NoMaD action calibration0.05 m
is separate from the1 m/s ceiling. Native requested inference rates3/5/4 Hz for
Omni/MBRA/NoMaD; NoMaD publishes9 Hz with1 s expiry. Achieved Hz measured.
Native latency overlaps old command execution; no retroactive control update.
Stale-subgoal responses discarded.

## Our prediction-only adapter

Connected-component local-goal repair handles blocked/out-of-map guide targets.
Unknown costly but traversable, ≤0.20 m unknown travel/plan. BIT*0.55 s initial
budget and one1.65 s extended attempt. Exact continuous metric origin on even
grids, never clear occupied starts; continuous predicted-obstacle edge checks.
Backend/history resets between routes. No alternate planner or GT fallback.

## Metrics / safety

Only the current executed cuboid tick/arc is collision-tested, including reverse
and rotation, never a future GT path veto. Common wall budget
`max(300,20*GT_reference_m+60)`; distance budget `3*GT_reference_m+5`.
These remain explicit failure outcomes.

Record success, actual path length, endpoint distance, raw and adjusted ratios,
SPL, subgoal progress, replans/Hz, inference/render/translation/rotation/idle/
warmup/drain durations and total monotonic wall time. The GT reference is a
conservative discrete graph, not exact continuous optimum. Do not clamp ratios.
Compare timing on paired successes and separately report failures.
Scene initialization amortized across methods/five routes and reported separately.
This compares deployed systems with heterogeneous controllers, not just models.

Named tmux,10-second idle-GPU check, UUID/minor isolation, owned-process cleanup,
Xid latch and code-bound acceptance remain mandatory. No Slurm for this workload.
No GPU resets, foreign-job interference, lock weakening, forged signatures or
automatic formal restart. CPU/interface probes do not prove full-route success.

## v6 support and size override

Trusted per-frame FOV support is eroded by2 pixels before numerical SHM ownership
updates. Borders without old trusted evidence remain unknown, never free. New
untrusted borders cannot overwrite historical trusted occupancy/confidence.
Interior remains latest-wins. Raw Single output is preserved for inspection.
BIT* radius=0.191421 m (20cm body circumradius +5cm); GeNie footprint uses
physical width plus5cm per side after resizing, outward-rounded in native pixels.
These are explicit adapter parameters; signatures differ from v5.
