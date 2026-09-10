# Native interfaces and declared adapters, v6 physical contract

Weights/architectures unchanged. Do not call every deployment operation
unmodified official code.

| Method | Execution |
|---|---|
| Our Model | Accepted Single → SHM → BIT*; external executor |
| LiMo-TEL/AUG | Released 50-point SE(2); XY + native yaw retained; external |
| GeNie-SAMTP | Released SAM-TP projection/polynomial planner; external |
| OmniVLA | Full existing model, waypoint4 PD and native joint limiter |
| MBRA/LoGoNav | Existing inference_vw/stateful PD, native commands |
| NoMaD | Existing full diffusion model, waypoint2 PD, native commands |

Launch: `--robot-max-v 1 --robot-max-w 1.5707963267948966 --nomad-metric-spacing 0.05`.
These are speed CEILINGS, not forced minimum speeds.

NoMaD calibration is explicitly 0.05 m, independent of actuator limits.
Previously raising max_v from 0.2 to 1 enlarged paths fivefold through max_v/4.
v5 retains the previously used 0.2/4 calibration. It remains a documented
deployment assumption, not proof of optimal calibration for every robot.

For NoMaD/MBRA, actuator output is
`s=max(1,abs(v)/v_max,abs(w)/w_max); (v_cmd,w_cmd)=(v/s,w/s)`.
This preserves commanded curvature, never accelerates commands, and reads no
map or goal. This limiter is a v5 adapter. Omni native coupled limits retained.

MBRA's own internal PD clipping is retained as native behavior; the added outer
limiter does not undo it and is usually a no-op after those native limits.

LiMo's SE(2) heading is no longer dropped. It chooses the legal tangent branch.
XY is exact; inconsistent intermediate native yaw is projected. Do not claim
this external execution is a published low-level LiMo controller.

Requested inference rates: Omni 3 Hz, MBRA 5 Hz, NoMaD 4 Hz.
NoMaD publication 9 Hz, expiry 1 s. Achieved rates measured. Omni and MBRA action
normalization unchanged. All route-local state resets between routes.

Native inference overlaps motion; old commands remain during latency, never
backdated. RGB and current target share the exposure pose. Stale-subgoal responses
are discarded. Missing/placeholder velocity is an interface error; zero is valid.
Native methods do not receive the external planner/recovery wrapper.

NoMaD's ImageGoal is explicitly not strict equal-input PointGoal. GeNie is the
released SAM-TP planner, not an unreleased full robot stack. This is a deployed
system comparison with heterogeneous native/external timing, not model-only
quality. Short interface acceptance does not establish long-route success.

## v6 physical parameters

All collision bodies are20×20×50 cm; safety margin5 cm, independent of body.
GeNie's native footprint parameter is recomputed using the actual resized BEV
metric cell size. For134×134 at0.03m resized to240, footprint18px covers the
0.30m padded width after outward rounding. This changes a native configuration
parameter only, not its polynomial planner. Runtime health records body/margin.
