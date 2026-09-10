# SHM navigation benchmark (v6)

This directory archives the code used for the seven-system, online navigation
benchmark stopped on 2026-09-10. It is **not** the earlier geometry-only test,
the M03 benchmark, or the M05+ training/visualization experiment.

The original deployment files in `snapshot/` are preserved byte-for-byte.
`DEPLOYMENT_METADATA.json` records the accepted core hashes and observed package
versions. `SOURCE_MANIFEST.json` inventories every bundled snapshot file.
Packaging adds documentation and CPU-only verification; it does not modify
model inference, planning, motion, sampling or scoring.

## Competitors

| Result key | System | Execution interface |
|---|---|---|
| `our_model` | Accepted Single Baseline + Single Hard Merge (SHM) + BIT* | External path executor |
| `limo_tel` | LiMo-TEL | External executor preserving the native planar path |
| `limo_aug` | LiMo-AUG | External executor preserving the native planar path |
| `omnivla` | OmniVLA, original 120k checkpoint, PointGoal mode | Released velocity controller |
| `mbra_logonav` | MBRA-PG / LoGoNav implementation | Released velocity controller |
| `nomad` | NoMaD | Released ImageGoal / waypoint controller |
| `genie_samtp` | GeNie-SAMTP with its BEV polynomial planner | External path executor |

SHM uses the accepted Single Baseline at epoch 10 / step 14,320, not an untrained
Merged head. Up to ten planning RGB frames are fused latest-wins using predicted
VGGT relative extrinsics. This benchmark's SHM variant uses camera-height metric
calibration; it is distinct from the original fixed-extent, Scale-Token-only SHM
visualizer. `protocol.json` is authoritative. Legacy classes in shared source
files do not add competitors: the active list is exactly the seven above.

## Frozen experiment contract

- Planned: 500 scene groups, five shared routes per group, 2,500 routes total.
  HM3D, HSSD, MP3D and stored ProcTHOR-10K each have 125 planned groups.
- Random scene sampling with replacement. Each initialized scene serves all
  seven systems on the same five frozen routes. Routes are 10–25 m long.
- Online rendered-camera execution and fresh model replanning, not a geometric
  check of a single predicted route.
- Linear speed ceiling 1 m/s; angular speed ceiling 90 degrees/s. Body
  0.20 x 0.20 x 0.50 m; camera height 0.50 m; planning safety margin 0.05 m.
  Reverse motion and in-place turns are supported; no lateral translation.
- RGB 640 x 480, horizontal FOV 90 degrees, zero camera pitch. Goal tolerance
  0.20 m. External executor replanning prefix 0.50 m; prediction-risk handling
  can shorten execution in unknown space.
- All systems share the privileged same-floor global guide and exact pose for
  transforming the current local subgoal / anchoring the plan. Models do not
  receive GT BEV, GT depth, obstacles, future global route or trajectory history.
  SHM fusion does not use that GT pose; it uses predicted VGGT relative poses.
- NoMaD receives the current subgoal's ImageGoal. It is **not a strict equal-input
  PointGoal comparator**. Native-controller inference can overlap motion;
  external path methods follow their separate executor contract.
- All-failed routes remain in the primary denominator. They are not resampled
  to improve success. Interrupted, uncommitted work can resume; committed
  successes and failures are immutable.

The stopped run preserved 4,331 committed method executions and 605 routes
completed by all seven systems (121 full groups). It did not finish the planned
2,500 routes. Results, RGB, trajectories and scene assets are not in this code
archive; uploading the code does not restart that experiment.

## Source layout

- `snapshot/baseline_verifier/procthor_20k_benchmark/shmcamera2500_20260909_v6/`:
  active runtime, planner adapter, robot/executor contract, dispatch, logging,
  aggregation and regression tests. Start with `protocol.json`, `PROTOCOL.md`,
  `EXECUTOR.md`, `NATIVE_RUNTIME_NOTES.md` and `REPAIR_REPORT.md`.
- `snapshot/baseline_verifier/procthor_20k_benchmark/`: shared scene/GT graph
  worker and planner implementations, plus the mixed Habitat adapter.
- `snapshot/baseline_verifier/src/`: exact deployed model and preprocessing
  support. Use this snapshot for inference reproduction, not unrelated newer
  training code elsewhere on the branch.
- `snapshot/baseline_verifier/vendor/backbone/`: deployed VGGT-Omega code and
  its upstream license. No backbone weights are bundled.
- `snapshot/databuilder/`: simulator initialization, voxel/visibility geometry,
  and renderer-device safety helpers required by the benchmark scene adapters.

## CPU-only checks

With a Python environment containing NumPy, SciPy and Pillow:

```bash
python benchmark/verify_snapshot.py
PYTHON=/path/to/cpu-capable/python bash benchmark/run_cpu_tests.sh
```

These checks do not launch CUDA models, Habitat, Unity, training or collection.
The original deployed CPU suite contains 107 tests. CPU passing is not a claim
that all models succeed or that a new GPU deployment has been qualified.

## Runtime prerequisites and launch boundary

This is a faithful deployment snapshot, **not a one-command portable installer**.
The archived launch/repair scripts retain their original absolute deployment
paths. Do not run historical cleanup, repair, resize or resume scripts against
unrelated outputs. Configure a separate deployment and fresh output root first.
Do not relax the source/protocol fingerprints or GPU safety checks.

Four distinct Python environments were used for the scene worker, Habitat,
SHM and other baselines. Their observed core versions are recorded in
`DEPLOYMENT_METADATA.json`; this is an environment record, not a complete lock
file. OmniVLA also requires the deployed OpenVLA-OFT Transformers fork rather
than assuming the same-version upstream wheel is equivalent.

Provide the following separately, keeping their licenses and access conditions:

- HM3D, HSSD, MP3D scene assets; the stored ProcTHOR-10K train houses and
  AI2-THOR engine/assets (engine build `ca10d107fb46cb051dba99af484181fda9947a28`).
- Accepted Single head: `p1b_stage1_36k_surface_nll_epoch10_step14320.pt`,
  SHA-256 `beab9448c77d5cac1b4ecccba690c4cd377ae282c9296e374da5b309bb1c654b`.
- Frozen VGGT checkpoint: `vggt_omega_1b_512_model.pt`, SHA-256
  `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.
- Baseline assets under the runtime's `--assets-root`: `src/less-is-more`,
  `src/OmniVLA`, `src/navigation-model-zoo/MBRA_PG_Official`,
  `src/visualnav-transformer`, `src/diffusion_policy`, `src/GENIE-SAMTP`;
  their corresponding complete released weights and local DINOv2 hub source.
  `navigation_baseline_runtime.py` specifies the exact relative file names.

External source revisions, when recoverable, and observed entrypoint/tree
hashes are recorded in `DEPLOYMENT_METADATA.json`. A null revision means an
extracted deployment without retained Git metadata, not permission to substitute
an arbitrary current upstream revision. External baseline repositories and
weights are not redistributed by this archive.

On a configured GPU host, qualify the executor and seven model interfaces
before using the archived `launch.sh GPU_INDEX LANE_INDEX`. Keep one independent
lane per assigned GPU, a consistent `REALTIME_LANES` value, detached logs and
the existing UUID, startup serialization and fatal-error guards. The launcher
deliberately refuses to run without `EXECUTOR_SIMULATOR_ACCEPTED.json`,
`MODEL_INTERFACE_ACCEPTED.json` and their real referenced evidence. Those live
authorization/acceptance files are not fabricated or shipped as portable proofs.

## Aggregation and metric interpretation

`summary.py --help` describes the read-only aggregator. It checks the source and
protocol fingerprints before reading result directories. The main comparison
must use the same fully compared routes for every system, not partially
completed per-method populations.

- Primary SR: successes / all fully compared routes, including all-failed ones.
- Conditional SR: successes / routes where at least one method succeeded;
  label this secondary, filtered denominator explicitly.
- SPL and raw path-length ratios use the conservative discrete GT graph route,
  not a continuous theoretical optimum. Raw successful-route ratios are not
  clamped and can fall below one because of graph discretization and the goal
  tolerance. Each model's own-success mean uses a different sample set.
- Record total wall time, request/inference/planning/render/motion/turn time,
  initialization, warmup, collisions, planner failures and timeout/budget exits.
  Overlapping timing components are not additive partitions of wall time.
- Compare completion speed on pairwise common successes; fast failures are not
  evidence of faster successful navigation. Report dataset source breakdowns.
- No held-out/generalization claim is made without a separate scene-overlap audit.

## Publication scope

This branch adds code and reproducibility metadata only. It does not contain
passwords, tokens, SSH keys, model checkpoints, datasets, raw benchmark outputs,
runtime caches or local operator notes. Existing training files on the P1B base
are not changed. VGGT-Omega remains subject to its upstream FAIR Noncommercial
Research License; see its bundled license and upstream project documentation.
