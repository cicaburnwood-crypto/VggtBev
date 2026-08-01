# VGGTBEV

An independent training project for the PDF's **Method II** observed-area BEV head. It reads
the VGGNAV simulator export and the VGGT-Omega checkpoint without modifying either source
project.

## Scope and output contract

This stage intentionally drops revealed/unobserved completion. The model has two outputs:

- `observed_logit`: whether a BEV cell is camera-observed or unknown.
- `occupancy_logit`: whether a cell is occupied, supervised **only on observed cells**.

At inference, the logits render to the existing lossless convention: occupied `0`, unknown
`112`, and free `255`. `complete` and `merged_complete` are excluded from every training batch.
The validator can read them through a deliberately named `revealed_labels_audit_only` field.

The supported aligned tasks are:

| Task | RGB input | Training target source | Active P2 model output |
|---|---|---|---|
| `single` | frame `t` | `bev_6p5m/masked/frame_t` | learned normalized range, 512×512 |
| `merged` | every frame `0..t` | 10 m merged masked label, nearest-resized | learned normalized range, 800×800 |
| `both` | every frame `0..t` | both targets above | both pixel-only products |

In `both` mode, the frozen VGGT extraction is shared. A single head splats only the latest
frame into the selected grid, and a separate merged head splats every frame into the configured
merged grid. The merged export is cumulative, so a truncated sliding window is not accepted as
an equivalent input: its evidence would not match the provided label history.

## Method II implementation

```text
RGB sequence
    │
    ▼
frozen VGGT-Omega ──► patch features + dense depth + confidence
    │
    ▼
VGGT-predicted backprojection (camera-head K + relative poses)
    │
    ▼
robust fixed-K estimate + latest-camera reconstruction
    │
    ▼
ground-plane fit → raw VGGT reference-radius normalization
    │
    ▼
integrated learned per-sequence zoom (no configured output extent)
    │
    ├──► confidence-weighted bilinear feature splat
    └──► explicit free-space ray evidence
                    │
                    ▼
             small 2D decoder
                    │
                    ├──► observed logit
                    └──► occupied logit on observed cells
```

The VGGT backbone is frozen for this stage. The trainable components are the point encoder
and BEV decoder. Training and runtime use the same geometry source: VGGT dense depth plus
camera-head intrinsics and relative poses estimated from RGB. The P2 configuration takes a
robust sequence-level median of the predicted intrinsics and fits the ground after
transforming points into the latest camera frame. Ground fitting removes tilt and places
the floor at zero. A per-sequence VGGT reference radius makes the coordinates
dimensionless, and an integrated trainable normalizer predicts their output zoom. The
6.5 m and 10 m values identify supervision sources only; they are never supplied as model
grid extents in the active normalized P2 mode.

Simulator intrinsics, camera extrinsics, and GT trajectory are excluded from every training
batch. They remain dataset-side label-construction and alignment-audit information only.
This prevents the head from training on perfect geometry that will not exist at runtime.

## Project layout

- `src/vggt_bev/data`: session discovery, strict synchronization, calibrated resize, batching.
- `src/vggt_bev/geometry`: coordinate conversions, lifting, bilinear splatting, ray evidence.
- `src/vggt_bev/models`: read-only VGGT adapter and the trainable Method II head.
- `src/vggt_bev/runtime.py`: checkpoint loading, label-free sequence input, and output export.
- `src/vggt_bev/runtime_server.py`: stateful HTTP bridge for cumulative live inference.
- `src/vggt_bev/losses.py`: observation loss plus occupancy and Dice losses masked by visibility.
- `src/vggt_bev/cli_validate.py`: deterministic schema, label, history, and axis checks.
- `src/vggt_bev/cli_train.py`: session-level split, training, validation, and compact checkpoints.
- `src/vggt_bev/cli_infer.py`: RGB-only runtime CLI for a VGGNAV camera history.
- `simulator_ui`: copied VGGNAV navigation UI plus synchronized GT/model comparison support.
- `scripts/run_simulator_compare.sh`: starts the model service and Habitat UI together.
- `scripts/run_random_simulator_compare.sh`: samples a valid scene, camera, start, and yaw.
- `models`: downloaded final dual-output heads and their SHA-256 manifest.
- `tests`: unit tests plus alignment tests against the supplied example export.

## Setup and checks

From `/home/user/Project/VGGTBEV`:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[dev]'
make check
```

`--system-site-packages` reuses the machine's CUDA-enabled PyTorch while keeping packages
installed by this project in its own `.venv`. The test target disables unrelated globally
registered pytest plugins.

Useful individual gates:

```bash
.venv/bin/vggt-bev-validate --target-mode single --extent-key bev_5m
.venv/bin/vggt-bev-validate --target-mode merged --extent-key bev_5m
.venv/bin/vggt-bev-train --config configs/method2_observed.toml --data-only
```

Legacy merged data may report small masked-versus-complete differences as an audit warning.
Visibility-fusion-v2 data treats any known-pixel difference as an error. The complete target
remains excluded from training.

## Runtime inference

Three final epoch-10 checkpoints are available:

| `--model` | Single output | Merged output | Local checkpoint |
|---|---:|---:|---|
| `3p5m` | 3.5 m, 512×512 | 8 m, 512×512 | `models/hm3d_600x3_method2_dual_bev_3p5m.pt` |
| `5m` | 5 m, 512×512 | 8 m, 512×512 | `models/hm3d_600x3_method2_dual_bev_5m.pt` |
| `6p5m` | 6.5 m, 512×512 | 8 m, 512×512 | `models/hm3d_600x3_method2_dual_bev_6p5m.pt` |

Run inference on frames `0..t` from a VGGNAV session:

```bash
.venv/bin/vggt-bev-infer \
  --model 5m \
  --session /home/user/Project/VGGNAV/output/random_sessions_test_3/session_000_00833-dHwjuKfkRUR \
  --target-frame -1 \
  --output-dir artifacts/runtime_5m
```

The active normalized runtime reads only `camera/frame_*.png`; it requires no
camera height. It does not read session metadata, GT trajectory, camera
intrinsics, simulator poses, or any BEV label. Both trained Method II heads and
the head-free geometry baseline use intrinsics and relative poses estimated by
VGGT. The CLI retains an optional height argument only for legacy metric
checkpoints.

The output directory contains:

- `single_masked.png`: current-frame observed BEV.
- `merged_masked.png`: cumulative observed BEV using frames `0..t`.
- `geometry_single.png`: head-free projection from VGGT-estimated depth and camera.
- `geometry_merged.png`: head-free projection using VGGT-estimated relative poses.
- `probabilities.npz`: occupancy and observation probabilities for both products.
- `runtime.json`: checkpoint hash, frame history, scale, grid geometry, and output paths.

All BEV PNGs use occupied `0`, unknown `112`, and free `255`. The merged training target is
cumulative from frame zero, so the CLI always loads a contiguous history beginning at frame
zero. Check downloaded model integrity with:

```bash
cd models
sha256sum -c SHA256SUMS
```

## Interactive simulator comparison

The UI has a live camera panel, point-and-click navigation map, and a three-row direct
comparison. Each row shows synchronized Simulator GT and Method II prediction:

| Physical area | Simulator GT | Method II prediction |
|---|---|---|
| 3.5 × 3.5 m | current-frame masked BEV | trained 3.5 m head |
| 5 × 5 m | current-frame masked BEV | trained 5 m head |
| 6.5 × 6.5 m | current-frame masked BEV | trained 6.5 m head |

The inference service loads all three legacy checkpoints but stores only one frozen VGGT
backbone. Each RGB sample is extracted by VGGT once, then the three independently trained
heads apply their checkpoint-compatible geometry mode and output a 512×512 grid for their
physical extent. The new scale-up P2 checkpoint instead emits a 512×512 single
grid and an 800×800 merged grid with a learned per-sequence normalized VGGT
range. Older downloaded checkpoints retain their legacy scale behavior for
compatibility.

The Habitat simulator remains in VGGNAV's existing Python 3.9 environment. The frozen VGGT
backbone and trained heads run in this project's PyTorch environment through a local HTTP
service. Neither original project is modified.

Pass a Habitat scene explicitly if the copied VGGNAV default scene drive is not mounted:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_simulator_compare.sh \
  5m \
  /path/to/scene.basis.glb \
  --navmesh /path/to/scene.navmesh
```

The launcher loads all three models regardless of its retained legacy model argument. Open
`http://127.0.0.1:8000`, click the global map for auto-path finding, or enable manual control
and use WASD.

Start a randomized comparison using the same simulator ranges as the collection pipeline:
camera height 0.30–0.80 m, horizontal FOV 60–120 degrees, one of the installed valid
Habitat test scenes, a random navigable point, and random yaw:

```bash
scripts/run_random_simulator_compare.sh 5m --no-browser
```

The launcher prints and exposes the sampled values through `/api/state`. Set
`VGGTBEV_RANDOM_SEED` to reproduce a session. The sampled physical camera height
configures Habitat. A normalized P2 checkpoint does not receive it; legacy
metric checkpoints may still consume it. FOV, pose, simulator intrinsics, and
simulator trajectory are not sent to the model server.

Set `VGGTBEV_SCENE_NAME=apartment_1` or `skokloster-castle` to force one installed
scene while keeping camera height, FOV, start, and yaw randomized.

Live sampling defaults to one inference per second and only accepts a new sample after the
agent's motion step changes. The RGB history resets after 34 samples. Useful overrides are:

```bash
VGGTBEV_MODEL_MAX_HISTORY=34 \
VGGTBEV_RUNTIME_PORT=8765 \
scripts/run_simulator_compare.sh 5m /path/to/scene.basis.glb \
  --model-hz 1.0 --port 8000 --no-browser
```

The runtime HTTP request contains camera RGB, segment/frame identifiers, and the output
threshold—no GT camera geometry. Method II uses VGGT-estimated intrinsics, depth, and poses.
The runtime server supplies the fixed mounting height. GT BEV and simulator poses stay
inside the comparison process for rendering; they are never sent to the model service.

## Training

Training configurations must set `data.geometry_source = "vggt"` (it is also the enforced
default). The collated model input contains RGB history, valid-pixel/frame masks, physical
camera mounting height, and BEV labels. It intentionally contains no simulator intrinsics,
camera-to-world poses, floor transform, or GT trajectory tensors.

Single-frame observed BEV:

```bash
.venv/bin/vggt-bev-train --config configs/method2_observed.toml
```

Cumulative multi-frame observed BEV:

```bash
.venv/bin/vggt-bev-train --config configs/method2_merged_observed.toml
```

P2 6.5 m single-frame plus 10 m merged training:

```bash
.venv/bin/vggt-bev-train \
  --config configs/remote_hm3d_p2_vggt_geometry_6p5m_10m.toml
```

For synchronized multi-GPU training, launch one process per visible GPU. Each
rank uses `training.batch_size_per_gpu` samples and DDP averages the head
gradients. For example, batch size four on three GPUs gives effective batch
size twelve:

```bash
CUDA_VISIBLE_DEVICES=6,7,8 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=3 \
  -m vggt_bev.cli_train \
  --config configs/remote_hm3d_p2_vggt_geometry_6p5m_10m.toml
```

Set `VGGTBEV_DISTRIBUTED_BACKEND=gloo` when NCCL is unavailable or unstable on
the host; model computation still runs on the assigned CUDA GPUs.

That configuration reads the actively growing schema-v4 collection at
`/data/project/VGGT_DATA/data_build_unlimited`. This is the authoritative
source-data root on `remote_gpu_a`; the previous disk-14T VGGNAV path is retired.
It expects:

```text
bev_6p5m/masked/
bev_6p5m/merged_masked_10m/
frame["bev"]["bev_6p5m"]["merged"]["10m"]["masked"]
```

It requires the aligned merged labels to declare a 10 m extent and
`merged_fusion_version = 2`, failing fast on older labels. Version 2 accumulates visibility
only; both merged products read occupancy from the same static collision truth. Migrated
sessions preserve the legacy directories and select corrected
`merged_masked_*_visibility_v2` labels through metadata. `split_group = "scene"` prevents
sessions from the same simulator scene
leaking across train and validation. `split_strategy = "stable_hash"` keeps existing
assignments stable as new sessions are appended to the unlimited collection.
At million-sample scale, `validate_paths_on_init = false` avoids repeating
millions of filesystem-stat calls independently on every DDP rank; the
streaming dataset audit remains the required path-integrity gate.

`model.vggt_execution = "live"` is mandatory. Every training forward runs the frozen
VGGT-Omega aggregator, dense-depth head, and camera head on the RGB history. Their estimated
depth, intrinsics, and relative poses are passed directly into the Method II lift. Cached
VGGT estimates, simulator depth arrays, simulator intrinsics/extrinsics, and GT trajectory
are rejected as model inputs. Only the masked BEVs are supervision targets.

The active non-metric P2 settings require
`metric_scale_mode = "vggt_normalized"`,
`data.coordinate_mode = "vggt_normalized"`,
`stabilize_intrinsics = true`, and `learn_depth_scale = false`. Camera height
and fixed extent tensors are absent from training and runtime batches. The source
labels remain the existing 6.5 m single and 10 m merged products, but they enter
the model only as 512×512 and 800×800 supervision images. Their physical extents
are therefore properties of the training distribution rather than hard runtime
limits. The integrated normalizer predicts one normalized VGGT unit-per-output-pixel
value for each sequence; the 800-pixel merged product consequently spans 800/512
times the normalized width of the single product at the same learned resolution.
Physical coverage can still vary between sequences. Free-space rays stop one cell
before the predicted surface.

Start with `--max-train-steps 1 --skip-validation` as a full-resolution live VGGT integration
smoke test. Use `--output-dir` to isolate its checkpoint from a real run. Batch size is fixed
at one for cumulative sequences because VGGT has no padding-mask input; this prevents padded
frames from contaminating inter-frame attention.

On a smaller GPU, the dedicated smoke command runs the complete forward, masked loss,
backward, and optimizer path with a 64×64 input while retaining the real checkpoint and label:

```bash
.venv/bin/vggt-bev-live-smoke
```

Checkpoints under `runs/` contain only the trainable heads, scale-mode state, optimizer, and
configuration. They do not duplicate the frozen 4.4 GB VGGT checkpoint.

## `remote_gpu` deployment

The P2 checkout is deployed at:

```text
/home/user/VGGT/method2_train
```

Its personal Conda environment is `vggtbev-p2`, and its active configuration is
`configs/remote_gpu_p2.toml`. The configuration expects:

```text
data:       /home/user/VGGT/databuilder/output/data_build_unlimited
VGGT code:  /home/user/VGGT/method2_train/backbone
checkpoint: /home/user/VGGT/method2_train/checkpoints/VGGT-Omega-1B-512/model.pt
runs:       /home/user/VGGT/method2_train/runs/p2_vggt_normalized
```

Before allocating a GPU, run the CPU-only readiness gate:

```bash
bash scripts/remote_gpu_preflight.sh
```

The gate checks the dataset manifest, exact 4,576,706,117-byte Omega checkpoint,
VGGT imports, installed environment, normalized coordinate mode, 512/800 output
sizes, and batch/GPU 1. It exits nonzero if any input is incomplete.

Training must never run directly on the login node. After the preflight passes,
submit through Slurm:

```bash
sbatch scripts/remote_gpu_p2.slurm
```

The default launcher requests one RTX 5090. For an explicitly approved multi-GPU
run, override the Slurm GPU request and export a matching `VGGTBEV_NPROC`; the
per-rank batch remains one until equal-history bucketing is implemented.

## Required gates before a long run

1. Run `make check` for every chosen extent key.
2. Run a one-step live VGGT smoke test on the intended GPU.
3. Overfit a tiny session subset and verify occupied/observed IoU increases.
4. Inspect rendered predictions for correct forward-up orientation and metric placement.
5. Only then scale the dataset and compare single versus cumulative training.

Revealed-map completion should be a later, separately gated objective or residual completion
head. It should not be enabled by changing the labels in this Method II pipeline.
