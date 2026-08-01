# OdinEye P1A

P1A is the cascaded geometry-conditioned Method I pipeline.

## Runtime contract

```text
RGB history + calibrated camera height (m)
   │
   ▼
frozen live VGGT-Ω (one shared forward)
   ├─ shared multi-scale tokens
   ├─ predicted depth + confidence
   └─ predicted intrinsics + temporal extrinsics
                         │
                         ▼
runtime geometry builder (no GT)
   ├─ latest-camera reference frame
   ├─ robust confidence-weighted ground plane/basis
   ├─ metre/VGGT-native isotropic scale
   ├─ fixed single/merged BEV grids
   └─ explicit last-FOV / historical-FOV-union support
                         │
shared tokens + explicit geometry condition
                         │
                         ▼
P1A direct BEV head
   ├─ internal single 512×512 tensor
   └─ internal merged 800×800 tensor
                         │
                         ▼
runtime support assembler
   ├─ final single: last-frame FOV-complete BEV
   └─ final merged: historical-FOV-union-complete BEV
       (outside support is forced unknown)
```

External runtime inputs are RGB history and calibrated camera height. Simulator
trajectory, GT depth, GT intrinsics and GT extrinsics are not runtime model
inputs. The geometry builder estimates the camera-to-ground distance in live
VGGT native units and computes one isotropic conversion factor from the
calibrated height. After that anchor, one P1A normalized-scale unit equals one
metre. The decoder cannot learn, predict or override either grid extent.

Both grids are latest-ego centred with forward image-up, matching the current
labels. Single x/z bounds are `[-3.25, 3.25]`; merged x/z bounds are
`[-5, 5]`, in anchored normalized-scale units. Failed/low-quality ground-scale
estimates are exposed through `geometry_valid` and `geometry_quality`.

The decoder still computes full `512²`/`800²` internal tensors. These are not
the final semantic contract. `fov_complete_semantic` is the deployable Model-B
output and `masked_observed_semantic` is the deployable Model-A output:

```text
final(p) = head(p), if runtime_fov_support(p)
         = unknown, otherwise
```

Model B also exposes occupied/free Beta evidence, occupancy probability,
epistemic uncertainty and `navigation_confidence`. Confidence is derived from
evidence correctness calibration, then gated by runtime support and geometry
quality; it is not produced by a sigmoid confidence head.

## Training contract

Existing 6.5 m single and 10 m merged labels are reused; no recollection is
required. Training constructs FOV supervision online:

- single support is the last-frame simulator camera FOV;
- merged support is the union of at most 10 historical FOVs transformed into
  the latest-ego frame;
- complete GT inside support supplies occupancy;
- masked GT partitions it into observed free space, sparse observed obstacle
  surfaces and occluded/unobserved guessed completion;
- outside support stays unknown and has no occupancy loss.

Observed free space and obstacle surfaces use separate, task-balanced losses;
surface hits use a metric tolerance. Guessed completion uses class-balanced
evidential NLL plus occupied/free overlap loss. Guessed supervision ramps in
after warm-up. Direct-priority PCGrad prevents guessed completion gradients
from opposing or overwhelming direct-observation gradients.

Evidence is calibrated to prediction correctness in both regions and is
regularized so observed content is normally more confident than guessed
content. This is an internally learned evidence distribution, not a separate
confidence output head.

Camera height follows the exact runtime path into the geometry builder. GT
camera centres remain diagnostics only. Invalid geometry causes a synchronized
training-batch skip and an all-unknown runtime semantic output.

One frozen live VGGT forward is shared by the enabled trainable paths.
Checkpoint schema is `p1a-fov-complete-confidence-v3`, format 14. Older P1A
checkpoints and pre-v3 manifests are intentionally incompatible.

## Independent layout

```text
VGGTBEV_Method1/
├── src/vggt_bev_method1/
├── configs/p1a_*.toml
├── vendor/backbone/       # private VGGT-Ω source
├── checkpoints/model.pt   # private frozen checkpoint
├── manifests/
├── runs/
└── tests/
```

The pipeline must not import another P1 project or use another P1 project's
backbone/checkpoint directory. The dataset may remain a common read-only input.

## Commands

```bash
python -m pip install --no-deps -e .
odineye-p1a-freeze-split --config configs/p1a_local_smoke.toml
odineye-p1a-train --config configs/p1a_local_smoke.toml --data-only
odineye-p1a-train --config configs/p1a_local_smoke.toml \
  --smoke-first-sample --max-train-steps 1 --skip-validation
```

Remote entry configs:

- `configs/p1a_remote_gpu_a.toml`
- `configs/p1a_remote_gpu.toml`

GPU work on `remote_gpu` must go through Slurm. Do not launch formal training
until dataset, manifest, scheduler and GPU preflight all pass.
