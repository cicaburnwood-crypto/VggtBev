# M05+ — DPT-lite, Role-separated Tokens, Per-cell Temporal Attention

M05+ is a separate experiment. It does not replace M05 and cannot load an M05
checkpoint as a strict resume checkpoint.

## Runtime contract

```text
RGB window (1..10 frames)
  -> reverse only at the frozen-VGGT boundary (latest -> oldest)
  -> frozen VGGT aggregator (one pass)
  -> split every 2048D patch token into local/global 1024D streams
  -> independent local/global projection and DPT-lite top-down fusion
  -> Camera/Register token contextualization without pooling
  -> latest-frame 256x256 anchor
  -> shared historical proposal decoder, once per historical frame
     (every BEV cell reads Camera + 16 Register slots by content attention)
  -> per-cell softmax over [no update, newest history, ..., oldest history]
  -> shared spatial refinement
     -> routing refinement -> FOV support + observed gate
     -> evidence refinement -> Beta occupancy evidence
  -> 10m x 10m, 256x256 merged BEV
```

Each output cell independently selects useful temporal evidence. The learned
null candidate preserves the latest anchor when every historical proposal is
harmful. This is internal content attention: runtime still consumes RGB only
and does not use extrinsics, camera height, GT, Single BEV, warping, explicit
geometry, FiLM-style modulation, or external fusion.

The prefix-token path is part of the BEV head and receives gradients only from
the BEV objectives. It has no pose/depth/calibration output, geometry target,
auxiliary geometry loss, standalone optimizer, or separately trained geometry
head.

The metric-scale branch remains independently trainable for later diagnostics.
It is not an input to the BEV head and is disabled in default inference.

The production token contract is exact: every frame must expose one camera
token followed by sixteen register tokens. Cached patch layers must agree with
that prefix tensor on batch, frame count, channel width, and patch-grid size.
Contract violations fail before decoding instead of silently changing token
semantics.

## Patch-token fusion

At each cached VGGT layer the released aggregator returns a concatenation of
the 1024D within-frame state and the 1024D inter-frame state. M05+ v3 splits
those halves before normalization. Each half is projected to 48 channels and
kept as an independent stream through a deep-to-shallow DPT-lite/FPN path.
Only the fused output of each pyramid level is reduced to the 96D BEV decoder
width. Consequently the high-resolution pyramid levels contain information
from every configured VGGT depth instead of being independent bilinear resizes.

## Camera/Register conditioning

The Camera token and Register bank have separate input projections. All 17
slots retain their identities through the ordered cross-frame content-attention
trunk. The trunk returns one Camera sequence plus sixteen Register sequences;
it never pools them into one frame vector. Each BEV cell then reads the Camera
and Register memory for the relevant frame, so special-token reduction happens
per cell rather than through one spatially uniform broadcast vector.

## Capacity change from M05

| Component | M05 | M05+ |
|---|---:|---:|
| BEV state width | 64 | 96 |
| Per-cell learned query width | 64 | 64, projected to 96 |
| Latest decoder layers | 2 | 3 |
| Historical proposal layers | 1 sequential update | 2 shared proposal layers |
| Patch fusion | independent 2048->64 projection | local/global 48D streams + DPT-lite/FPN |
| Prefix context | pooled implicit geometry | unpooled Camera/Register, 1024 / 8 layers / 16 heads |
| Spatial refinement | 1 shared | 2 shared + 1 routing + 1 evidence |

Keeping the dense BEV state at 96 and query content at 64 avoids an unnecessary
dense activation explosion. Prefix capacity remains concentrated in the
short 170-token maximum sequence, but it is trained only as a content path for
the end-to-end BEV objective and is not interpreted or supervised as geometry.

## Temporal initialization invariants

- Historical proposal MLPs are bias-free and zero preserving.
- Their final projections start at exactly zero, so a fresh M05+ is precisely
  the latest anchor rather than a randomly perturbed merge.
- Proposal output is `proposal - anchor`; normalized and unnormalized decoder
  states are never subtracted.
- The Null route starts at probability `0.90`. Its logit contains `log(H)` for
  `H` historical frames, so that prior is independent of window length. The
  learned Null scorer starts as a zero residual around that explicit prior.

## Training contract

The fixed 10m x 10m GT, dataset loader, evidential occupancy formulation,
routing targets, and Scale supervision remain inherited from M05. The frozen
512x512 categorical GT is reduced to the native 256x256 training/output grid
with nearest-neighbor sampling. M05+'s branch combination is intentionally
corrected to make Merged the primary objective:

```text
L_BEV = 1.0 * L_merged + 0.10 * L_latest_auxiliary
```

The former convex mix assigned only half of the Merged gradient to temporal-only
parameters. The additive form preserves the full Merged gradient while keeping
latest-frame supervision as a weak training-only regularizer. Historical M05
retains its original convex-loss behavior.

The pipeline ID is shared by configuration, model runtime, and checkpoint
metadata. The incompatible architecture uses checkpoint schema
`m05plus-dpt-role-token-per-query-temporal-fixed-metric-512-v3` and must start
fresh. V1/v2 checkpoints cannot strict-resume v3.

## Current execution path

The production A100 path keeps the model, outputs, and primary losses intact
while removing redundant execution:

- All historical proposals are evaluated with the same shared decoder in
  batches of up to nine frames instead of nine Python-level decoder calls.
- Per-cell temporal logits and proposal fusion are tensorized across history.
- Latest and Merged states share one batched spatial decode whenever the
  training-only Latest auxiliary branch is active.
- Cross-attention uses one 65,536-query chunk. Attention rematerialization is
  disabled because the 256x256 grid fits in A100 memory, avoiding recomputation.
- Spatial activations use channels-last layout; cuDNN benchmarking, BF16
  autocast, fused AdamW, pinned persistent workers, and four-batch prefetching
  are enabled.
- DDP unused-parameter traversal is disabled. Zero-valued graph dependencies
  retain every conditional parameter in the graph for one-frame samples.

The Latest auxiliary objective is evaluated on every step for the first 20% of
training. It is then evaluated every fourth step at four times its nominal
multiplier, preserving its expected `0.10` contribution while avoiding three
of four auxiliary decodes. History caps `[3, 5, 8, 10, 10, 10, 10, 10, 10, 10]`
progressively expose longer windows over the ten epochs.

## Frozen 8×A100 run

The checked-in production profile is
`configs/m05_plus_a100_8gpu_10e_256_frozen.toml`:

| Item | Frozen value |
|---|---:|
| Epochs | 10 |
| Native BEV output | 256 x 256 over 10m x 10m |
| A100 GPUs | 8 |
| Batch per GPU / global batch | 1 / 8 |
| Optimizer | fused AdamW |
| LR / minimum LR / weight decay | `1e-4` / `1e-5` / `0.02` |
| Warmup | 3% |
| Gradient clip | `1.0`, separately for BEV and Scale |
| Validation batches | 500 |
| Checkpoint interval | 5,000 steps |
| Total optimizer steps | 669,140 |

The valid frozen union contains 561,533 sessions across 1,076 scenes:

| Source | Sessions |
|---|---:|
| Pro6000 (`medical_6000`) | 291,041 |
| 5090 (`super_5090_boq06`) | 157,459 |
| Industrial A100 | 113,033 |
| **Total** | **561,533** |

The grouped split contains 535,307 training sessions and 26,226 validation
sessions. Its content SHA-256 is
`7cce63f1592d696266aadc2f81a9c97637c169c943dafb305156e1c98cf482b3`.
One 5090 directory with a stale `COMPLETE` marker but no `metadata.json` is
recorded as invalid and excluded. The 571 MiB session-record cache binds to the
same manifest hash so eight ranks do not independently rescan the dataset.

Production entry point:

```bash
scripts/launch_m05_plus_a100_8gpu.sh
```

The launcher uses the existing `openpi` environment, eight local ranks, static
localhost rendezvous at `127.0.0.1:29505`, and validates the cache/manifest hash
before training. The data-specific manifest and cache are server artifacts and
are intentionally not stored in Git.

Generic entry point:

```bash
GPU_COUNT=4 scripts/train_m05_plus.sh \
  configs/m05_plus_temporal_attention_10m_template.toml
```
