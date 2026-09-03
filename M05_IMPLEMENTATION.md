# M05: Latest-Anchored Reverse-Gated Merged BEV + Scale

M05 is a fresh successor to M04. It preserves the external task contract:
one ordered RGB window enters one frozen VGGT pass, and runtime returns one
512 x 512 Merged BEV in 6.5 VGGT units plus metres-per-VGGT Scale and
uncertainty. There is no pose, camera-height, GT, planner, or navigation input.

## Why M05 exists

M04 decoded the latest frame and all preceding frames in two parallel branches.
The nine history frames competed in one flattened attention context and were
fused only once. Its factorized row/column query also had substantially less
native-pixel capacity than the proven Single Baseline. This made sharp current
edges and per-frame conflict resolution unnecessarily difficult.

M05 makes four linked changes:

1. A full learned query vector is restored for every native 512-grid cell,
   while continuous Fourier coordinates preserve a smooth spatial prior.
2. The latest frame is decoded first and fixes the final ego-centric reference.
3. Historical frames are visited from newest to oldest. One shared deformable
   update cell contributes a bounded learned residual at each visit. Frame
   reliability acts on gate odds, and no old frame directly replaces the
   latest state.
4. During training only, the latest state is decoded by the same refinement and
   heads used by the final Merged output. The proven Single-Baseline map loss is
   applied to both outputs and averaged. Runtime does not expose or compute this
   auxiliary map.

This is still one end-to-end forward call, not a persistent runtime recurrent
map. The reverse loop is statically unrolled inside one RGB-window inference.
The model creates no per-frame BEV files and performs no external map fusion.

## Existing data only

M05 reuses `merged_complete_10m` and `merged_masked_10m`, both stored as
categorical PNG/JPEG rasters at native 512 x 512 resolution over 10 x 10
metric metres. Decoded JPEG labels are snapped back to the exact simulator
palette before target construction. The current-frame auxiliary
target is derived on the fly from the same complete truth and the already
stored current masked BEV. No data recollection, upsampling, or relabelling is
required. GT Scale is used only to inverse-sample training labels into VGGT
units; cells outside the 10 m source remain hard ignored.

## Tested execution settings

On one RTX PRO 6000 Blackwell, real ten-frame data with BF16 and full
deformable-attention rematerialization used about 13 GiB at batch 1. Increasing
`cross_query_chunk_size` from 8192 to 65536 reduced the steady training step
from about 3.27 s to 1.34 s without changing model semantics. Larger chunks did
not improve throughput. Disabling rematerialization increased memory to about
47.5 GiB and was slower, so M05 keeps full checkpointing. Batch 6 was stable at
about 53.6 GiB and maximized throughput in the short sweep. Batch 8 was also
stable for the all-ten-frame worst case at 69.9 GiB. A direct batch-10 probe
with real ten-frame input, forward, backward, clipping, and fused AdamW updates
then passed with 86.18 GiB peak allocated and 91.22 GiB peak reserved on the
95.6-GiB RTX PRO 6000. Batch 10 is therefore the largest verified safe physical
batch for the dedicated pro6000 profile when lower gradient variance is
preferred. Batch 11 is not safe: the measured slope predicts roughly 94.3 GiB
allocated before allocator reserve/headroom. Batch 1 remains the portable default because the
full training target also includes lower-memory GPUs. The learning rate is
intentionally not scaled merely because the physical batch is larger; an LR
range test should precede any optimizer change.

The batch-8 integration probe used an artifact-refreshed 1,100-session
scene-disjoint manifest. Its inspected batch contained eight different
sessions with the same nine-frame history, so the successful backward step
tested a genuine lower-variance batch rather than eight copies of one sample.
The batch-10 worst-case probe completed two finite optimizer updates at 13.45 s
and 12.45 s, with steady throughput 0.803 sample/s. The trainer logs both local
and global batch size; under DDP the effective global batch is
`10 * world_size` for this pro6000 profile.

## Objective

The latest and Merged predictions share one evidential/routing head and use the
same Single-Baseline objective: grouped guessed occupancy, direct visible
surface occupancy, observed/free routing, support BCE + Dice, and wrong-
evidence regularization. Their normalized losses are averaged, then added to
the independent robust Student-t Scale likelihood. BEV and Scale have disjoint
trainable paths and are gradient-clipped separately.

## Use

```bash
python -m vggt_bev_method1.cli_train_m05 \
  --config configs/m05_reverse_gated_10m_template.toml \
  --data-only

torchrun --standalone --nproc-per-node=1 \
  -m vggt_bev_method1.cli_train_m05 \
  --config configs/m05_reverse_gated_10m_template.toml

python -m vggt_bev_method1.cli_eval_m05 \
  --config configs/m05_reverse_gated_10m_template.toml \
  --checkpoint runs/m05_reverse_gated_first260k_15e_fresh_v1/m05_latest.pt \
  --output-dir runs/m05_reverse_gated_first260k_15e_fresh_v1/eval
```
