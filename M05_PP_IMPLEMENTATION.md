# M05++ — Coarse Temporal Reasoning, Latest-guided Fine Correction

M05++ is an independent experiment and uses checkpoint schema
`m05pp-coarse256-latest-skip512-fixed-metric-v1`. It does not overwrite or
strict-resume an M05+ checkpoint.

## Resolution hierarchy

```text
RGB window (1..10 frames)
  -> frozen VGGT once, latest frame first at the backbone boundary
  -> local/global 1024D token halves
  -> independent 96D local/global DPT-lite/FPN streams
  -> role-separated Camera/Register trunk (4 layers, width 512)
  -> 256x256 learned Latest queries + 3 decoder layers
  -> 256x256 shared historical proposals + per-cell temporal softmax
  -> 256x256 two-block heavy spatial refinement
  -> bilinear state upsampling to 512x512
  -> one 512x512 deformable correction reading latest-frame DPT patches
  -> depthwise-separable routing/evidence refinement
  -> 512x512 evidential Merged BEV over fixed 10m x 10m
```

The 512 correction is a learned residual, not an image-independent upsampler.
It owns a native 512 query table and directly samples every level of the
latest-frame DPT pyramid. Its final delta projection starts at zero, so initial
predictions preserve the stable coarse semantics while gradients activate the
fine image-guided path.

All cross-frame computation ends at 256. No historical patch pyramid is read
by the 512 correction. Heavy full-channel 3x3 residual blocks also end at 256;
the final grid uses only depthwise-separable spatial blocks and 1x1 output
heads.

## Token and geometry contract

- Every VGGT patch token is split into local/global 1024D halves.
- Each half is independently projected to 96 channels and remains separate
  through the DPT top-down path; fused query context remains width 96.
- Camera and sixteen Register slots have separate projections and retain slot
  identity through four 512D prefix-attention layers.
- Per-cell prefix reading occurs on the 256 grid.
- Runtime inputs remain RGB only. There are no extrinsics, camera height,
  Single BEV input, warps, FiLM, explicit geometry module, pose/depth head, or
  geometric auxiliary loss.
- The independent Scale branch remains diagnostic/training output only and
  never conditions BEV features.

## Parameter allocation

Measured from `configs/m05_pp_a100_8gpu_10e_frozen.toml`:

| Component | Parameters |
|---|---:|
| Prefix trunk, 4x512 | 14,732,288 |
| Local/global DPT patch pyramid, 96+96 | 2,876,032 |
| Coarse 256 query | 4,215,360 |
| Native 512 query + one latest correction | 17,031,840 |
| History proposal + temporal selection | 344,459 |
| Heavy 256 spatial refinement | 332,544 |
| Lightweight 512 routing/evidence output | 21,316 |
| Parallel Scale path | 1,779,475 |
| Other shared BEV/prefix readers | 2,006,129 |
| **Total M05++ head** | **43,339,443** |

The measured total is 43.34M rather than the rough 39M design estimate because
M05++ retains a full learned 64D identity vector for every 512 cell. Replacing
that table with a factorized query would reduce parameters further, but is not
part of this starting point.

## I/O and losses

Input is a chronological 1–10 RGB-frame window. It is reversed only at the
frozen VGGT boundary so the latest frame owns VGGT's distinguished first-frame
tokens. Output is a 512x512 fixed-metric evidential BEV with FOV support,
Observed Gate, guessed occupied/free Beta evidence, semantic composition, and
confidence. Cell size is `10/512 = 0.01953125 m`.

The primary objective is unchanged:

```text
L_total = L_merged + scheduled(0.10 * L_latest_auxiliary) + L_scale
```

Merged and training-only Latest outputs both pass through the same fine
correction and lightweight output heads. The Latest auxiliary is evaluated on
every early step and later once every four steps at four times its multiplier,
preserving its expected contribution.

Missing depth files are marked invalid only for Scale supervision. Their RGB
and complete/masked BEV targets remain valid and train the BEV path normally.
Missing RGB or BEV artifacts still fail immediately.

## Production training profile

`configs/m05_pp_a100_8gpu_10e_frozen.toml` uses:

| Item | Value |
|---|---:|
| A100 GPUs | 8 |
| Epochs | 10 |
| History caps | `[3,5,8,10,10,10,10,10,10,10]` |
| Batch per GPU / global batch | 1 / 8 |
| Optimizer | fused AdamW |
| LR / minimum LR / weight decay | `1e-4` / `1e-5` / `0.02` |
| Warmup | 3% |
| Gradient clip | 1.0 separately for BEV and Scale |
| Attention rematerialization | disabled |
| Cross-query chunk | 65,536 |
| Validation batches | 500 |
| Checkpoint interval | 5,000 steps |

The frozen manifest and 571 MiB session-record cache are shared read-only with
M05+. M05+'s existing run, checkpoint, and log remain preserved.

The initial formal 8xA100 window reached step 140 with a steady mean of
approximately 0.806 seconds/optimizer-step at the epoch-0 three-frame cap.
Allocated memory stabilized near 31.1 GiB per GPU with 88–100% utilization.
Accounting for longer histories and the later sparse Latest auxiliary schedule,
the provisional full-run ETA is 6–7 days.

```bash
scripts/launch_m05_pp_a100_8gpu.sh
```
