# M05 short-training and physical-batch diagnostic

Date: 2026-09-01 (Asia/Hong_Kong)

> Historical M05-v2 diagnostic. Its 6.5-VGGT-unit target regrid was retired by
> the fixed-metric M05-v3 contract. The reported 0.133--0.179 source coverage
> is evidence of that retired coordinate mismatch, not a property of M05-v3.
> Its optimizer/VRAM measurements remain historical only; v3 must be measured
> afresh before selecting a production batch size.

## Decision

Keep the current M05 structure, `history_gate_initial_bias = -1.0`, and the
natural equal dual supervision
`latest_auxiliary_loss_weight = 0.50`. The controlled short runs do not support
adding an artificial boundary loss or weakening Latest supervision. On a
95.6-GiB RTX PRO 6000, use physical batch 10: it is now directly verified with
the worst-case ten-frame history and real forward, backward, gradient clipping,
and fused AdamW update.

This is an architecture/loss sanity check, not a claim of converged model
quality. The holdout set is deliberately tiny, so its absolute metrics have
high variance; comparisons below are paired and use identical initialization
and samples.

## Controlled method

- Frozen VGGT features were cached once in micro-batches of one. The complete
  M05 head then trained with a true physical head batch; there was no gradient
  accumulation and the runtime model was not altered.
- Each run used ten-frame histories, fresh identical initialization, AdamW at
  `1e-4`, zero weight decay for the forced-overfit diagnostic, and 100 updates.
- Final paired selection was deterministically stratified by dataset source.
  Training used one HM3D, one MP3D, and one ProcTHOR-10K session. Holdout used
  one HM3D, HSSD, ProcTHOR-10K, and MP3D session.
- Metrics include fused occupancy, guessed occupancy, FOV support, observed
  gate, two-pixel boundary F1, temporal recall, Latest auxiliary quality,
  scale, supervision coverage, and probabilities outside metric GT support.
- The diagnostic does not save checkpoints.

## Capacity and rejected variants

- A single training sample can be strongly memorized. Reducing Latest weight
  from 0.50 to 0.25 raised memorization but reduced holdout quality. It is an
  overfit direction, not a generalization improvement.
- Initializing the history gate at -2.0 produced an apparently higher fused
  holdout IoU only by collapsing predicted support. Its holdout support IoU was
  0.0498 and support boundary F1 was 0.034, so this variant is rejected.
- On a 31.36-GiB RTX 5090, true M05-head batch 3 is the stable maximum:
  29.69-GiB peak allocated and 30.25-GiB peak reserved. Batch 4 OOMed after
  reaching 30.64 GiB allocated and 31.31 GiB process memory.

## Paired stratified result at update 100

| Holdout metric | Latest weight 0.50 | Latest weight 0.40 |
|---|---:|---:|
| Merged fused occupied IoU | 0.1470 | **0.1623** |
| Merged fused occupied F1 | 0.2563 | **0.2792** |
| Guessed occupied IoU | **0.4712** | 0.4528 |
| FOV support IoU | **0.4597** | 0.4329 |
| Observed gate IoU | **0.4877** | 0.4734 |
| Support boundary F1, 2 px | **0.1909** | 0.1730 |
| Gate boundary F1, 2 px | **0.4229** | 0.4064 |
| Historical observed recall | **0.6573** | 0.6286 |
| Historical support recall | **0.5623** | 0.5337 |
| Latest fused occupied IoU | **0.2047** | 0.1812 |
| Latest support IoU | **0.3018** | 0.2881 |
| Latest gate IoU | **0.6685** | 0.6421 |
| Scale mean relative error | 0.2037 | 0.2037 |

Weight 0.40 improves only the thresholded fused IoU while degrading every
measured routing, boundary, temporal, Latest, and guessed-occupancy metric. It
also lowers mean support probability outside supervised GT from 0.0161 to
0.0105. The likely mechanism is a smaller predicted support region, not a
better geometric reconstruction. Retaining equal Latest/Merged supervision is
therefore the less artificial and better-supported choice.

Scale results are identical because Scale is a parallel branch and the tested
weight changes only the BEV objective. This is expected and confirms that the
paired experiment did not accidentally couple Scale to BEV loss.

## 96-GiB physical batch result

The direct batch-10 probe ran on an otherwise empty 96-GiB accelerator after
a ten-second per-second eligibility monitor:

- input: ten copies of a real ten-frame training sample;
- update 1: loss 4.1184, 13.45 s;
- update 2: loss 3.6482, 12.45 s;
- steady throughput: 0.803 sample/s;
- peak allocated: 86.18 GiB;
- peak reserved: 91.22 GiB of about 95.59 GiB;
- all losses finite and optimizer updates completed.

The measured per-sample memory slope predicts roughly 94.3 GiB allocated for
batch 11 before allocator reserve/headroom, while batch 10 already reserves
91.22 GiB. Batch 11 is therefore not a safe long-run setting. Batch 10 is the
largest verified safe physical batch and reduces gradient variance to about
80% of batch 8 (gradient standard deviation about 89.4%, under IID scaling).
The learning rate remains `1e-4`; no unvalidated linear LR scaling is applied.

## Retired v2 coordinate defect

The stratified training samples have mean metric-source coverage 0.1788 and an
effective output extent of 29.34 m; holdout coverage is only 0.1330 with an
effective extent of 32.02 m. The source GT raster covers 10 m, so most of the
VGGT-unit output grid lies outside available metric supervision and is hard
ignored. Visuals confirm a small supervised GT island inside a much larger
prediction grid. This is more consequential than the 0.40-vs-0.50 loss-weight
change.

M05-v3 removes this regrid. Its 10 m output is supervised directly by the
native 10 m source, so geometric coordinate coverage is one and only the
independent Void/GT-valid mask removes pixels from BEV losses.

## Reproduction tools

- Diagnostic runner: `scripts/diagnose_m05_overfit.py`.
- Batch probe: `scripts/benchmark_m05_batch.py`.

Machine-specific run directories and logs are intentionally excluded from the
repository.
