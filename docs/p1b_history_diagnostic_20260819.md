# P1B History-Length Diagnostic

Date: 2026-08-19

The completed five-epoch 6.5 m Merged routing checkpoint was evaluated on
1,000 scene-disjoint validation windows. Metrics use pixel counts accumulated
within each history-length bucket at threshold 0.5.

| History | Windows | Support IoU | Gate IoU |
|---:|---:|---:|---:|
| 1 | 98 | 0.9424 | 0.8301 |
| 2 | 129 | 0.9327 | 0.8351 |
| 3 | 134 | 0.9085 | 0.7993 |
| 4 | 133 | 0.8877 | 0.7773 |
| 5 | 115 | 0.8703 | 0.7370 |
| 6 | 112 | 0.8590 | 0.7306 |
| 7 | 93 | 0.8644 | 0.7411 |
| 8 | 74 | 0.8507 | 0.7071 |
| 9 | 65 | 0.8433 | 0.6938 |
| 10 | 47 | 0.8682 | 0.6985 |

Overall Support IoU was 0.8784 and overall Observed Gate IoU was 0.7508.
From history 1 to history 9, Support loses 9.9 percentage points and Gate
loses 13.6 percentage points. History 10 has only 47 windows and should not be
used to claim a reversal of the trend.

## Interpretation

Single-history performance is substantially stronger, while both tasks
degrade as cross-frame alignment becomes necessary. The trend is not perfectly
monotonic, because scene motion and coverage vary by bucket, but it supports
cross-frame pose/alignment as a major bottleneck. It does not support treating
the problem as only a binary-mask loss-weight issue.

This result is the evidence gate for P1C: train and validate the explicit
relative SE(2) head before evaluating pose-conditioned Merged routing.
