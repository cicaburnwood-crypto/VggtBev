# P1B GT Void audit

This package is a read-only data-quality module. It is not imported by the
P1B dataset, model, loss, evaluator, or checkpoint code.

For every selected training prefix it reports four corresponding GT rasters:

- `single_complete`;
- `single_observed`;
- `merged_complete`;
- `merged_observed`.

Each CSV row contains two deliberately distinct ratios:

- `void_fraction_of_grid = void_pixels / (H * W)`;
- `void_fraction_of_gt_known = void_pixels_intersect_known_GT / known_GT_pixels`.

The first answers how much of the BEV canvas is Void. The second answers how
much semantic supervision in that specific GT raster overlaps Void. Neither
measurement changes the GT file or the P1B loss.

```bash
odineye-p1b-void-audit \
  --dataset-root /path/to/BEV \
  --manifest /path/to/split.json \
  --void-index /path/to/coverage/index.json \
  --output-dir /path/to/report
```

`--frame-mode final` is the default and matches one-prefix-per-session
training. `--frame-mode all` audits every prefix up to `--maximum-history`.

## Threshold visualizer

After the full audit CSV is complete, start the standalone threshold explorer:

```bash
PYTHONPATH=src python tools/serve_void_threshold_visualizer.py \
  --csv runs/void_sweep_413824_v1/final/void_fraction_per_gt_bev.csv \
  --expected-sessions 413824 \
  --host 127.0.0.1 --port 8898
```

Each dataset source has its own slider using `void_fraction_of_grid`. Only the
two Single GT BEV records participate in filtering; the Merged records remain
diagnostic-only. A session is excluded when either Single fraction is strictly
greater than that source's selected threshold. Equality is retained. The
generated `.threshold-index-v3.npz` cache makes subsequent starts immediate
and is automatically invalidated when the CSV size or modification time
changes.

For the canonical 413,824-session sweep, the guarded launcher refuses to start
until the final `COMPLETE` marker exists:

```bash
P1B_VOID_AUDIT_PYTHON=/path/to/python \
  scripts/start_void_threshold_visualizer_413824.sh
```
