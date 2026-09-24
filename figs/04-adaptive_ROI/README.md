# Figure 4: Adaptive ROI generation

The plotting script is self-contained for the retained Figure 4 artifacts.
From this directory, run:

```text
python code/plot_roi_generation_process.py
```

It reads only `data/` (the representative frame, heatmap, ROI masks and
detection-point cache) and writes `fig/roi_generation_process.png`. It does
not require the original videos, model checkpoint, or a machine-specific
absolute path. The legacy `--force-roi` option is retained for command-line
compatibility but is ignored in packaged mode.
