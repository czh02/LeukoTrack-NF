# Figure 5: Adaptive ROI generation

Manuscript figure: **Fig. 5** (`fig:roi_generation`). The historical package
index `04-` is retained in the directory name only.

The plotting script is self-contained for the retained Figure 5 artifacts. From
this directory, run:

```text
python code/plot_roi_generation_process.py
```

It reads only `data/` (the representative frame, heatmap, ROI masks and
detection-point cache) and writes `fig/roi_generation_process.png`. It does not
require the original videos, a model checkpoint, or a machine-specific absolute
path. The legacy `--force-roi` option is retained for command-line compatibility
but is ignored in packaged mode.

---

*This public archive ships the rendered figure and the figure code only. The
`data/` inputs referenced above travel with the controlled reproducibility
package (Supplementary Section S2) and are not redistributed here.*
