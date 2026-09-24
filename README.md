# LeukoTrack-NF figure packages

Curated figure archive for the manuscript **"LeukoTrack-NF: A Dual-Path
Attention-Enhanced YOLO Framework for Video-Based Leukocyte Quantification in
Nailfold Capillaroscopy"**.

Each `figs/NN-*/` directory contains the public materials for one figure cited
by `main.tex`:

- `fig/` — the rendered figure used by the manuscript (PDF/PNG);
- `code/` — the plotting/processing code that renders or supports the figure;
- `README.md` — package-specific notes (where present).

Figure numbers follow the current `main.tex` (order of first appearance).

| Manuscript figure | Package | Content |
|---|---|---|
| Fig. 1 | `figs/02-system_framework` | System framework diagram |
| Fig. 2 | `figs/03-enhancement_comparison` | Enhancement / metric comparison panels |
| Fig. 3 & S3 | `figs/06-UIS_diagnostic` | UIS on/off diagnostic panels |
| Fig. 4 | `figs/01-DPA_family` | DPA detector family diagram (draw.io source included) |
| Fig. 5 | `figs/04-adaptive_ROI` | Adaptive vessel-ROI generation process |
| Fig. 6 | `figs/05-ST_map_workflow` | ST-map workflow with apparent velocity |
| Fig. 7 | `figs/07-training_dynamics` | Training dynamics of four configurations |
| Fig. 8 | `figs/11-feature_response` | Feature-map / activation comparison |
| Fig. 9 | `figs/09-DPA_performance_profile` | Radar / normalized performance profile |
| Fig. 10 | `figs/08-counting_consistency` | Manual vs. predicted counting consistency |
| Fig. 11 | `figs/10-ST_manual_agreement` | ST velocity vs. manual reader agreement |
| Fig. 12 | `figs/12-detection_comparison` | Same-frame detection comparison |

## Typography

All matplotlib-rendered figures (Figs. 5, 7–12) are set in **Latin Modern
Roman** to match the manuscript body (registered at render time from a local
TeX distribution; the scripts fall back to Arial when unavailable). Symbols
not present in Latin Modern (µ, −) fall back to DejaVu Sans. Figures 1 and 4
are draw.io exports; Figures 2, 3/S3 and 6 are raster panels whose typography
is baked into the pixels.

## Regenerating the figures

Run the plotting scripts with the project environment, from each package
directory, e.g.:

```powershell
# Fig. 7 (figs/07-training_dynamics)
D:\anaconda\envs\yolov8\python.exe code\plot_training_focus.py

# Fig. 10 (figs/08-counting_consistency)
D:\anaconda\envs\yolov8\python.exe code\make_counting_consistency_scatter.py

# Fig. 11 (figs/10-ST_manual_agreement)
D:\anaconda\envs\yolov8\python.exe code\make_velocity_manual_st_agreement.py
```

Some figures additionally consume packaged inputs or controlled data that are
not redistributed in this public archive:

- Fig. 5 consumes the de-identified frame, heatmap, ROI mask and detection
  points that ship with the private reproducibility package.
- Fig. 12 requires the controlled experiment and dataset locations and the
  displayed cases must be pinned explicitly, e.g.
  `--frames T0124 ChildA0346 2d_reg0464 MdT0314 --crop-top 2d_reg0464:0.25`
  (without `--frames` the script auto-selects frames and will NOT reproduce
  the published figure). See `figs/12-detection_comparison/README.md`.

## Scope and data boundary

This public archive intentionally contains **only rendered figures and
figure code**. It does not contain:

- original human-subject videos or raw image collections;
- annotation files, prediction files or other analysis inputs (these ship
  with the controlled reproducibility package);
- real names, hospital identifiers, consent forms, or identity-linkage keys;
- large model checkpoints;
- superseded figure variants.

The displayed imagery is de-identified and was derived from the two 2026
acquisition batches under approval no. GUET-EC-2026-001 (Guilin University of
Electronic Technology) and no. 2026041601 (People's Hospital of Nanhai
Economic Development Zone).

## Interpretation boundary

Having a rendered figure and a plotting script does not by itself establish
end-to-end reproducibility. A figure is complete only when the exact input
data, processing parameters, source code, and, where applicable, model
checkpoint can be identified and retained.
