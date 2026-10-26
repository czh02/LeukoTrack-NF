# LeukoTrack-NF figure packages

Curated figure archive for the manuscript

> **LeukoTrack-NF: A Dual-Path Attention-Enhanced YOLO Framework for
> Video-Based Leukocyte Quantification in Nailfold Capillaroscopy**

under submission to *Medical Image Analysis* (Elsevier). The archive is aligned with
manuscript revision **10.8** (archive updated 2026-10-26).

Each `figs/NN-*/` directory contains the public materials for one figure cited
by the manuscript source:

- `fig/` — the rendered figure used by the manuscript (PDF; the retained raster
  source is kept alongside it where one exists);
- `code/` — the plotting/processing code that renders or supports the figure;
- `README.md` — package-specific notes (where present).

## Figure ↔ package mapping

The `NN-` directory indices are historical package identifiers and are **not**
manuscript figure numbers. Manuscript numbering follows the order of first
appearance in the manuscript source and is the order listed here.

| Manuscript figure | Label in `main.tex` | Package directory | Content |
|---|---|---|---|
| Fig. 1 | `fig:framework` | `figs/02-system_framework` | System framework diagram |
| Fig. 2 | `fig:enhancement` | `figs/03-enhancement_comparison` | Enhancement / metric comparison panels |
| Fig. 3 | `fig:uis_comparison` | `figs/06-UIS_diagnostic` | UIS on/off diagnostic panels |
| Fig. 4 | `fig:dpa_family` | `figs/01-DPA_family` | DPA detector family diagram |
| Fig. 5 | `fig:roi_generation` | `figs/04-adaptive_ROI` | Adaptive vessel-ROI generation process |
| Fig. 6 | `fig:st_map` | `figs/05-ST_map_workflow` | ST-map workflow with apparent velocity |
| Fig. 7 | `fig:training_dynamics` | `figs/07-training_dynamics` | Training dynamics of four configurations |
| Fig. 8 | `fig:activation_comparison` | `figs/11-feature_response` | Feature-map / activation comparison |
| Fig. 9 | `fig:detection_comparison` | `figs/12-detection_comparison` | Same-frame detection comparison |
| Fig. 10 | `fig:dpa_normalized_profile` | `figs/09-DPA_performance_profile` | Radar / normalized performance profile |
| Fig. 11 | `fig:dpa_refine_counting_consistency` | `figs/08-counting_consistency` | Manual vs. predicted counting consistency |
| Fig. 12 | `fig:st_velocity_agreement` | `figs/10-ST_manual_agreement` | ST velocity vs. manual reader agreement |

All twelve figures are main-text figures; this archive contains no
supplementary figure.

## Method summary

The figures accompany **LeukoTrack-NF**, an offline system for quantifying
leukocyte passage events in nailfold capillaroscopy (NFC) videos. The method
combines three components introduced in the manuscript:

- **MD2** — the modulation-depth representation, a motion-stabilised,
  weak-signal enhancement of the video;
- **UIS** — Unlabeled Instance Suppression, which enables reliable detector
  training under sparse (selectively annotated) labels;
- **DPA** — a Dual-Path Attention detector family that fuses the enhanced and
  raw frames so the two representations compensate for each other's limits;
  DPA-DH adds attention heads over raw-frame features for localisation, while
  DPA-Refine re-weights fused features for counting.

A two-pass procedure links frame-level detections into vessel-level event
counts, and apparent-velocity estimates are validated against manual
ST-based measurements.

## Typography and artwork

- The whole artwork set uses one sans-serif family, **Arial** — the face the
  Elsevier artwork guidelines name first. The draw.io export (Fig. 4) realises
  it with the metric-compatible Liberation Sans; in the raster panels (Figs. 1,
  2, 3 and 6) the lettering is baked into the pixels.
- No figure contains Type 3 fonts. Where lettering is not part of a raster
  panel it is either drawn as vector outlines or embedded as a Type 0
  (TrueType) subset.
- Figs. 7, 10, 11 and 12 are fully vector.
- Figs. 4, 5, 8 and 9 are vector drawings with embedded raster micrograph panels.
- Figs. 1, 2, 3 and 6 are raster panels delivered as PDF wrappers: the bitmap is
  embedded with FlateDecode at column width and the wrapper carries no font
  resources (600 dpi for Figs. 1 and 3, 447 dpi for Fig. 2, 640 dpi for Fig. 6).
  `main.tex` and `main_zh.tex` include the PDF.

## Regenerating the figures

The rendered PDFs are the figures of record; the scripts are published as the
code that produced them. **Every plotting script reads its inputs from the
package `data/` directory, which ships with the controlled reproducibility
package (Supplementary Section S2) and is not redistributed here.** The
commands below therefore document how each figure was made — they do not run
against this archive alone. This is the intended boundary, not a gap in the
release.

With the package inputs staged, run each script from its own package directory:

```text
# Fig. 7  (figs/07-training_dynamics)      reads data/<run>/results.csv
python code/plot_training_focus.py

# Fig. 11 (figs/08-counting_consistency)   reads data/_count_metrics_v2.csv
python code/make_counting_consistency_scatter.py

# Fig. 12 (figs/10-ST_manual_agreement)    reads data/velocity_validation_*.csv
python code/make_velocity_manual_st_agreement.py
```

`figs/04-adaptive_ROI` and `figs/06-UIS_diagnostic` additionally read
de-identified frames, masks and label files from their `data/` directories.
`figs/12-detection_comparison` also requires the controlled experiment and
dataset locations, and its displayed cases must be pinned explicitly:

```text
python code/run_detection_comparison.py \
  --experiment-root "<ultralytics checkout>" \
  --dataset-root "<controlled dataset root>" \
  --device 0 --skip-inference \
  --frames P02-0124 P01-0346 P03-0464 P04-0314 \
  --crop-top P03-0464:0.25
```

Without `--frames` the script auto-selects frames and will **not** reproduce the
published figure. See `figs/12-detection_comparison/README.md`.

## Scope and data boundary

This public archive intentionally contains **only rendered figures and figure
code**. It does not contain:

- original human-subject videos or raw image collections;
- annotation files, prediction files or other analysis inputs (these ship with
  the controlled reproducibility package described in Supplementary Section S2);
- real names, hospital identifiers, consent forms, or identity-linkage keys;
- original frame identifiers: any identifier that carried a cohort-naming prefix
  is replaced here by a neutral token (for example `P01-0346`), and the mapping
  to the original dataset filenames is held in the controlled reproducibility
  package;
- large model checkpoints;
- superseded figure variants.

Any `data/` or `archive/` path still mentioned in an individual package README
refers to the controlled reproducibility package; those directories are not
redistributed here.

The displayed imagery is de-identified and was derived from the two 2026
acquisition batches under approval no. GUET-EC-2026-001 (Guilin University of
Electronic Technology) and no. 2026041601 (People's Hospital of Nanhai Economic
Development Zone).

## Interpretation boundary

Having a rendered figure and a plotting script does not by itself establish
end-to-end reproducibility. A figure is complete only when the exact input data,
processing parameters, source code, and, where applicable, model checkpoint can
be identified and retained.

## Release status

This archive is the figure-and-figure-code component released alongside the
manuscript submission. The full detection/counting source code, model
checkpoints and the controlled reproducibility package are released upon
acceptance of the manuscript; they are not part of this pre-submission
snapshot.

## Licensing

Two licences apply, split by material type:

- **Code** — every script, configuration and diagram source in this archive
  (`.py`, `.yaml`, `.drawio`) is released under the [MIT License](LICENSE).
- **Figures** — every rendered file under `figs/**/fig/` (`.pdf`, `.png`) is
  released under [CC BY 4.0](LICENSE-figures); reuse should credit the
  manuscript named above.

Neither licence extends to the underlying human-subject recordings or the
de-identified derived data, which remain governed by the ethics approval and
informed-consent terms in Supplementary Section S2.
