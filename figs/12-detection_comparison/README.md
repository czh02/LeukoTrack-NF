# Figure 9: Same-frame detection comparison

Manuscript figure: **Fig. 9** (`fig:detection_comparison`). The historical
package index `12-` is retained in the directory name only.

This package generates a qualitative comparison on the 350-frame UIS-free
localization set.

## Evidence boundary

- Main images: MD2 frames under the controlled `automatic/images` directory.
- Manual reference: `artificial/labels`, the 350-file, 4,678-object manual
  annotation set reported in the manuscript.
- DPA auxiliary input: spatially aligned raw frames under `automatic/original`.
- Raw auxiliary frames and model weights remain in controlled storage and are
  not copied into this repository.

The similarly named `automatic/labels` directory is not used as reference. It
contains model-generated labels (348 files and 4,283 boxes), not the manuscript
manual-reference set.

## Models

- YOLOv8n-MD2+UIS: `Baseline_yolov8n/weights/best.pt`
- DPA-DH: `DPA_v1_DH/weights/best_slim.pt`
- DPA-Refine: `DPA_v1_Refine/weights/best_slim.pt`

All raw predictions use the manuscript evaluation settings: 640-pixel input,
confidence threshold 0.001, NMS IoU 0.7, and at most 300 detections per image.
The visualization applies a common confidence threshold of 0.25 and matches
predictions to manual boxes at IoU 0.5.

## Run

Run from the customized Ultralytics checkout so that the DPA modules can be
loaded, with the controlled dataset as `--dataset-root`:

```text
python code/run_detection_comparison.py \
  --experiment-root "<ultralytics checkout>" \
  --dataset-root "<controlled dataset root>" \
  --device 0 \
  --frames P02-0124 P01-0346 P03-0464 P04-0314 \
  --crop-top P03-0464:0.25
```

After inference, the figure can be regenerated without loading model weights.
**The displayed cases must be pinned explicitly** — without `--frames`, the
script auto-selects frames and will NOT reproduce the published figure:

```text
python code/run_detection_comparison.py \
  --experiment-root "<ultralytics checkout>" \
  --dataset-root "<controlled dataset root>" \
  --device 0 --skip-inference \
  --frames P02-0124 P01-0346 P03-0464 P04-0314 \
  --crop-top P03-0464:0.25
```

Frame identifiers are anonymised in this public archive: `P01-0346` stands for
the original dataset filename, and the mapping is held in the controlled
reproducibility package.

Box encoding in the model panels:

- blue solid: matched prediction (TP)
- vermillion solid with an internal cross: unmatched prediction (FP)
- magenta dashed: unmatched manual target (FN)
- orange solid in the first column: manual reference

The selected examples are descriptive samples, not an estimate of
population-level superiority. Quantitative conclusions remain those reported for
the complete 350-frame set.

Formal outputs are written to `fig/detection_comparison.png` and
`fig/detection_comparison.pdf`; the grayscale audit render
`fig/detection_comparison_grayscale.png` stays in the controlled
reproducibility package.

---

*This public archive ships the rendered figure and the figure code only. The
`data/` inputs referenced above travel with the controlled reproducibility
package (Supplementary Section S2) and are not redistributed here.*
