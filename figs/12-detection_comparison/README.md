# Figure 12: Same-frame detection comparison

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

Run from the customized Ultralytics checkout so the DPA modules can be loaded:

```powershell
D:\anaconda\envs\yolov8\python.exe `
  E:\LeukoTrack-NF\reproducibility\figures\12-detection_comparison\code\run_detection_comparison.py `
  --experiment-root "D:\Nailfold ablaton experment\ultralytics-8.3.163" `
  --dataset-root "D:\Nailfold ablaton experment\ultralytics-8.3.163\datasets\all-lable-test\automatic" `
  --device 0 `
  --frames T0124 ChildA0346 2d_reg0464 MdT0314 `
  --crop-top 2d_reg0464:0.25
```

After inference, the figure can be regenerated without loading model weights.
**The displayed cases must be pinned explicitly** — without `--frames`, the
script auto-selects frames and will NOT reproduce the published figure:

```powershell
D:\anaconda\envs\yolov8\python.exe `
  E:\LeukoTrack-NF\reproducibility\figures\12-detection_comparison\code\run_detection_comparison.py `
  --experiment-root "D:\Nailfold ablaton experment\ultralytics-8.3.163" `
  --dataset-root "D:\Nailfold ablaton experment\ultralytics-8.3.163\datasets\all-lable-test\automatic" `
  --device 0 --skip-inference `
  --frames T0124 ChildA0346 2d_reg0464 MdT0314 `
  --crop-top 2d_reg0464:0.25
```

Box encoding in model panels:

- blue solid: matched prediction (TP)
- vermillion solid with an internal cross: unmatched prediction (FP)
- magenta dashed: unmatched manual target (FN)
- orange solid in the first column: manual reference

The selected examples are descriptive samples, not an estimate of
population-level superiority. Quantitative conclusions remain those reported
for the complete 350-frame set.

Formal outputs are written to `fig/detection_comparison.png` and
`fig/detection_comparison.pdf`; the grayscale audit is
`fig/detection_comparison_grayscale.png`.
