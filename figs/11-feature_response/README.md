# Figure 8: Feature-response visualization

Manuscript figure: **Fig. 8** (`fig:activation_comparison`). The historical
package index `11-` is retained in the directory name only.

The default entry point generates the 3x3 feature-response comparison without
opening an annotation window:

```text
python code/compare_three_samples.py
```

The optional interactive ROI annotation mode is available with:

```text
python code/compare_three_samples.py --interactive
```

All inputs are package-relative: three MD2 images and their aligned original
auxiliary images under `data/images/`, and the Baseline and DPA-Refine slim
checkpoints under `data/weights/`. The generated figure is written to `fig/`.

`code/compose_manuscript_figure.py` composes the manuscript panel from the
rendered 3x3 grids. The grid composites it reads (`Grid3x3_*.png`) are
intermediates that travel with the controlled reproducibility package and are
not redistributed here, so this composition step does not run from this archive
alone — `fig/feature_response_manuscript.pdf` is the figure of record.

---

*This public archive ships the rendered figure and the figure code only. The
`data/` inputs referenced above — including the model checkpoints under
`data/weights/` — travel with the controlled reproducibility package
(Supplementary Section S2) and are not redistributed here.*
