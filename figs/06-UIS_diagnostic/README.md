# Figure 3: UIS diagnostic

Manuscript figure: **Fig. 3** (`fig:uis_comparison`). The historical package
index `06-` is retained in the directory name only.

The retained two-example figure can be regenerated from this directory:

```text
python code/plot_uis_comparison_2panel.py
```

The scripts use only package-relative inputs under `data/`:

- `data/before/`: the two UIS-free MD2 images;
- `data/after/`: the two MD2+UIS images;
- `data/labels/`: the corresponding YOLO label files.

Outputs are written to `fig/`. Original source frames and the full dataset are
not required for this retained figure package.

`code/plot_uis_comparison_2panel.py` is the main entry point and imports the
shared helper module `code/uis_plot_helpers.py`. The older preprocessing script
(`数据模糊化批量处理.py`) is retained separately in the package
`archive/` and is not part of the figure run.

---

*This public archive ships the rendered figure and the figure code only. The
`data/` inputs referenced above travel with the controlled reproducibility
package (Supplementary Section S2) and are not redistributed here.*
