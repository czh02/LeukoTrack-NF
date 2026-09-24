# Figure 6: UIS diagnostic

The retained two-example Figure 6 can be regenerated from this directory:

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
is retained separately under `archive/` and is not part of the figure run.
