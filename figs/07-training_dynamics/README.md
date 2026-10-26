# Figure 7: Training dynamics

Manuscript figure: **Fig. 7** (`fig:training_dynamics`). The historical package
index `07-` coincides with the current manuscript number.

This package contains the four-model training-curve figure used as Figure 7 in
`main.tex`.

## Layout

- `code/plot_training_focus.py`: plotting script;
- `fig/DPA_Training_Focus_4Models.pdf`: the vector figure used by the manuscript;
- `fig/DPA_Training_Focus_4Models.png`: the retained raster render;
- `data/`: the four retained `results.csv` files and matching `args.yaml`
  records — see the boundary note below.

From this directory, run:

```text
python code/plot_training_focus.py
```

The script reads only from `data/` and writes the regenerated figure to `fig/`;
it contains no machine-specific absolute paths.

---

*This public archive ships the rendered figure and the figure code only. The
`data/` inputs referenced above travel with the controlled reproducibility
package (Supplementary Section S2) and are not redistributed here.*
