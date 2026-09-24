# Figure 7: Training Dynamics

This package contains the four-model training-curve figure used as Figure 7
in `main.tex`.

## Layout

- `code/plot_training_focus.py`: plotting script;
- `data/`: the four retained `results.csv` files and matching `args.yaml`
  records;
- `fig/DPA_Training_Focus_4Models.png`: rendered figure.

From this directory, run:

```text
python code/plot_training_focus.py
```

The script reads only from `data/` and writes the regenerated figure to
`fig/`; it contains no machine-specific absolute paths.
