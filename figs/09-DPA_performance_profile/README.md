# Figure 9: Normalized DPA-family performance profile

Run from this directory:

```text
python code/make_dpa_normalized_profile.py
```

The script uses the tabulated values retained in `code/` and writes the color
PDF/PNG and grayscale PNG to `fig/`. The heatmap uses a visible light-blue-grey-
to-medical-blue sequential palette ending at `#0072B2`, matching the manuscript's other
figures. It retains only the normalized-score body and cell values; no separate
gradient colorbar is displayed.
