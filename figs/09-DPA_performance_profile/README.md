# Figure 10: Normalized DPA-family performance profile

Manuscript figure: **Fig. 10** (`fig:dpa_normalized_profile`). The historical
package index `09-` is retained in the directory name only.

Run from this directory:

```text
python code/make_dpa_normalized_profile.py
```

The script uses the tabulated values retained in `code/` and writes the colour
PDF/PNG renders to `fig/`. The heatmap uses a visible light-blue-grey-to-
medical-blue sequential palette ending at `#0072B2`, matching the manuscript's
other figures. It retains only the normalized-score body and the cell values; no
separate gradient colorbar is displayed.

`code/make_dpa_radar_preview.py` is the earlier radar variant and is retained
for provenance; the manuscript uses `fig/dpa_radar_preview_revised.pdf`. The
normalized-profile render that `code/make_dpa_normalized_profile.py` produces is
one of those superseded variants and is therefore not shipped; only the figure
of record is kept under `fig/`.

---

*This public archive ships the rendered figure and the figure code only. The
`data/` inputs referenced above travel with the controlled reproducibility
package (Supplementary Section S2) and are not redistributed here.*
