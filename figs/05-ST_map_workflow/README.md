# Figure 6: ST-map workflow

Manuscript figure: **Fig. 6** (`fig:st_map`). The historical package index `05-`
is retained in the directory name only.

The manuscript uses `fig/st_map_workflow_velocity_selected.pdf`, the manually
recomposed final figure; `fig/st_map_workflow_velocity_selected.png` is the
retained raster render.

`code/make_st_map_test1.py` rebuilds the six-panel ST-map body from calibrated
`ST_line_*.npy` matrices together with their JSON metadata. It applies only a
per-panel percentile display stretch; the underlying ST matrices are not
filtered, enhanced or otherwise modified. The script documents how the ST-map
panel is produced — it is **not** an end-to-end regeneration of the composed
workflow figure, which was assembled manually, and no such end-to-end command is
claimed for this package.

The former `make_st_map_workflow.py` was retired to the package `archive/`: it is
an older velocity/ST composition script that cannot regenerate the current
manually recomposed figure. It is not redistributed here.

---

*This public archive ships the rendered figure and the figure code only. The
`data/` inputs referenced above travel with the controlled reproducibility
package (Supplementary Section S2) and are not redistributed here.*
