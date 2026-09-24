# -*- coding: utf-8 -*-
"""Compose the manuscript feature-response figure (Figure 8 in main.tex).

The 3x3 comparison raster carries cv2-drawn
column titles; the manuscript version replaces that title band with matplotlib
text set in Arial so typography matches the other figures. Inputs are
package-relative; outputs overwrite fig/feature_response_manuscript.{pdf,png}.
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PACKAGE = Path(__file__).resolve().parents[1]
GRID_PNG = PACKAGE / "fig" / "Grid3x3_Comparison_Layer15_red_boxes_20260528_204117.png"
OUT_PDF = PACKAGE / "fig" / "feature_response_manuscript.pdf"
OUT_PNG = PACKAGE / "fig" / "feature_response_manuscript.png"

TITLES = [
    ("Input image", "black"),
    ("Baseline layer 15\n(max activation)", "black"),
    ("Ours (DPA-Refine) layer 15\n(max activation)", "red"),
]
FIG_WIDTH_IN = 7.22
TITLE_BAND_IN = 0.55
FONT_SIZE = 9.5


def crop_title_band(rgb: np.ndarray) -> np.ndarray:
    """Remove the cv2 title band: cut everything above the first content row."""
    gray = rgb.min(axis=2)
    content_rows = np.where((gray < 245).mean(axis=1) > 0.5)[0]
    if len(content_rows) == 0:
        raise RuntimeError("Grid image appears blank")
    top = content_rows.min()
    return rgb[top:, :, :]


def column_centers(rgb: np.ndarray) -> list[float]:
    """Centres of the three panel columns, as fractions of image width."""
    gray = rgb.min(axis=2)
    white_cols = (gray > 245).mean(axis=0) > 0.98
    runs, start = [], None
    for x, is_white in enumerate(white_cols):
        if is_white and start is None:
            start = x
        elif not is_white and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, len(white_cols)))
    gaps = [(a, b) for a, b in runs if (b - a) <= 30]
    if len(gaps) != 2:
        raise RuntimeError(f"Expected 2 column separators, found {len(gaps)}")
    edges = [0] + [ (a + b) // 2 for a, b in gaps ] + [rgb.shape[1]]
    return [ (edges[i] + edges[i + 1]) / 2 / rgb.shape[1] for i in range(3) ]


def main() -> None:
    # Manuscript-wide typeface (2026-09-24): Latin Modern Roman, matching the
    # approved Figure 7. Registered from the local TeX distribution; falls
    # back to Arial when MiKTeX is absent.
    from matplotlib import font_manager

    _lm_dir = Path(r"C:\Users\A2856\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm")
    _lm_ok = False
    if _lm_dir.is_dir():
        for _name in (
            "lmroman10-regular.otf",
            "lmroman10-bold.otf",
            "lmroman10-italic.otf",
            "lmroman10-bolditalic.otf",
        ):
            _p = _lm_dir / _name
            if _p.is_file():
                font_manager.fontManager.addfont(str(_p))
                _lm_ok = True
    plt.rcParams.update({"font.family": "Latin Modern Roman" if _lm_ok else "Arial"})
    rgb = np.asarray(Image.open(GRID_PNG).convert("RGB"))
    grid = crop_title_band(rgb)
    centers = column_centers(grid)

    h, w = grid.shape[:2]
    img_h_in = FIG_WIDTH_IN * h / w
    fig_h_in = img_h_in + TITLE_BAND_IN
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h_in), facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, img_h_in / fig_h_in])
    ax.imshow(grid)
    ax.axis("off")
    for cx, (text, color) in zip(centers, TITLES):
        fig.text(
            cx,
            1.0 - (TITLE_BAND_IN / 2) / fig_h_in,
            text,
            ha="center",
            va="center",
            fontsize=FONT_SIZE,
            color=color,
        )
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, facecolor="white")
    fig.savefig(OUT_PNG, dpi=400, facecolor="white")
    plt.close(fig)
    print(f"saved: {OUT_PDF}")
    print(f"saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
