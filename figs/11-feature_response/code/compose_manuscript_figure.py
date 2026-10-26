# -*- coding: utf-8 -*-
"""Compose the manuscript feature-response figure (Figure 8 in main.tex).

The 3x3 comparison raster produced by `compare_three_samples.py` carries cv2-drawn
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
    # Manuscript-wide typeface (2026-10-07): Arial. Elsevier's artwork
    # guidelines require Type 1 or TrueType (never Type 3) and name Arial /
    # Helvetica first, so every figure is set in one sans-serif family -- the
    # same lettering Figures 1 and 4 already use, since they are drawn in the
    # metric-compatible Helvetica/Arial clone -- which keeps the lettering
    # uniform across the whole artwork set. pdf.fonttype=42 makes matplotlib
    # embed the outlines as genuine TrueType.
    from matplotlib import font_manager

    _available_fonts = {_f.name for _f in font_manager.fontManager.ttflist}
    _fig_family = next(
        (
            _c
            for _c in ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans")
            if _c in _available_fonts
        ),
        "DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "font.family": _fig_family,
            "pdf.fonttype": 42,  # embed TrueType; Elsevier rejects Type 3
            "ps.fonttype": 42,
        }
    )
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
    # dpi=400 keeps the embedded raster >=300 dpi at print size (Elsevier);
    # the default 100 dpi produced a 722 px raster of the 1871 px source grid.
    fig.savefig(OUT_PDF, dpi=400, facecolor="white")
    fig.savefig(OUT_PNG, dpi=400, facecolor="white")
    plt.close(fig)
    print(f"saved: {OUT_PDF}")
    print(f"saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
