# -*- coding: utf-8 -*-
"""
Focused training curves (4 models × 2 metrics) — style matched to
training-curve reference (SCI_Full_Metrics_AllModels.png).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "data"
OUT = ROOT / "fig" / "DPA_Training_Focus_4Models.png"

START_EPOCH = 5
SMOOTHING = 0.85
MARK_EVERY = 8

# Wong-style palette (colorblind-friendly)
COLOR_TEAL = "#009E73"    # teal
COLOR_BLUE = "#0072B2"
COLOR_GREY = "#6B7280"
COLOR_GRID = "#E5E7EB"
COLOR_ORANGE = "#E69F00"
COLOR_PURPLE = "#CC79A7"

MODEL_SERIES = [
    {
        "dir": "Baseline_yolov8n",
        "label": "YOLOv8n",
        "color": COLOR_TEAL,
        "linestyle": "--",
        "marker": "^",
        "linewidth": 1.2,
        "markersize": 3.8,
        "ours": False,
    },
    {
        "dir": "DPA_v1_StructOnly",
        "label": "DPA-Base",
        "color": COLOR_BLUE,
        "linestyle": "--",
        "marker": "s",
        "linewidth": 1.2,
        "markersize": 3.8,
        "ours": False,
    },
    {
        "dir": "DPA_v1_GF",
        "label": "DPA-GF",
        "color": COLOR_GREY,
        "linestyle": "--",
        "marker": "X",
        "linewidth": 1.2,
        "markersize": 3.8,
        "ours": False,
    },
    {
        "dir": "DPA_v1_Refine",
        "label": "DPA-Refine",
        "color": COLOR_ORANGE,
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.1,
        "markersize": 5.0,
        "ours": True,
    },
]

PANELS = [
    ("metrics/mAP50(B)", "mAP@0.5 (%)", (78, 96)),
    ("metrics/recall(B)", "Recall (%)", (58, 96)),
]


LM_FONT_DIR = Path(r"C:\Users\A2856\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm")


def _register_lmroman():
    """The approved reference figure is set in Latin Modern Roman (LaTeX
    serif); register it from the local TeX distribution when available."""
    ok = False
    if LM_FONT_DIR.is_dir():
        for name in ("lmroman10-regular.otf", "lmroman10-bold.otf"):
            p = LM_FONT_DIR / name
            if p.is_file():
                font_manager.fontManager.addfont(str(p))
                ok = True
    return ok


def _apply_paper_rc():
    family = "Latin Modern Roman" if _register_lmroman() else "serif"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 9,
            # Sizes verified against the approved reference by direct
            # pdftotext-bbox A/B comparison of identical Type 3 words
            # (systematic extractor bias cancels): title and axis label are
            # 10 pt — NOT the 15/12 estimated from the first pixel pass —
            # ticks and legend 9 pt, inset ticks also 9 pt.
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3.6,
            "ytick.major.size": 3.6,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "figure.dpi": 120,
            "savefig.dpi": 600,
        }
    )


def _style_axes(ax):
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True, length=3.6, width=0.8, labelsize=9)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.8, width=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def smooth(points, factor=SMOOTHING):
    smoothed = []
    for p in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + p * (1 - factor))
        else:
            smoothed.append(p)
    return smoothed


def _load_series(path: Path, col_name: str):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    y = df[col_name].values.astype(float)
    if "metrics" in col_name:
        y = y * 100
    x = df["epoch"].values
    mask = x >= START_EPOCH
    x_plot = x[mask]
    y_plot = smooth(y, SMOOTHING)
    y_plot = [y_plot[i] for i, m in enumerate(mask) if m]
    return x_plot, y_plot


def main():
    _apply_paper_rc()

    # Vertical (portrait) layout: the two panels are stacked so the figure
    # fills a single column at ~1:1 scale instead of being shrunk from a
    # wide landscape arrangement.
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.7), gridspec_kw={"hspace": 0.605})
    lines_legend, labels_legend = [], []

    for i, (col_name, title, ylim) in enumerate(PANELS):
        ax = axes[i]
        plotted = {}
        for spec in MODEL_SERIES:
            csv_path = RUN_DIR / spec["dir"] / "results.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(csv_path)
            x_plot, y_plot = _load_series(csv_path, col_name)
            me = max(1, len(x_plot) // MARK_EVERY)
            z = 10 if spec["ours"] else 2
            line, = ax.plot(
                x_plot,
                y_plot,
                label=spec["label"],
                color=spec["color"],
                linestyle=spec["linestyle"],
                linewidth=spec["linewidth"],
                marker=spec["marker"],
                markersize=spec["markersize"],
                markevery=me,
                markerfacecolor=spec["color"],
                markeredgecolor=spec["color"],
                markeredgewidth=0.5 if spec["ours"] else 0.35,
                alpha=1.0 if spec["ours"] else 0.95,
                zorder=z,
            )
            plotted[spec["label"]] = (x_plot, y_plot)
            if i == 0:
                lines_legend.append(line)
                labels_legend.append(spec["label"])

        ax.set_title(title, fontweight="bold", pad=5)
        ax.set_xlabel("Epoch", labelpad=2)
        _style_axes(ax)
        ax.set_ylim(*ylim)

        # DPA-GF is compressed against the lower limit of the mAP axis.
        # Keep one inset there; the full Recall trajectory is already visible.
        gf = plotted.get("DPA-GF")
        if i == 0 and gf is not None and len(gf[0]) > 0:
            x_gf, y_gf = gf
            y_min, y_max = min(y_gf), max(y_gf)
            pad = max(0.35, 0.10 * (y_max - y_min))
            # Bounds are the measured reference geometry in axes fraction:
            # x 0.62-0.97, y 0.16-0.55 (lower-left anchor, 35% x 39%).
            # Axes.inset_axes takes absolute axes-fraction bounds, which avoids
            # the relative-unit/bbox_to_anchor restriction of the old helper.
            inset = ax.inset_axes([0.62, 0.16, 0.35, 0.39])
            inset.plot(
                x_gf,
                y_gf,
                color=COLOR_GREY,
                linestyle="--",
                linewidth=1.2,
                marker="X",
                markersize=3.0,
                markevery=max(1, len(x_gf) // MARK_EVERY),
                markerfacecolor=COLOR_GREY,
                markeredgecolor=COLOR_GREY,
                zorder=3,
            )
            inset.set_xlim(min(x_gf), max(x_gf))
            inset.set_ylim(y_min - pad, y_max + pad)
            inset.set_xticks([30, 60])
            inset.set_yticks([66, 72, 78])
            inset.tick_params(
                direction="in",
                top=True,
                right=True,
                length=2.5,
                width=0.6,
                labelsize=9,
            )
            for spine in inset.spines.values():
                spine.set_linewidth(0.6)
            # The reference inset carries no grid; main-axes styling already
            # applies inward minor ticks here via rcParams-driven defaults.

    # Margins are tuned so the canvas stays exactly 252 x 338.4 pt (the
    # reference page size) and every text block fits inside it; saving without
    # bbox_inches="tight" keeps the 1:1 column-width scale.
    fig.subplots_adjust(left=0.135, right=0.965, top=0.932, bottom=0.108, hspace=0.605)
    # Panel 1 carries the shared model legend inside the axes. It sits in the
    # empty band between the top curve cluster and the DPA-GF trace, left of
    # the DPA-GF inset, exactly as in the approved reference layout.
    axes[0].legend(
        lines_legend,
        labels_legend,
        loc="center",
        bbox_to_anchor=(0.30, 0.44),
        frameon=False,
        fontsize=9,
        handlelength=1.8,
        labelspacing=0.3,
        borderpad=0.2,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": the margins above already fit every text block,
    # and keeping the full canvas preserves the reference's 252 x 338.4 pt page.
    plt.savefig(OUT, dpi=600)
    # The manuscript includes the PDF; keep the PNG and PDF in sync.
    plt.savefig(OUT.with_suffix(".pdf"), dpi=600)
    plt.close(fig)
    print(f"saved: {OUT}")
    print(f"saved: {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
