"""Create the DPA-Refine vessel-level counting consistency scatter plot."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PACKAGE_DIR / "fig"
DATA_PATH = PACKAGE_DIR / "data" / "counting_consistency_points.csv"

COLORS = {"Video 1": "#0072B2", "Video 2": "#E69F00", "Video 3": "#009E73", "Video 4": "#CC79A7"}
MARKERS = {"Video 1": "o", "Video 2": "s", "Video 3": "^", "Video 4": "D"}


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    required = {"video", "manual", "predicted"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing required columns in {DATA_PATH}: {required}")
    DATA = {
        name: {
            "manual": group["manual"].astype(float).tolist(),
            "predicted": group["predicted"].astype(float).tolist(),
        }
        for name, group in frame.groupby("video", sort=False)
    }
    x_all = np.concatenate([np.asarray(values["manual"], dtype=float) for values in DATA.values()])
    y_all = np.concatenate([np.asarray(values["predicted"], dtype=float) for values in DATA.values()])
    fit_slope, fit_intercept = np.polyfit(x_all, y_all, deg=1)

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
    _fig_family = "Latin Modern Roman" if _lm_ok else "Arial"

    plt.rcParams.update(
        {
            "font.family": _fig_family,
            "font.size": 7.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.54, 3.15), constrained_layout=True)
    for name, values in DATA.items():
        ax.scatter(
            values["manual"], values["predicted"], s=28, marker=MARKERS[name],
            color=COLORS[name], edgecolor=COLORS[name], linewidth=0.0, alpha=0.92, label=name, zorder=3,
        )

    line_x = np.array([0.0, 30.0])
    identity_line, = ax.plot(line_x, line_x, color="#8A8A8A", linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    fit_line, = ax.plot(line_x, fit_slope * line_x + fit_intercept, color="#333333", linewidth=1.1, zorder=2)

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_yticks(np.arange(0, 31, 5))
    ax.set_xlabel("Mean manual reference count")
    ax.set_ylabel("Predicted leukocyte-event count")
    ax.grid(False)
    ax.set_aspect("equal", adjustable="box")

    handles, labels = ax.get_legend_handles_labels()
    handles.extend([identity_line, fit_line])
    labels.extend(["Identity (y=x)", "Linear fit"])
    ax.legend(
        handles, labels, frameon=False, loc="upper left", ncol=2,
        handletextpad=0.25, columnspacing=0.8, borderpad=0.1, labelspacing=0.2,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.75)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / "dpa_refine_counting_consistency"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")

    gray = plt.figure(figsize=(3.54, 3.15), constrained_layout=True)
    gray_ax = gray.add_subplot(111)
    for name, values in DATA.items():
        gray_ax.scatter(values["manual"], values["predicted"], s=24, marker=MARKERS[name], color="#555555", edgecolor="#555555", linewidth=0.0)
    gray_ax.plot(line_x, line_x, color="#999999", linestyle=(0, (3, 2)), linewidth=0.9)
    gray_ax.plot(line_x, fit_slope * line_x + fit_intercept, color="#222222", linewidth=1.1)
    gray_ax.set_xlim(0, 30)
    gray_ax.set_ylim(0, 30)
    gray_ax.set_xticks(np.arange(0, 31, 5))
    gray_ax.set_yticks(np.arange(0, 31, 5))
    gray_ax.set_xlabel("Mean manual reference count")
    gray_ax.set_ylabel("Predicted leukocyte-event count")
    gray_ax.grid(False)
    gray_ax.set_aspect("equal", adjustable="box")
    gray_ax.spines["top"].set_visible(False)
    gray_ax.spines["right"].set_visible(False)
    gray_ax.tick_params(direction="out", length=3, width=0.75)
    gray.savefig(OUT_DIR / "dpa_refine_counting_consistency_grayscale.png", dpi=300, bbox_inches="tight", facecolor="white")

    plt.close(gray)
    plt.close(fig)


if __name__ == "__main__":
    main()
