"""Create a radar-chart preview from the current manuscript DPA values."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PACKAGE_DIR / "fig"

MODELS = [
    "YOLOv8n-MD2+UIS",
    "DPA-Base",
    "DPA-DA",
    "DPA-DH",
    "DPA-GF",
    "DPA-Refine",
]
METRICS = ["mAP", "MAE", "RMSE", "Total-count\n|bias|"]

# Values copied from the current manuscript's DPA-family comparison tables.
RAW = np.array(
    [
        [24.94, 4.36, 5.67, 23.29],
        [22.77, 10.33, 13.11, 114.75],
        [24.22, 8.96, 11.91, 88.15],
        [25.61, 5.86, 7.56, 43.43],
        [20.95, 7.62, 9.59, 69.22],
        [22.87, 3.57, 4.88, 9.99],
    ],
    dtype=float,
)


def normalize_scores(values: np.ndarray) -> np.ndarray:
    transformed = values.copy()
    transformed[:, 1:] *= -1.0
    lo = transformed.min(axis=0)
    hi = transformed.max(axis=0)
    return np.divide(
        transformed - lo,
        hi - lo,
        out=np.zeros_like(transformed),
        where=(hi - lo) != 0,
    )


def main() -> None:
    scores = normalize_scores(RAW)
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#CC79A7", "#009E73"]
    linestyles = ["-", (0, (5, 2)), (0, (1.5, 1.5)), "-.", (0, (4, 1.5, 1, 1.5)), "-"]
    markers = ["o", "o", "s", "^", "D", "P"]

    # Typography matched to the counting-consistency scatter (manuscript
    # Figure 10) so the single-column figures render at the same text size.
    # This figure's tight PDF canvas is 299.3 pt wide versus Figure 10's
    # 237.6 pt, so at \linewidth (252 pt) it is scaled by 0.842 rather than
    # 1.060; the nominal sizes below are back-solved for that factor and land
    # on the same rendered 8.98 / 7.95 / 7.42 pt as Figure 10.
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
            "font.size": 10.6,
            "axes.titlesize": 9,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 10.6,
            "pdf.fonttype": 42,  # embed TrueType outlines; Elsevier rejects Type 3 / CID Type 0C
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.75, 4.05), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRICS)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#4B5563")
    ax.set_rlabel_position(18)
    ax.grid(color="#D5DEE8", linewidth=0.55)
    ax.spines["polar"].set_color("#94A3B8")
    ax.spines["polar"].set_linewidth(0.75)

    axis_labels = ax.get_xticklabels()
    axis_labels[1].set_ha("left")
    axis_labels[3].set_ha("right")

    for model, row, color, linestyle, marker in zip(MODELS, scores, colors, linestyles, markers):
        values = row.tolist() + row[:1].tolist()
        ax.plot(
            angles,
            values,
            color=color,
            linewidth=1.25 if model == "DPA-Refine" else 1.0,
            linestyle=linestyle,
            marker=marker,
            markersize=3.5 if model == "DPA-Refine" else 3.0,
            alpha=0.95 if model == "DPA-Refine" else 0.82,
            label=model,
            zorder=4 if model == "DPA-Refine" else 3,
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=9.9,
        handlelength=2.5,
        columnspacing=0.9,
        labelspacing=0.5,
    )
    fig.subplots_adjust(left=0.14, right=0.91, top=0.95, bottom=0.24)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / "dpa_radar_preview_revised"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Generated:")
    print(base.with_suffix(".png"))
    print(base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
