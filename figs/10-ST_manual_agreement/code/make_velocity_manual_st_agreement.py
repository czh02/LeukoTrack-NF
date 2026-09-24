"""Create the ST-derived blood-flow velocity agreement figure and data summary.

The input values are 28 vessel-level pairs from the four counting videos. The
manual reference for each vessel is the mean of three manual measurements; the
individual measurements are intentionally not exported.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PACKAGE_DIR / "fig"
DATA_DIR = PACKAGE_DIR / "data"


DATA = {
    "Video 1": (
        [510.59, 484.38, 505.52, 512.44, 513.67, 502.90, 506.42],
        [458.99, 469.9633333, 584.4633333, 597.5766667, 548.4466667, 468.2566667, 510.4033333],
    ),
    "Video 2": (
        [268.61, 95.05, 204.28, 302.14, 397.43, 409.96, 416.66, 377.75, 372.13],
        [211.8033333, 95.87666667, 183.84, 225.0766667, 374.8633333, 359.6866667, 265.97, 285.8166667, 346.9466667],
    ),
    "Video 3": (
        [63.60, 229.94, 237.40, 83.44, 72.59, 136.80],
        [57.88666667, 135.89, 203.4166667, 78.74333333, 67.32, 122.60],
    ),
    "Video 4": (
        [316.00, 475.18, 296.42, 335.90, 465.35, 400.58],
        [261.0733333, 460.21, 266.40, 249.1233333, 451.2833333, 421.9966667],
    ),
}


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    covariance = np.cov(x, y, ddof=1)[0, 1]
    return float(2 * covariance / (np.var(x, ddof=1) + np.var(y, ddof=1) + (x.mean() - y.mean()) ** 2))


def main() -> None:
    manual = np.concatenate([np.asarray(values[1], dtype=float) for values in DATA.values()])
    st_speed = np.concatenate([np.asarray(values[0], dtype=float) for values in DATA.values()])
    groups = np.concatenate([np.repeat(name, len(values[0])) for name, values in DATA.items()])

    differences = st_speed - manual
    pair_means = (st_speed + manual) / 2
    bias = float(differences.mean())
    limits = bias + np.array([-1.96, 1.96]) * differences.std(ddof=1)
    mae = float(np.abs(differences).mean())
    rmse = float(np.sqrt(np.mean(differences ** 2)))
    ccc = concordance_correlation(st_speed, manual)
    relative_difference = np.abs(differences) / np.maximum(np.abs(manual), np.finfo(float).eps)
    within_20 = int(np.count_nonzero(relative_difference <= 0.20))

    colors = {
        "Video 1": "#0072B2",  # blue
        "Video 2": "#E69F00",  # orange
        "Video 3": "#009E73",  # green
        "Video 4": "#CC79A7",  # purple
    }
    markers = {"Video 1": "o", "Video 2": "s", "Video 3": "^", "Video 4": "D"}

    # Typography matched to the counting-consistency scatter (manuscript
    # Figure 10). Both are included at \linewidth (252 pt), but Figure 10's
    # equal-aspect axes crop its canvas to 237.6 pt wide while this figure
    # fills its 260.4 pt canvas. Equal *rendered* text size therefore needs the
    # values below instead of Figure 10's nominal 7.5/8.5/7.5/7.0; the two
    # render at the same point size on the page.
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
            "font.size": 8.2,
            "axes.labelsize": 9.3,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 7.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    # Vertical (portrait) layout: panels (a)/(b) stacked for single-column
    # use at ~1:1 scale.
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.3), constrained_layout=True)
    ax_agreement, ax_ba = axes

    lo = min(manual.min(), st_speed.min()) - 25
    hi = max(manual.max(), st_speed.max()) + 25
    for name in DATA:
        mask = groups == name
        ax_agreement.scatter(
            manual[mask],
            st_speed[mask],
            s=28,
            marker=markers[name],
            color=colors[name],
            edgecolor=colors[name],
            linewidth=0.0,
            alpha=0.92,
            label=name,
            zorder=3,
        )
    ax_agreement.plot([lo, hi], [lo, hi], color="#4D4D4D", linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    ax_agreement.set_xlim(lo, hi)
    ax_agreement.set_ylim(lo, hi)
    ax_agreement.set_xlabel(r"Manual ($\mu$m/s)")
    ax_agreement.set_ylabel(r"Automatic ($\mu$m/s)")
    ax_agreement.legend(frameon=False, loc="lower right", handletextpad=0.35, borderpad=0.15, labelspacing=0.2)
    ax_agreement.text(0.02, 0.98, "(a)", transform=ax_agreement.transAxes, ha="left", va="top", fontweight="bold", fontsize=8.2)

    for name in DATA:
        mask = groups == name
        ax_ba.scatter(
            pair_means[mask],
            differences[mask],
            s=28,
            marker=markers[name],
            color=colors[name],
            edgecolor=colors[name],
            linewidth=0.0,
            alpha=0.92,
            zorder=3,
        )
    ax_ba.axhline(0, color="#7F7F7F", linestyle=(0, (2, 2)), linewidth=0.9, zorder=1)
    ax_ba.axhline(bias, color="#0072B2", linewidth=1.1, zorder=2)
    ax_ba.axhline(limits[0], color="#4D4D4D", linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
    ax_ba.axhline(limits[1], color="#4D4D4D", linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
    ax_ba.set_xlim(pair_means.min() - 25, pair_means.max() + 25)
    ax_ba.set_ylim(limits[0] - 25, limits[1] + 25)
    ax_ba.set_xlabel(r"Pair mean ($\mu$m/s)")
    ax_ba.set_ylabel(r"Auto $-$ manual ($\mu$m/s)")
    ax_ba.text(0.02, 0.98, "(b)", transform=ax_ba.transAxes, ha="left", va="top", fontweight="bold", fontsize=8.2)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.75)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / "velocity_manual_st_agreement"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

    # Grayscale preview is kept under fig/ for visual QA and is not cited by the manuscript.
    gray = plt.figure(figsize=(7.2, 3.0), constrained_layout=True)
    gray_axes = gray.subplots(1, 2)
    for src, dst in zip(axes, gray_axes):
        dst.set_xlim(src.get_xlim())
        dst.set_ylim(src.get_ylim())
        dst.set_xlabel(src.get_xlabel())
        dst.set_ylabel(src.get_ylabel())
        dst.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.75)
        dst.set_axisbelow(True)
        dst.spines["top"].set_visible(False)
        dst.spines["right"].set_visible(False)
    gray_axes[0].plot([lo, hi], [lo, hi], color="#4D4D4D", linestyle=(0, (4, 3)), linewidth=1.0)
    gray_axes[1].axhline(0, color="#7F7F7F", linestyle=(0, (2, 2)), linewidth=0.9)
    gray_axes[1].axhline(bias, color="#333333", linewidth=1.1)
    gray_axes[1].axhline(limits[0], color="#4D4D4D", linestyle=(0, (4, 3)), linewidth=0.9)
    gray_axes[1].axhline(limits[1], color="#4D4D4D", linestyle=(0, (4, 3)), linewidth=0.9)
    for name in DATA:
        mask = groups == name
        gray_axes[0].scatter(manual[mask], st_speed[mask], s=28, marker=markers[name], color="#555555", edgecolor="#555555", linewidth=0.0)
        gray_axes[1].scatter(pair_means[mask], differences[mask], s=28, marker=markers[name], color="#555555", edgecolor="#555555", linewidth=0.0)
    gray.savefig(OUT_DIR / "velocity_manual_st_agreement_grayscale.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(gray)
    plt.close(fig)

    with (DATA_DIR / "velocity_validation_paired_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "vessel", "automatic_st", "manual_mean"])
        for video, (automatic_values, manual_values) in DATA.items():
            for index, (automatic_value, manual_value) in enumerate(zip(automatic_values, manual_values), start=1):
                writer.writerow([video, f"V{index}", automatic_value, manual_value])

    with (DATA_DIR / "velocity_validation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "bias_um_s", "mae_um_s", "rmse_um_s", "ccc", "loa_lower_um_s", "loa_upper_um_s", "within_20_percent_n"])
        writer.writerow([len(st_speed), bias, mae, rmse, ccc, limits[0], limits[1], within_20])

    print(f"n={len(st_speed)}")
    print(f"CCC={ccc:.6f}")
    print(f"MAE={mae:.6f} um/s")
    print(f"RMSE={rmse:.6f} um/s")
    print(f"Bias={bias:.6f} um/s")
    print(f"LoA=[{limits[0]:.6f}, {limits[1]:.6f}] um/s")
    print(f"Within 20%={within_20}/{len(st_speed)}")


if __name__ == "__main__":
    main()
