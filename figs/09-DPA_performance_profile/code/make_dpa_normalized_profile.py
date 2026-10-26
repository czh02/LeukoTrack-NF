"""Create the descriptive normalized DPA-family performance profile."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


PACKAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PACKAGE_DIR / "fig"

DPA_BLUE = LinearSegmentedColormap.from_list(
    "DPA_blue",
    ["#D6E0E8", "#A9C7DC", "#4C97C2", "#0072B2"],
)

MODELS = ["DPA-Base", "DPA-DA", "DPA-DH", "DPA-GF", "DPA-Refine"]
METRICS = [
    "UIS-free\nmAP@0.5:0.95",
    "MAE\n(inverted)",
    "RMSE\n(inverted)",
    "Absolute total-count\nbias (%) (inverted)",
]

# Values are taken from the current UIS-free localization and 28-ROI counting
# tables. The last column uses the absolute magnitude of the reported signed
# total-count bias, so over-counting and under-counting are treated symmetrically.
RAW = np.array(
    [
        [22.77, 10.33, 13.11, 114.75],
        [24.22, 8.96, 11.91, 88.15],
        [25.61, 5.86, 7.56, 43.43],
        [20.95, 7.62, 9.59, 69.22],
        [22.87, 3.57, 4.88, 9.99],
    ],
    dtype=float,
)


def normalize_higher_is_better(values: np.ndarray) -> np.ndarray:
    lo = values.min(axis=0)
    hi = values.max(axis=0)
    span = hi - lo
    return np.divide(values - lo, span, out=np.zeros_like(values), where=span != 0)


def main() -> None:
    transformed = RAW.copy()
    transformed[:, 1:] = -transformed[:, 1:]
    scores = normalize_higher_is_better(transformed)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.15, 3.05), constrained_layout=True)
    im = ax.imshow(scores, cmap=DPA_BLUE, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(METRICS)), METRICS)
    ax.set_yticks(np.arange(len(MODELS)), MODELS)
    ax.tick_params(length=0, pad=5)

    for row in range(scores.shape[0]):
        for col in range(scores.shape[1]):
            color = "white" if scores[row, col] < 0.42 else "black"
            ax.text(col, row, f"{scores[row, col]:.2f}", ha="center", va="center", color=color, fontsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / "dpa_normalized_performance_profile"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")

    gray = plt.figure(figsize=(7.15, 3.05), constrained_layout=True)
    gray_ax = gray.add_subplot(111)
    gray_im = gray_ax.imshow(scores, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
    gray_ax.set_xticks(np.arange(len(METRICS)), METRICS)
    gray_ax.set_yticks(np.arange(len(MODELS)), MODELS)
    gray_ax.tick_params(length=0, pad=5)
    for row in range(scores.shape[0]):
        for col in range(scores.shape[1]):
            color = "white" if scores[row, col] < 0.42 else "black"
            gray_ax.text(col, row, f"{scores[row, col]:.2f}", ha="center", va="center", color=color, fontsize=10)
    gray.savefig(OUT_DIR / "dpa_normalized_performance_profile_grayscale.png", dpi=300, bbox_inches="tight", facecolor="white")

    plt.close(gray)
    plt.close(fig)

    print("Normalized scores (rows: DPA variants; columns: metrics):")
    for model, row in zip(MODELS, scores):
        print(model, " ".join(f"{value:.6f}" for value in row))


if __name__ == "__main__":
    main()
