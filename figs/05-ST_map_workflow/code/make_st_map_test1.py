"""Rebuild the manuscript ST-map panel from calibrated test1 matrices.

The script reads six raw ``ST_line_*.npy`` matrices together with their
JSON metadata.  It applies only a per-panel percentile display stretch; the
underlying ST matrices are not filtered, enhanced, or modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SOURCE = Path(
    r"D:\NailfoldAnalyzer\data\test_video\test1\ST_Output"
)
DEFAULT_OUTPUT = Path(r"E:\LeukoTrack-NF\figs\2026721\st_map.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a publication-ready six-panel ST-map figure."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_panel(source_dir: Path, index: int) -> tuple[np.ndarray, dict]:
    matrix_path = source_dir / f"ST_line_{index}.npy"
    metadata_path = source_dir / f"ST_line_{index}.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Missing ST matrix: {matrix_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing ST metadata: {metadata_path}")

    matrix = np.load(matrix_path, allow_pickle=False)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError(f"Expected a non-empty 2-D matrix: {matrix_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return np.asarray(matrix, dtype=np.float32), metadata


def display_limits(matrix: np.ndarray) -> tuple[float, float]:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError("ST matrix contains no finite values")
    low, high = np.percentile(finite, (1.0, 99.5))
    if high <= low:
        low = float(finite.min())
        high = float(finite.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def build_figure(source_dir: Path) -> mpl.figure.Figure:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.48, 3.65), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        left=0.065,
        right=0.992,
        bottom=0.145,
        top=0.955,
        wspace=0.32,
        hspace=0.52,
    )
    slots = [
        grid[0, 0],
        grid[0, 1],
        grid[0, 2],
        grid[1, 0],
        grid[1, 1],
        grid[1, 2],
    ]

    axes: list[mpl.axes.Axes] = []
    for index, slot in enumerate(slots, start=1):
        matrix, metadata = load_panel(source_dir, index)
        frame_count = int(metadata.get("frame_count", matrix.shape[1]))
        fps = float(metadata.get("fps", 0.0))
        if fps <= 0:
            raise ValueError(f"Invalid fps in ST_line_{index}.json: {fps}")
        duration_s = float(metadata.get("duration_s", frame_count / fps))
        distance_px = float(
            metadata.get("line_length_source_px", max(matrix.shape[0] - 1, 1))
        )
        vmin, vmax = display_limits(matrix)

        ax = fig.add_subplot(slot)
        ax.imshow(
            matrix,
            cmap="gray",
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(0.0, duration_s, 0.0, distance_px),
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.set_title(f"Vessel line {index}", pad=2.0)
        ax.set_xlim(0.0, duration_s)
        ax.set_xticks([0.0, 8.0, 16.0])
        ax.set_ylim(0.0, distance_px)
        ax.set_yticks([0.0, distance_px / 2.0, distance_px])
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f"))
        ax.grid(False)
        axes.append(ax)

    fig.supxlabel("Time (s)", x=0.53, y=0.025, fontsize=7.2)
    fig.supylabel(
        "Distance along vessel line (pixels)", x=0.012, y=0.54, fontsize=7.2
    )
    return fig


def save_figure(fig: mpl.figure.Figure, output_path: Path, dpi: int) -> list[Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = output_path.with_name("st_map_blurry_backup.png")
    if output_path.exists() and not backup_path.exists():
        shutil.copy2(output_path, backup_path)

    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(
        output_path,
        dpi=dpi,
        facecolor="white",
        bbox_inches=None,
        metadata={"Software": "Matplotlib; source: test1 raw ST matrices"},
    )
    fig.savefig(
        pdf_path,
        dpi=dpi,
        facecolor="white",
        bbox_inches=None,
        metadata={"Title": "Six test1 vessel-aligned spatiotemporal maps"},
    )
    return [output_path, pdf_path, backup_path]


def main() -> None:
    args = parse_args()
    if args.dpi < 300:
        raise ValueError("DPI must be at least 300 for manuscript output")
    figure = build_figure(args.source_dir)
    try:
        paths = save_figure(figure, args.output, args.dpi)
    finally:
        plt.close(figure)
    for path in paths:
        if path.exists():
            print(path)


if __name__ == "__main__":
    main()
