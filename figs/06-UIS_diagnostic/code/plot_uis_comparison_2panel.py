# -*- coding: utf-8 -*-
"""Build a compact two-example UIS comparison with an averaged density curve."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import uis_plot_helpers as base


RETAINED_SAMPLES = [
    "dT0066",
    "ChildA0027",
]
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]

OUT_DIR = Path(__file__).resolve().parents[1] / "fig"
OUT_COMPARE = OUT_DIR / "uis_comparison_abcd.png"
OUT_GRAY_AVG = OUT_DIR / "uis_grayscale_avg2_abcd.png"
OUT_COMBINED = OUT_DIR / "uis_combined_abcd.png"

DISPLAY_H = 220
LABEL_BAND_H = 30
COL_GAP = 8
ROW_GAP = 8


def save_png(rgb, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, dpi=(600, 600))


def save_combined_pdf(rgb, path: Path) -> None:
    """Embed the combined raster in a vector-PDF page (lossless), matching the
    width the manuscript figure used before."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[:2]
    fig_w = 3.49
    fig = plt.figure(figsize=(fig_w, fig_w * h / w))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(rgb)
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def make_label_band(label: str, width: int) -> np.ndarray:
    band = np.full((LABEL_BAND_H, width, 3), 255, dtype=np.uint8)
    text_size = cv2.getTextSize(label, base.CV_FONT, base.CV_LABEL_FS, base.CV_LABEL_TH)[0]
    x = max(0, (width - text_size[0]) // 2)
    y = (LABEL_BAND_H + text_size[1]) // 2
    cv2.putText(
        band,
        label,
        (x, y),
        base.CV_FONT,
        base.CV_LABEL_FS,
        (0, 0, 0),
        base.CV_LABEL_TH,
        cv2.LINE_AA,
    )
    return band


def build_abcd_grid(samples) -> np.ndarray:
    # Match every tile to the displayed size of the retained lower-row sample.
    reference_before, _, reference_boxes = samples[1]
    reference = base.resize_to_height(
        base.draw_red_boxes(reference_before, reference_boxes),
        DISPLAY_H,
    )
    target_w = reference.shape[1]

    source_panels = [
        (samples[0][0], samples[0][2]),
        (samples[0][1], samples[0][2]),
        (samples[1][0], samples[1][2]),
        (samples[1][1], samples[1][2]),
    ]

    panels = []
    for (image, boxes), label in zip(source_panels, PANEL_LABELS):
        boxed = base.draw_red_boxes(image, boxes)
        tile = cv2.resize(boxed, (target_w, DISPLAY_H), interpolation=cv2.INTER_AREA)
        panels.append(np.vstack((tile, make_label_band(label, target_w))))

    col_gap = np.full((DISPLAY_H + LABEL_BAND_H, COL_GAP, 3), 255, dtype=np.uint8)
    row_gap = np.full((ROW_GAP, target_w * 2 + COL_GAP, 3), 255, dtype=np.uint8)
    row_1 = np.hstack((panels[0], col_gap, panels[1]))
    row_2 = np.hstack((panels[2], col_gap, panels[3]))
    return np.vstack((row_1, row_gap, row_2))


def render_average_curve(gray_pairs) -> np.ndarray:
    """Render the retained-pair mean curves without grid lines or a legend."""
    base._apply_paper_rc()

    before_list = [pair[0] for pair in gray_pairs]
    after_list = [pair[1] for pair in gray_pairs]
    centers, avg_before = base.average_density_curves(before_list)
    _, avg_after = base.average_density_curves(after_list)
    (x_min, x_max), (y_min, y_max) = base.compact_axis_limits(
        centers,
        avg_before,
        avg_after,
    )

    fig, ax = plt.subplots(figsize=base.SINGLE_GRAY_FIGSIZE)
    ax.plot(
        centers,
        avg_before,
        color=base.BEFORE_COLOR,
        linewidth=1.8,
        solid_capstyle="round",
    )
    ax.plot(
        centers,
        avg_after,
        color=base.AFTER_COLOR,
        linewidth=1.8,
        linestyle="--",
        solid_capstyle="round",
    )
    ax.set_xlabel("Grayscale Value", labelpad=2)
    ax.set_ylabel("Density", labelpad=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(False)
    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        length=4,
        width=0.9,
        labelsize=8,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.16)
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(fig)
    buffer.seek(0)
    return np.array(Image.open(buffer).convert("RGB"))


def main() -> None:
    base.SINGLE_GRAY_FIGSIZE = (7.4, 2.45)

    gray_pairs = []
    samples = []
    for stem in RETAINED_SAMPLES:
        before, after, boxes = base.load_sample(stem)
        samples.append((before, after, boxes))
        gray_pairs.append(
            (
                cv2.cvtColor(before, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(after, cv2.COLOR_BGR2GRAY),
            )
        )

    compare_bgr = build_abcd_grid(samples)
    compare_rgb = cv2.cvtColor(compare_bgr, cv2.COLOR_BGR2RGB)
    avg_rgb = render_average_curve(gray_pairs)
    combined_rgb = base.stack_vertical(compare_rgb, avg_rgb, gap=10)

    save_png(compare_rgb, OUT_COMPARE)
    save_png(avg_rgb, OUT_GRAY_AVG)
    save_png(combined_rgb, OUT_COMBINED)
    save_combined_pdf(combined_rgb, OUT_COMBINED.with_suffix(".pdf"))

    print(f"Samples : {', '.join(RETAINED_SAMPLES)}")
    print(f"Saved   : {OUT_COMPARE} ({compare_rgb.shape[1]}x{compare_rgb.shape[0]})")
    print(f"Saved   : {OUT_GRAY_AVG} ({avg_rgb.shape[1]}x{avg_rgb.shape[0]})")
    print(f"Saved   : {OUT_COMBINED} ({combined_rgb.shape[1]}x{combined_rgb.shape[0]})")


if __name__ == "__main__":
    main()
