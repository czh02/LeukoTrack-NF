# -*- coding: utf-8 -*-
"""Publication-style UIS figures: 4x2 comparison with density-curve insets + standalone grayscale grid."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

FIG_DIR = Path(__file__).resolve().parents[1]
NEW_DIR = FIG_DIR / "fig"
DATA_DIR = FIG_DIR / "data"

BEFORE_DIR = DATA_DIR / "before"
RESULT_CLEAN = DATA_DIR / "after"
ARTIFICIAL_LABELS = DATA_DIR / "labels"

SAMPLES = [
    "dT0066",
    "2d_reg0194",
    "ChildA0027",
    "T0305",
]

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]

AFTER_OVERRIDES: dict[str, Path] = {}
LABEL_OVERRIDES: dict[str, Path] = {}

# Center-crop sample width to a reference aspect (remove both sides, keep full height).
# Historically the reference was sample T0305 (983x515) in the four-sample
# layout; the retained two-example package keeps the identical crop via this
# fixed reference size so the figure does not depend on T0305 files.
WIDTH_CROP_REF_SIZE: dict[str, tuple[int, int]] = {
    "ChildA0027": (983, 515),
}

OUT_COMPARE = NEW_DIR / "uis_comparison_4panel.png"
OUT_GRAY = NEW_DIR / "uis_grayscale_4panel.png"
OUT_OVERLAY = NEW_DIR / "uis_overlay_4panel.png"
OUT_GRAY_AVG = NEW_DIR / "uis_grayscale_avg4.png"
OUT_COMBINED = NEW_DIR / "uis_combined_4panel.png"

COMBINED_GAP = 10

SINGLE_GRAY_FIGSIZE = (5.2, 3.4)

# Density-curve inset overlaid on each comparison row (bottom-right)
INSET_FIGSIZE = (1.55, 0.95)
INSET_DPI = 240
INSET_WIDTH_RATIO = 0.36
INSET_HEIGHT_RATIO = 0.62
INSET_MARGIN = 10

GRAY_PANEL_FIGSIZE = (7.4, 5.0)
COL_SEP = 4
ROW_SEP = 6
ROW_LABEL_W = 34
ROW_LABEL_SEP = 6
TITLE_ROW_H = 44
DISPLAY_H = 240
ROW_TITLES = [
    "Before UIS (UIS-free MD2)",
    "After UIS (MD2+UIS)",
]
RED_BGR = (0, 0, 255)
TITLE_BLACK_BGR = (0, 0, 0)
TITLE_RED_BGR = (0, 0, 180)

# OpenCV text: SIMPLEX + thickness=1 reads lighter than DUPLEX/thickness=2
CV_FONT = cv2.FONT_HERSHEY_SIMPLEX
CV_TITLE_FS = 0.72
CV_TITLE_TH = 1
CV_LABEL_FS = 0.72
CV_LABEL_TH = 1

BEFORE_COLOR = "#0072B2"
AFTER_COLOR = "#E69F00"
COLOR_GRID = "#E5E7EB"


def _apply_paper_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "figure.dpi": 120,
            "savefig.dpi": 600,
        }
    )


def _style_axes(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8, linestyle="-")
    ax.tick_params(direction="in", top=True, right=True, length=4, width=0.9, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)


def resolve_after_path(stem: str) -> Path:
    if stem in AFTER_OVERRIDES:
        path = AFTER_OVERRIDES[stem]
        if path.is_file():
            return path
        raise FileNotFoundError(f"After UIS override missing: {path}")

    for split in ("test", "val", "train"):
        path = RESULT_CLEAN / split / "images" / f"{stem}.png"
        if path.is_file():
            return path
    # Packaged layout keeps the retained frames flat under data/after/.
    flat = RESULT_CLEAN / f"{stem}.png"
    if flat.is_file():
        return flat
    raise FileNotFoundError(f"No After UIS image for {stem} under {RESULT_CLEAN}")


def resolve_label_path(stem: str) -> Path:
    if stem in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[stem]

    for split in ("test", "val", "train"):
        path = RESULT_CLEAN / split / "labels" / f"{stem}.txt"
        if path.is_file():
            return path
    alt = ARTIFICIAL_LABELS / f"{stem}.txt"
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"No label file for {stem}")


def load_yolo_boxes(label_path: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, bw, bh = map(float, parts[:5])
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        boxes.append((x1, y1, x2, y2))
    return boxes


def draw_red_boxes(img_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]], thickness: int = 2) -> np.ndarray:
    out = img_bgr.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), RED_BGR, thickness, lineType=cv2.LINE_AA)
    return out


def resize_to_height(img_bgr: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    target_w = int(round(w * target_h / h))
    return cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)


def pad_to_width(img_bgr: np.ndarray, target_w: int, fill: int = 255) -> np.ndarray:
    """Keep aspect ratio; pad horizontally instead of stretching."""
    h, w = img_bgr.shape[:2]
    if w == target_w:
        return img_bgr
    if w > target_w:
        return cv2.resize(img_bgr, (target_w, h), interpolation=cv2.INTER_AREA)
    pad_total = target_w - w
    left = pad_total // 2
    right = pad_total - left
    return cv2.copyMakeBorder(img_bgr, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(fill, fill, fill))


def make_row_label(label: str, row_h: int) -> np.ndarray:
    """One (a)-(d) label per comparison row, drawn after final layout (no stretch)."""
    cell = np.ones((row_h, ROW_LABEL_W, 3), dtype=np.uint8) * 255
    fs, thickness = CV_LABEL_FS, CV_LABEL_TH
    tw, th = cv2.getTextSize(label, CV_FONT, fs, thickness)[0]
    x = max(2, (ROW_LABEL_W - tw) // 2)
    y = int((row_h + th) // 2)
    cv2.putText(cell, label, (x, y), CV_FONT, fs, (0, 0, 0), thickness, cv2.LINE_AA)
    return cell


def make_title_cell(text: str, cell_w: int, color: tuple[int, int, int]) -> np.ndarray:
    fs, thickness = CV_TITLE_FS, CV_TITLE_TH
    tw, th = cv2.getTextSize(text, CV_FONT, fs, thickness)[0]
    row = np.ones((TITLE_ROW_H, cell_w, 3), dtype=np.uint8) * 255
    x = max(8, (cell_w - tw) // 2)
    y = int((TITLE_ROW_H + th) // 2)
    cv2.putText(row, text, (x, y), CV_FONT, fs, color, thickness, cv2.LINE_AA)
    return row


def make_column_header(col_w: int) -> np.ndarray:
    gutter = np.ones((TITLE_ROW_H, ROW_LABEL_W + ROW_LABEL_SEP, 3), dtype=np.uint8) * 255
    sep_v = np.ones((TITLE_ROW_H, COL_SEP, 3), dtype=np.uint8) * 255
    left = make_title_cell(ROW_TITLES[0], col_w, TITLE_BLACK_BGR)
    right = make_title_cell(ROW_TITLES[1], col_w, TITLE_RED_BGR)
    return np.hstack((gutter, left, sep_v, right))


def _reference_size(stem: str) -> tuple[int, int]:
    path = BEFORE_DIR / f"{stem}.png"
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Missing reference image: {path}")
    h, w = img.shape[:2]
    return w, h


def crop_width_to_reference_aspect(
    img_bgr: np.ndarray,
    ref_w: int,
    ref_h: int,
) -> tuple[np.ndarray, int]:
    """Center-crop width so w/h matches reference; keep full height."""
    h, w = img_bgr.shape[:2]
    target_w = int(round(h * ref_w / ref_h))
    target_w = max(1, min(target_w, w))
    if w <= target_w:
        return img_bgr, 0
    x0 = (w - target_w) // 2
    return img_bgr[:, x0 : x0 + target_w].copy(), x0


def shift_boxes_for_crop(
    boxes: list[tuple[int, int, int, int]],
    x0: int,
    crop_w: int,
) -> list[tuple[int, int, int, int]]:
    shifted: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in boxes:
        nx1, nx2 = x1 - x0, x2 - x0
        if nx2 <= 0 or nx1 >= crop_w:
            continue
        shifted.append((max(0, nx1), y1, min(crop_w, nx2), y2))
    return shifted


def apply_width_crop_if_needed(
    stem: str,
    before: np.ndarray,
    after: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    ref_size = WIDTH_CROP_REF_SIZE.get(stem)
    if ref_size is None:
        return before, after, boxes

    ref_w, ref_h = ref_size
    before, x0 = crop_width_to_reference_aspect(before, ref_w, ref_h)
    after, _ = crop_width_to_reference_aspect(after, ref_w, ref_h)
    boxes = shift_boxes_for_crop(boxes, x0, before.shape[1])
    return before, after, boxes


def load_sample(stem: str) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    before_path = BEFORE_DIR / f"{stem}.png"
    after_path = resolve_after_path(stem)
    label_path = resolve_label_path(stem)

    before = cv2.imread(str(before_path))
    after = cv2.imread(str(after_path))
    if before is None:
        raise FileNotFoundError(f"Missing before image: {before_path}")
    if after is None:
        raise FileNotFoundError(f"Missing after image: {after_path}")

    h, w = before.shape[:2]
    if after.shape[:2] != (h, w):
        after = cv2.resize(after, (w, h), interpolation=cv2.INTER_LINEAR)

    boxes = load_yolo_boxes(label_path, w, h)
    return apply_width_crop_if_needed(stem, before, after, boxes)


def build_tile(img_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]], tile_w: int) -> np.ndarray:
    tile = resize_to_height(draw_red_boxes(img_bgr, boxes), DISPLAY_H)
    return pad_to_width(tile, tile_w)


def _vstack_tiles(tiles: list[np.ndarray]) -> np.ndarray:
    sep_h = np.ones((ROW_SEP, tiles[0].shape[1], 3), dtype=np.uint8) * 255
    grid = tiles[0]
    for tile in tiles[1:]:
        grid = np.vstack((grid, sep_h, tile))
    return grid


def _style_axes_inset(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color=COLOR_GRID, linewidth=0.6, linestyle="-")
    ax.tick_params(direction="in", top=True, right=True, length=2.5, width=0.7, labelsize=5.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def _global_ymax(gray_pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    ymax = 0.0
    for gray_before, gray_after in gray_pairs:
        for gray in (gray_before, gray_after):
            counts, _ = np.histogram(gray.ravel(), bins=128, range=(0, 256), density=True)
            ymax = max(ymax, float(smooth_density(counts).max()))
    return ymax


def render_density_inset(
    gray_before: np.ndarray,
    gray_after: np.ndarray,
    ymax: float,
    show_ylabel: bool = False,
) -> np.ndarray:
    _apply_paper_rc()
    fig, ax = plt.subplots(figsize=INSET_FIGSIZE, dpi=INSET_DPI)
    plot_density_curve(ax, gray_before, "Before UIS", BEFORE_COLOR)
    plot_density_curve(ax, gray_after, "After UIS", AFTER_COLOR)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, ymax * 1.12)
    ax.set_xlabel("Gray", fontsize=6, labelpad=1)
    if show_ylabel:
        ax.set_ylabel("Density", fontsize=6, labelpad=1)
    _style_axes_inset(ax)
    fig.subplots_adjust(left=0.16 if show_ylabel else 0.12, right=0.98, top=0.96, bottom=0.22)

    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=INSET_DPI, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def overlay_inset(row_bgr: np.ndarray, inset_rgb: np.ndarray) -> np.ndarray:
    out = row_bgr.copy()
    row_h, row_w = out.shape[:2]
    inset_w = max(48, int(round(row_w * INSET_WIDTH_RATIO)))
    inset_h = max(36, int(round(row_h * INSET_HEIGHT_RATIO)))
    inset_bgr = cv2.cvtColor(inset_rgb, cv2.COLOR_RGB2BGR)
    inset_bgr = cv2.resize(inset_bgr, (inset_w, inset_h), interpolation=cv2.INTER_AREA)

    x1 = row_w - inset_w - INSET_MARGIN
    y1 = row_h - inset_h - INSET_MARGIN
    out[y1 : y1 + inset_h, x1 : x1 + inset_w] = inset_bgr
    return out


def density_curve(gray: np.ndarray, bins: int = 128) -> tuple[np.ndarray, np.ndarray]:
    counts, bin_edges = np.histogram(gray.ravel(), bins=bins, range=(0, 256), density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return centers, smooth_density(counts)


def average_density_curves(gray_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    curves = [density_curve(gray)[1] for gray in gray_list]
    centers = density_curve(gray_list[0])[0]
    return centers, np.mean(np.stack(curves, axis=0), axis=0)


def compact_axis_limits(
    centers: np.ndarray,
    curve_before: np.ndarray,
    curve_after: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Tight x/y limits around the main mass to emphasize Before vs After differences."""
    peak = max(float(curve_before.max()), float(curve_after.max()))
    threshold = peak * 0.012
    active = (curve_before > threshold) | (curve_after > threshold)
    if active.any():
        x_hi = float(centers[active].max()) + 6.0
    else:
        x_hi = 70.0
    x_hi = float(np.clip(x_hi, 32.0, 88.0))

    in_view = centers <= x_hi
    y_hi = max(float(curve_before[in_view].max()), float(curve_after[in_view].max())) * 1.10
    y_hi = max(y_hi, peak * 1.04)
    return (0.0, x_hi), (0.0, y_hi)


def render_averaged_grayscale_figure(gray_pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Mean density of 4 Before images vs 4 After images (single-panel publication style)."""
    _apply_paper_rc()

    before_list = [pair[0] for pair in gray_pairs]
    after_list = [pair[1] for pair in gray_pairs]
    centers, avg_before = average_density_curves(before_list)
    _, avg_after = average_density_curves(after_list)

    (x_min, x_max), (y_min, y_max) = compact_axis_limits(centers, avg_before, avg_after)

    fig, ax = plt.subplots(figsize=SINGLE_GRAY_FIGSIZE)
    ax.plot(centers, avg_before, color=BEFORE_COLOR, linewidth=1.8, solid_capstyle="round", label="Before UIS")
    ax.plot(
        centers,
        avg_after,
        color=AFTER_COLOR,
        linewidth=1.8,
        linestyle="--",
        solid_capstyle="round",
        label="After UIS",
    )

    ax.set_xlabel("Grayscale Value", labelpad=2, fontsize=13)
    ax.set_ylabel("Density", labelpad=2, fontsize=13)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    _style_axes(ax)
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        fontsize=10,
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.16)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def build_averaged_grayscale_figure(gray_pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    rgb = render_averaged_grayscale_figure(gray_pairs)
    NEW_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(OUT_GRAY_AVG)
    return rgb


def resize_rgb_to_width(img_rgb: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if w == target_w:
        return img_rgb
    target_h = int(round(h * target_w / w))
    resized = Image.fromarray(img_rgb).resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.array(resized)


def stack_vertical(top_rgb: np.ndarray, bottom_rgb: np.ndarray, gap: int = COMBINED_GAP) -> np.ndarray:
    target_w = top_rgb.shape[1]
    bottom_rgb = resize_rgb_to_width(bottom_rgb, target_w)
    gap_row = np.ones((gap, target_w, 3), dtype=np.uint8) * 255
    return np.vstack((top_rgb, gap_row, bottom_rgb))


def build_comparison_grid(
    stems: list[str],
    panel_labels: list[str],
    gray_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """4 rows x 2 cols; optionally overlay density inset on each row."""
    samples = [load_sample(stem) for stem in stems]
    ymax = _global_ymax(gray_pairs) if gray_pairs else 0.0

    tile_ws: list[int] = []
    for before, after, boxes in samples:
        tile_ws.append(resize_to_height(draw_red_boxes(before, boxes), DISPLAY_H).shape[1])
        tile_ws.append(resize_to_height(draw_red_boxes(after, boxes), DISPLAY_H).shape[1])
    col_w = max(tile_ws)

    sample_rows: list[np.ndarray] = []
    sep_v = np.ones((DISPLAY_H, COL_SEP, 3), dtype=np.uint8) * 255
    label_sep = np.ones((DISPLAY_H, ROW_LABEL_SEP, 3), dtype=np.uint8) * 255
    for idx, ((before, after, boxes), label) in enumerate(zip(samples, panel_labels)):
        left = build_tile(before, boxes, col_w)
        right = build_tile(after, boxes, col_w)
        content = np.hstack((left, sep_v, right))
        row = np.hstack((make_row_label(label, DISPLAY_H), label_sep, content))
        if gray_pairs is not None:
            gb, ga = gray_pairs[idx]
            inset = render_density_inset(gb, ga, ymax, show_ylabel=(idx == 0))
            row = overlay_inset(row, inset)
        sample_rows.append(row)

    body = _vstack_tiles(sample_rows)
    header = make_column_header(col_w)
    sep_h = np.ones((ROW_SEP, body.shape[1], 3), dtype=np.uint8) * 255
    return np.vstack((header, sep_h, body))


def gray_stats(gray: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "min": float(gray.min()),
        "max": float(gray.max()),
        "p5": float(np.percentile(gray, 5)),
        "p95": float(np.percentile(gray, 95)),
    }


def smooth_density(counts: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    radius = int(max(3, round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return np.convolve(counts, kernel, mode="same")


def plot_density_curve(ax, gray: np.ndarray, label: str, color: str) -> np.ndarray:
    counts, bin_edges = np.histogram(gray.ravel(), bins=128, range=(0, 256), density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    smooth = smooth_density(counts)

    ax.fill_between(centers, smooth, color=color, alpha=0.18, linewidth=0)
    ax.plot(centers, smooth, color=color, linewidth=1.8, solid_capstyle="round", label=label)
    return smooth


def build_grayscale_grid(panel_labels: list[str], gray_pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    _apply_paper_rc()

    fig, axes = plt.subplots(2, 2, figsize=GRAY_PANEL_FIGSIZE, sharex=True, sharey=False)
    ymax = 0.0

    for ax, panel_label, (gray_before, gray_after) in zip(axes.ravel(), panel_labels, gray_pairs):
        smooth_b = plot_density_curve(ax, gray_before, "Before UIS", BEFORE_COLOR)
        smooth_a = plot_density_curve(ax, gray_after, "After UIS", AFTER_COLOR)
        ymax = max(ymax, smooth_b.max(), smooth_a.max())

        ax.set_title(panel_label, fontsize=11, pad=4)
        ax.set_xlim(0, 255)
        _style_axes(ax)

    for ax in axes.ravel():
        ax.set_ylim(0, ymax * 1.12)

    for ax in axes[1, :]:
        ax.set_xlabel("Grayscale Value", labelpad=2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Density", labelpad=2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        fontsize=9,
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.12, wspace=0.22, hspace=0.28)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def main() -> None:
    gray_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for stem in SAMPLES:
        before, after, _ = load_sample(stem)
        gray_pairs.append((cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)))

    compare_plain_bgr = build_comparison_grid(SAMPLES, PANEL_LABELS)
    compare_overlay_bgr = build_comparison_grid(SAMPLES, PANEL_LABELS, gray_pairs=gray_pairs)

    compare_plain_rgb = cv2.cvtColor(compare_plain_bgr, cv2.COLOR_BGR2RGB)
    compare_overlay_rgb = cv2.cvtColor(compare_overlay_bgr, cv2.COLOR_BGR2RGB)
    gray_rgb = build_grayscale_grid(PANEL_LABELS, gray_pairs)

    avg_rgb = build_averaged_grayscale_figure(gray_pairs)
    combined_rgb = stack_vertical(compare_plain_rgb, avg_rgb)

    NEW_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(compare_plain_rgb).save(OUT_COMPARE)
    Image.fromarray(compare_overlay_rgb).save(OUT_OVERLAY)
    Image.fromarray(gray_rgb).save(OUT_GRAY)
    Image.fromarray(combined_rgb).save(OUT_COMBINED)

    print(f"Samples : {', '.join(SAMPLES)}")
    print(f"Saved   : {OUT_COMPARE} (plain)")
    print(f"Saved   : {OUT_OVERLAY} (density inset overlay)")
    print(f"Saved   : {OUT_GRAY}")
    print(f"Saved   : {OUT_GRAY_AVG} (4-image mean density)")
    print(f"Saved   : {OUT_COMBINED} (comparison + mean density)")
    for stem, (gb, ga) in zip(SAMPLES, gray_pairs):
        print(f"  {stem} before mean={gray_stats(gb)['mean']:.1f} after mean={gray_stats(ga)['mean']:.1f}")


if __name__ == "__main__":
    main()
