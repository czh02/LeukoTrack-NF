# -*- coding: utf-8 -*-
"""
Generate publication figure for the improved adaptive ROI pipeline (2x2 panels).
Uses NailfoldAnalyzer ROI logic on test1 dual-video data when artifacts are missing.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = FIG_DIR / "data"
PAPER_FIG = FIG_DIR / "fig" / "roi_generation_process.png"
ROI_OUT = DATA_DIR

CLUSTER_COLORS = np.array(
    [
        [0.12, 0.47, 0.71],
        [1.00, 0.50, 0.05],
        [0.17, 0.63, 0.17],
        [0.84, 0.15, 0.16],
        [0.58, 0.40, 0.74],
        [0.55, 0.34, 0.29],
        [0.89, 0.47, 0.76],
        [0.50, 0.50, 0.50],
        [0.74, 0.74, 0.13],
        [0.09, 0.75, 0.81],
        [0.65, 0.65, 0.65],
        [0.30, 0.30, 0.30],
    ],
    dtype=np.float32,
)


def _ensure_roi_artifacts(force: bool = False) -> Path:
    heatmap_path = ROI_OUT / "heatmap.png"
    mask_path = ROI_OUT / "adaptive_roi_mask.png"
    work_frames = ROI_OUT / "_dpa_roi_work" / "main_frames"
    if not force and work_frames.is_dir() and POINTS_CACHE.is_file():
        return ROI_OUT

    if not force and heatmap_path.is_file() and mask_path.is_file() and work_frames.is_dir():
        if not POINTS_CACHE.is_file():
            _collect_and_cache_points(ROI_OUT)
        return ROI_OUT

    if not NAILFOLD_ROOT.is_dir():
        raise FileNotFoundError(f"NailfoldAnalyzer not found: {NAILFOLD_ROOT}")
    for p in (MODEL_PATH, TEST1_MAIN, TEST1_AUX):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")

    sys.path.insert(0, str(NAILFOLD_ROOT))
    from utils.dpa_registry import bootstrap_local_ultralytics, register_all_dpa_modules

    bootstrap_local_ultralytics()
    from utils.custom_modules import CustomModuleRegistry
    from utils.model_manager import ModelManager
    from services.model_service import ModelService
    from services.video_analyzer import VideoAnalyzer

    register_all_dpa_modules(CustomModuleRegistry)
    ModelManager.set_model(str(MODEL_PATH))
    model = ModelService.get_model()
    analyzer = VideoAnalyzer(model=model, log_callback=lambda m: print(f"[ROI] {m}"))
    ROI_OUT.mkdir(parents=True, exist_ok=True)
    analyzer.generate_roi_dual_video(
        str(TEST1_MAIN),
        str(TEST1_AUX),
        str(ROI_OUT),
        roi_conf=0.25,
        keep_work_dir=True,
    )
    _collect_and_cache_points(ROI_OUT)
    ModelService.clear()
    return ROI_OUT


def _load_background_frame(roi_dir: Path) -> np.ndarray:
    frames_dir = roi_dir / "_dpa_roi_work" / "main_frames"
    frame_files = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No extracted frames under {frames_dir}")
    mid = frame_files[len(frame_files) // 2]
    bgr = cv2.imread(str(mid))
    if bgr is None:
        raise RuntimeError(f"Cannot read frame: {mid}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


POINTS_CACHE = ROI_OUT / "detection_points.npz"


def _save_points_cache(points, shape_hw) -> None:
    h, w = shape_hw
    arr = np.asarray(points, dtype=np.int32)
    POINTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(POINTS_CACHE, points=arr, height=h, width=w)


def _load_points_cache():
    if not POINTS_CACHE.is_file():
        return None
    data = np.load(POINTS_CACHE)
    points = [tuple(row) for row in data["points"]]
    return points, (int(data["height"]), int(data["width"]))


def _collect_and_cache_points(roi_dir: Path):
    cached = _load_points_cache()
    if cached is not None:
        return cached

    sys.path.insert(0, str(NAILFOLD_ROOT))
    from services.video_analyzer import VideoAnalyzer
    from utils.dpa_registry import bootstrap_local_ultralytics, register_all_dpa_modules
    from utils.custom_modules import CustomModuleRegistry
    from utils.model_manager import ModelManager
    from services.model_service import ModelService
    from services.dpa_inference import setup_dpa_inference

    bootstrap_local_ultralytics()
    register_all_dpa_modules(CustomModuleRegistry)
    ModelManager.set_model(str(MODEL_PATH))
    model = ModelService.get_model()
    analyzer = VideoAnalyzer(model=model, log_callback=lambda m: print(f"[PTS] {m}"))

    frames_dir = roi_dir / "_dpa_roi_work" / "main_frames"
    aux_dir = roi_dir / "_dpa_roi_work" / "aux_frames"
    frame_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    frame_paths = [str(p) for p in frame_paths]
    first = cv2.imread(frame_paths[0])
    h, w = first.shape[:2]

    setup_dpa_inference(model, str(aux_dir), log=lambda m: print(f"[DPA] {m}"))
    points = analyzer._collect_roi_detection_points(frame_paths, 0.25)
    ModelService.clear()
    _save_points_cache(points, (h, w))
    return points, (h, w)


def _build_cluster_overlay(points, shape_hw):
    sys.path.insert(0, str(NAILFOLD_ROOT))
    from services.video_analyzer import VideoAnalyzer

    groups = VideoAnalyzer._group_points_by_x_peaks(points)
    h, w = shape_hw
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    va = VideoAnalyzer(model=None, log_callback=lambda m: None)
    for idx, group in enumerate(groups):
        color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        hm = va._generate_heatmap((h, w, 3), group)
        hm = VideoAnalyzer._filter_heatmap_regions(hm)
        peak = float(hm.max())
        if peak <= 1e-6:
            continue
        cluster_mask = (hm >= peak * 0.12).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cluster_mask = cv2.dilate(cluster_mask, kernel, iterations=1)
        sel = cluster_mask > 0
        overlay[sel, :3] = color
        overlay[sel, 3] = 0.58

    return overlay, len(groups)


def _heatmap_bwr(heatmap: np.ndarray) -> np.ndarray:
    hm = heatmap.astype(np.float32)
    peak = float(hm.max())
    if peak <= 1e-6:
        return np.zeros((*hm.shape, 3), dtype=np.float32)
    norm = hm / peak
    cmap = plt.get_cmap("bwr")
    rgba = cmap(norm)
    return rgba[..., :3]


def _roi_contour_overlay(background: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    out = background.astype(np.float32) / 255.0
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge = np.zeros(mask_u8.shape, dtype=np.uint8)
    cv2.drawContours(edge, contours, -1, 1, 2)
    pink = np.array([1.0, 0.20, 0.55], dtype=np.float32)
    edge_idx = edge > 0
    out[edge_idx] = 0.35 * out[edge_idx] + 0.65 * pink
    return np.clip(out, 0, 1)


def _crop_to_content(
    *arrays: np.ndarray,
    mask: np.ndarray | None = None,
    margin: int = 12,
) -> tuple[list[np.ndarray], tuple[int, int, int, int]]:
    """Crop panels to shared bounding box (from mask or non-dark pixels)."""
    if mask is not None and np.any(mask > 0):
        ys, xs = np.where(mask > 0)
    else:
        ref = arrays[0]
        if ref.ndim == 3:
            gray = ref.mean(axis=2)
        else:
            gray = ref
        ys, xs = np.where(gray > np.percentile(gray, 35))
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = arrays[0].shape[:2]
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(h - 1, y1 + margin)
    x1 = min(w - 1, x1 + margin)
    cropped = [arr[y0 : y1 + 1, x0 : x1 + 1] for arr in arrays]
    return cropped, (y0, y1 + 1, x0, x1 + 1)


def _relabel_cluster_colors(cluster_rgb: np.ndarray) -> np.ndarray:
    """Recolor the packaged cluster overlay with the canonical left-to-right
    label palette (CLUSTER_COLORS), matching the manuscript convention."""
    h, w, _ = cluster_rgb.shape
    flat = cluster_rgb.reshape(-1, 3).astype(np.int32)
    nz = np.any(flat != 0, axis=1)
    if not np.any(nz):
        return cluster_rgb
    colors_unique, inverse, counts = np.unique(
        flat[nz], axis=0, return_inverse=True, return_counts=True
    )
    # Major colors = flat fills; antialiased edge pixels are assigned to the
    # nearest major color.
    major = colors_unique[counts > max(200, flat.shape[0] // 10000)]
    dist = np.linalg.norm(
        flat[nz][:, None, :] - major[None, :, :], axis=2
    )
    nearest = np.argmin(dist, axis=1)
    ys, xs = np.divmod(np.where(nz)[0], w)
    order = np.full(len(major), -1, dtype=np.int64)
    for ci in range(len(major)):
        sel = nearest == ci
        if not np.any(sel):
            continue
        order[ci] = np.median(xs[sel])
    rank = np.argsort(np.where(order >= 0, order, np.iinfo(np.int64).max))
    out = np.zeros((h * w, 3), dtype=np.float32)
    for rank_idx, ci in enumerate(rank):
        sel = nearest == ci
        if not np.any(sel):
            continue
        out[np.where(nz)[0][sel]] = CLUSTER_COLORS[rank_idx % len(CLUSTER_COLORS)]
    relabeled = out.reshape(h, w, 3)
    alpha = (np.any(relabeled > 0, axis=2)).astype(np.float32)
    return np.dstack([relabeled, alpha])




def _load_packaged_inputs(data_dir: Path):
    """Load only the checked-in Figure 4 inputs; no video/model is needed."""
    background = cv2.imread(str(data_dir / "frame_00261.png"))
    heatmap = cv2.imread(str(data_dir / "heatmap.png"))
    mask = cv2.imread(str(data_dir / "adaptive_roi_mask.png"), cv2.IMREAD_GRAYSCALE)
    cluster = cv2.imread(str(data_dir / "adaptive_roi_mask_color.png"))
    if background is None or heatmap is None or mask is None or cluster is None:
        raise FileNotFoundError(f"Incomplete Figure 4 inputs under {data_dir}")
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cluster = cv2.cvtColor(cluster, cv2.COLOR_BGR2RGB)
    cluster = _relabel_cluster_colors(cluster)
    cluster = np.dstack([cluster[..., :3], cluster[..., 3] * 0.58])
    shape_hw = background.shape[:2]
    return background, heatmap, mask, cluster, shape_hw


def _packaged_cluster_overlay(points, shape_hw):
    """Render a deterministic x-direction point-cluster overlay locally."""
    h, w = shape_hw
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    colors = CLUSTER_COLORS
    if not points:
        return overlay
    points = np.asarray(points, dtype=np.int32)
    # Split the retained detections into x-ordered groups, matching the
    # directional clustering diagnostic without importing NailfoldAnalyzer.
    order = np.argsort(points[:, 0])
    groups = np.array_split(points[order], min(7, len(points)))
    for idx, group in enumerate(groups):
        if len(group) == 0:
            continue
        color = colors[idx % len(colors)]
        for x, y in group:
            cv2.circle(overlay, (int(x), int(y)), 7, (*color, 0.65), -1)
    return overlay


def plot_figure(roi_dir: Path, output: Path) -> None:
    background, heatmap_show, roi_mask, cluster_overlay, shape_hw = _load_packaged_inputs(roi_dir)
    roi_overlay = _roi_contour_overlay(background, roi_mask)

    (background, heatmap_show, cluster_overlay, roi_overlay), _ = _crop_to_content(
        background, heatmap_show, cluster_overlay, roi_overlay, mask=roi_mask
    )

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
            "font.size": 9,
            "axes.titlesize": 9,
            "figure.dpi": 150,
        }
    )

    h, w = background.shape[:2]
    panel_aspect = w / h
    # Match the manuscript figure's physical size (~3.5 in wide) and title
    # font (~9 pt bold); every title must stay on a single line.
    panel_h = 0.74
    panel_w = panel_h * panel_aspect
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(panel_w * 2 + 0.29, panel_h * 2 + 0.42),
        gridspec_kw={"wspace": 0.06, "hspace": 0.42},
    )
    titles = [
        "(a) Original frame",
        "(b) Activity heatmap",
        "(c) Peak clustering",
        "(d) Adaptive ROI mask",
    ]

    panels = [
        background,
        heatmap_show,
        (background, cluster_overlay),
        roi_overlay,
    ]
    for ax, title, panel in zip(axes.ravel(), titles, panels):
        if isinstance(panel, tuple):
            ax.imshow(panel[0])
            ax.imshow(panel[1])
        else:
            ax.imshow(panel)
        ax.set_title(title, fontweight="normal", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    # The manuscript includes the PDF; keep the PNG and PDF in sync.
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")
    print(f"Saved: {output.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description="Plot improved ROI generation figure")
    parser.add_argument("--force-roi", action="store_true", help="Re-run ROI generation")
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER_FIG,
        help="Output PNG path",
    )
    parser.add_argument(
        "--roi-dir",
        type=Path,
        default=ROI_OUT,
        help="Directory with heatmap.png and adaptive_roi_mask.png",
    )
    args = parser.parse_args()

    # The packaged mode reads only data/; --force-roi is retained for CLI
    # compatibility but does not invoke video or model processing.
    plot_figure(args.roi_dir, args.output)


if __name__ == "__main__":
    main()
