"""Generate a same-frame qualitative detector comparison for Figure 12.

The script runs all detectors at the manuscript evaluation settings, stores the
low-threshold predictions, scores every frame against the manual UIS-free
labels, selects three complementary cases, and renders an N x 4 comparison
(three rows by default, four when --frames lists four stems).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
PREDICTION_DIR = PACKAGE / "data" / "predictions"
SELECTED_DIR = PACKAGE / "data" / "selected"
FIG_DIR = PACKAGE / "fig"
SAVEFIG_DPI = 400

# Keep in-figure typography consistent with the other manuscript figures.
# Manuscript-wide typeface (2026-09-24): Latin Modern Roman, matching the
# approved Figure 7. Registered from the local TeX distribution; falls back
# to Arial when MiKTeX is absent.
from matplotlib import font_manager as _font_manager

_LM_FONT_DIR = Path(r"C:\Users\A2856\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm")


def _manuscript_font_family() -> str:
    _ok = False
    if _LM_FONT_DIR.is_dir():
        for _name in (
            "lmroman10-regular.otf",
            "lmroman10-bold.otf",
            "lmroman10-italic.otf",
            "lmroman10-bolditalic.otf",
        ):
            _p = _LM_FONT_DIR / _name
            if _p.is_file():
                _font_manager.fontManager.addfont(str(_p))
                _ok = True
    return "Latin Modern Roman" if _ok else "Arial"


plt.rcParams.update({"font.family": _manuscript_font_family()})

MODEL_LABELS = {
    "baseline": "YOLOv8n-MD2+UIS",
    "dpa_dh": "DPA-DH",
    "dpa_refine": "DPA-Refine",
}

GT_COLOR = "#E69F00"
TP_COLOR = "#0072B2"
FP_COLOR = "#D55E00"
FN_COLOR = "#CC79A7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--display-conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument(
        "--frames",
        nargs="+",
        metavar="SAMPLE",
        help="Override automatic selection with image stems (3 or 4).",
    )
    parser.add_argument(
        "--crop-top",
        nargs="*",
        default=[],
        metavar="STEM:FRACTION",
        help="Trim the top of a displayed row, e.g. 2d_reg0464:0.25 (display only).",
    )
    return parser.parse_args()


def weight_paths(root: Path) -> dict[str, Path]:
    base = root / "runs" / "detect" / "DPA_v1_original_ablation"
    return {
        "baseline": base / "Baseline_yolov8n" / "weights" / "best.pt",
        "dpa_dh": base / "DPA_v1_DH" / "weights" / "best_slim.pt",
        "dpa_refine": base / "DPA_v1_Refine" / "weights" / "best_slim.pt",
    }


def find_images(directory: Path) -> dict[str, Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in suffixes
    }


def validate_inputs(dataset_root: Path, weights: dict[str, Path]) -> tuple[dict[str, Path], Path, Path]:
    image_dir = dataset_root / "images"
    aux_dir = dataset_root / "original"
    manual_labels = dataset_root.parent / "artificial" / "labels"

    missing = [str(path) for path in [image_dir, aux_dir, manual_labels, *weights.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    images = find_images(image_dir)
    auxiliary = find_images(aux_dir)
    if len(images) != 350:
        raise RuntimeError(f"Expected 350 main images, found {len(images)}")
    if set(images) != set(auxiliary):
        missing_aux = sorted(set(images) - set(auxiliary))
        extra_aux = sorted(set(auxiliary) - set(images))
        raise RuntimeError(f"Main/auxiliary names differ. Missing={missing_aux}, extra={extra_aux}")

    label_files = sorted(path for path in manual_labels.glob("*.txt") if path.name.lower() != "classes.txt")
    object_count = sum(1 for path in label_files for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(label_files) != 350 or object_count != 4678:
        raise RuntimeError(
            f"Manual-label audit failed: files={len(label_files)}, objects={object_count}; expected 350 and 4678"
        )
    if set(images) != {path.stem for path in label_files}:
        raise RuntimeError("Manual-label filenames do not match the 350 main images")
    return images, aux_dir, manual_labels


def run_inference(
    images_dir: Path,
    aux_dir: Path,
    weights: dict[str, Path],
    device: str,
    batch: int,
) -> None:
    from ultralytics import YOLO

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    for key, weight in weights.items():
        print(f"[infer] {MODEL_LABELS[key]}: {weight}")
        model = YOLO(str(weight))
        if key.startswith("dpa_"):
            if not hasattr(model.model, "set_auxiliary_path"):
                raise RuntimeError(f"{MODEL_LABELS[key]} does not expose set_auxiliary_path")
            model.model.set_auxiliary_path(str(aux_dir))

        records = {}
        results = model.predict(
            source=str(images_dir),
            imgsz=640,
            conf=0.001,
            iou=0.7,
            max_det=300,
            batch=batch,
            device=device,
            save=False,
            stream=True,
            verbose=False,
        )
        for result in results:
            boxes = result.boxes
            records[Path(result.path).stem] = [
                {
                    "xyxy": [round(float(v), 4) for v in xyxy],
                    "confidence": round(float(conf), 6),
                    "class_id": int(cls),
                }
                for xyxy, conf, cls in zip(
                    boxes.xyxy.detach().cpu().tolist(),
                    boxes.conf.detach().cpu().tolist(),
                    boxes.cls.detach().cpu().tolist(),
                )
            ]
        if len(records) != 350:
            raise RuntimeError(f"{MODEL_LABELS[key]} returned {len(records)} images, expected 350")
        output = PREDICTION_DIR / f"{key}.json"
        output.write_text(json.dumps(records, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"[saved] {output}")


def read_gt(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, cx, cy, bw, bh = map(float, line.split()[:5])
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        boxes.append([x1, y1, x2, y2])
    return boxes


def box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt: list[list[float]], pred: list[list[float]], threshold: float) -> dict:
    candidates = sorted(
        ((box_iou(g, p), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)),
        reverse=True,
    )
    matched_gt, matched_pred, matches = set(), set(), []
    for iou, gi, pi in candidates:
        if iou < threshold:
            break
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
            matches.append((gi, pi, iou))
    return {
        "matches": matches,
        "matched_gt": matched_gt,
        "matched_pred": matched_pred,
        "tp": len(matches),
        "fp": len(pred) - len(matches),
        "fn": len(gt) - len(matches),
        "mean_iou": float(np.mean([m[2] for m in matches])) if matches else 0.0,
    }


def source_group(stem: str) -> str:
    return re.sub(r"\d+$", "", stem).lower()


def evaluate_frames(
    images: dict[str, Path],
    labels_dir: Path,
    predictions: dict[str, dict],
    display_conf: float,
    match_iou: float,
) -> tuple[list[dict], dict]:
    rows, details = [], {}
    for stem, image_path in images.items():
        with Image.open(image_path) as image:
            width, height = image.size
        gt = read_gt(labels_dir / f"{stem}.txt", width, height)
        detail = {"gt": gt, "models": {}}
        row = {"frame": stem, "source_group": source_group(stem), "gt_count": len(gt)}
        for key in MODEL_LABELS:
            kept = [item for item in predictions[key][stem] if item["confidence"] >= display_conf]
            pred_boxes = [item["xyxy"] for item in kept]
            matched = match_boxes(gt, pred_boxes, match_iou)
            detail["models"][key] = {"predictions": kept, **matched}
            for metric in ("tp", "fp", "fn", "mean_iou"):
                row[f"{key}_{metric}"] = matched[metric]

        row["fn_gain"] = row["baseline_fn"] - min(row["dpa_dh_fn"], row["dpa_refine_fn"])
        row["fp_gain"] = row["baseline_fp"] - min(row["dpa_dh_fp"], row["dpa_refine_fp"])
        row["iou_gain"] = max(row["dpa_dh_mean_iou"], row["dpa_refine_mean_iou"]) - row["baseline_mean_iou"]
        row["combined_gain"] = 2.0 * row["fn_gain"] + row["fp_gain"] + 4.0 * row["iou_gain"]
        rows.append(row)
        details[stem] = detail
    return rows, details


def choose_cases(rows: list[dict]) -> list[str]:
    eligible = [row for row in rows if 3 <= row["gt_count"] <= 24]
    rules = [
        ("missed-response recovery", lambda r: (r["fn_gain"], r["combined_gain"])),
        ("false-positive suppression", lambda r: (r["fp_gain"], r["combined_gain"])),
        ("box-overlap improvement", lambda r: (r["iou_gain"], r["combined_gain"])),
    ]
    selected, used_groups = [], set()
    for _, key_fn in rules:
        candidates = sorted(eligible, key=key_fn, reverse=True)
        choice = next(
            (r for r in candidates if r["frame"] not in selected and r["source_group"] not in used_groups),
            None,
        )
        if choice is None:
            choice = next((r for r in candidates if r["frame"] not in selected), None)
        if choice:
            selected.append(choice["frame"])
            used_groups.add(choice["source_group"])

    if len(selected) < 3:
        fallback = sorted(eligible, key=lambda r: r["combined_gain"], reverse=True)
        selected.extend(r["frame"] for r in fallback if r["frame"] not in selected) 
    return selected[:3]


def guard_crop(stem: str, detail: dict, height: int, cut: int) -> None:
    """Refuse a display crop that would cut through or above any drawn box."""
    affected = [box for box in detail["gt"] if box[1] < cut]
    for model in detail["models"].values():
        affected += [item["xyxy"] for item in model["predictions"] if item["xyxy"][1] < cut]
    if affected:
        raise RuntimeError(
            f"--crop-top {stem}: {len(affected)} box(es) start above the cut line "
            f"({cut} of {height} px); cropping would desynchronize the overlay."
        )


def draw_dashed_rectangle(ax, box, color, linewidth=1.0, dash=4.0, gap=2.5):
    x1, y1, x2, y2 = box
    segments = []
    for start, end, fixed, horizontal in [
        (x1, x2, y1, True), (x1, x2, y2, True), (y1, y2, x1, False), (y1, y2, x2, False)
    ]:
        pos = start
        while pos < end:
            stop = min(pos + dash, end)
            segments.append(([pos, stop], [fixed, fixed]) if horizontal else ([fixed, fixed], [pos, stop]))
            pos += dash + gap
    for xs, ys in segments:
        ax.plot(xs, ys, color=color, linewidth=linewidth, solid_capstyle="butt")


def render_figure(
    selected: list[str],
    images: dict[str, Path],
    labels_dir: Path,
    details: dict,
    rows_by_frame: dict[str, dict],
    crop_top: dict[str, float] | None = None,
    output_stem: str = "detection_comparison",
) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["reference", "baseline", "dpa_dh", "dpa_refine"]
    titles = ["Manual reference", *[MODEL_LABELS[key] for key in columns[1:]]]
    crop_top = crop_top or {}
    row_images, row_offsets = {}, {}
    for stem in selected:
        array = np.asarray(Image.open(images[stem]).convert("RGB"))
        fraction = crop_top.get(stem, 0.0)
        offset = 0.0
        if fraction > 0.0:
            cut = int(round(array.shape[0] * fraction))
            guard_crop(stem, details[stem], array.shape[0], cut)
            array = array[cut:, :, :]
            offset = float(cut)
        row_images[stem] = array
        row_offsets[stem] = offset

    # Widths stay uniform (every row fills the same cell width), but Samples 2
    # and 4 keep their native frame height so wide strips are not stretched
    # vertically; the remaining rows follow the Sample 1 cell height.
    # Boxes are drawn in pixel space, so overlays stay aligned.
    figure_width = 7.2
    left_margin = 0.035
    right_margin = 0.008
    column_gap = 0.006
    column_width = (1.0 - left_margin - right_margin - 3 * column_gap) / 4
    column_width_inches = column_width * figure_width
    row_gap_inches = 0.045
    title_space_inches = 0.23
    legend_space_inches = 0.19
    bottom_margin_inches = 0.025
    reference_aspect = (
        row_images[selected[0]].shape[1] / row_images[selected[0]].shape[0]
    )
    native_height_rows = {1, 3}  # Sample 2 and Sample 4
    row_heights_inches = []
    for row_index, stem in enumerate(selected):
        if row_index in native_height_rows:
            aspect = row_images[stem].shape[1] / row_images[stem].shape[0]
        else:
            aspect = reference_aspect
        row_heights_inches.append(column_width_inches / aspect)
    figure_height = (
        sum(row_heights_inches)
        + row_gap_inches * (len(selected) - 1)
        + title_space_inches
        + legend_space_inches
        + bottom_margin_inches
    )
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    axes = []
    row_top = 1.0 - title_space_inches / figure_height
    for row_height_inches in row_heights_inches:
        row_height = row_height_inches / figure_height
        row_bottom = row_top - row_height
        row_axes = []
        for col_index in range(4):
            left = left_margin + col_index * (column_width + column_gap)
            row_axes.append(fig.add_axes([left, row_bottom, column_width, row_height]))
        axes.append(row_axes)
        row_top = row_bottom - row_gap_inches / figure_height

    for row_index, stem in enumerate(selected):
        image = row_images[stem]
        offset = row_offsets[stem]
        gt = [
            [box[0], box[1] - offset, box[2], box[3] - offset]
            for box in details[stem]["gt"]
        ]
        shutil.copy2(images[stem], SELECTED_DIR / images[stem].name)
        shutil.copy2(labels_dir / f"{stem}.txt", SELECTED_DIR / f"{stem}.txt")
        # Keep the status badge fully inside the panel: its bbox pad is 1.5 pt,
        # so anchor the padded box clear of the bottom edge (12 px at 400 dpi).
        badge_pad_px = 1.5 / 72 * SAVEFIG_DPI
        badge_y = (badge_pad_px + 12.0) / (row_heights_inches[row_index] * SAVEFIG_DPI)

        for col_index, key in enumerate(columns):
            ax = axes[row_index][col_index]
            ax.imshow(image, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                ax.set_title(titles[col_index], fontsize=8.2, pad=4)
            if col_index == 0:
                for box in gt:
                    x1, y1, x2, y2 = box
                    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=GT_COLOR, linewidth=1.35))
                status = f"GT: {len(gt)}"
            else:
                model = details[stem]["models"][key]
                for gi, box in enumerate(gt):
                    if gi not in model["matched_gt"]:
                        draw_dashed_rectangle(ax, box, FN_COLOR, linewidth=1.15)
                for pi, item in enumerate(model["predictions"]):
                    x1, y1, x2, y2 = item["xyxy"]
                    y1 -= offset
                    y2 -= offset
                    color = TP_COLOR if pi in model["matched_pred"] else FP_COLOR
                    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.35))
                    if pi not in model["matched_pred"]:
                        ax.plot([x1, x2], [y1, y2], color=FP_COLOR, linewidth=0.9)
                        ax.plot([x1, x2], [y2, y1], color=FP_COLOR, linewidth=0.9)
                status = f"TP {model['tp']}  FP {model['fp']}  FN {model['fn']}  IoU {model['mean_iou']:.2f}"
            # Pin the axes to the exact image extent: ax.plot autoscale would
            # otherwise expand the limits past the bottom edge whenever a GT
            # box touches y = height (y2n = 1.0), shrinking the displayed
            # image in the model panels relative to the manual-reference one.
            ax.set_xlim(-0.5, image.shape[1] - 0.5)
            ax.set_ylim(image.shape[0] - 0.5, -0.5)
            ax.text(
                0.02,
                badge_y,
                status,
                transform=ax.transAxes,
                fontsize=6.1,
                color="white",
                va="bottom",
                bbox={"facecolor": "black", "alpha": 0.68, "edgecolor": "none", "pad": 1.5},
            )
        axes[row_index][0].text(
            -0.08,
            0.5,
            f"Sample {row_index + 1}",
            transform=axes[row_index][0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=7.2,
            fontweight="bold",
        )

    # Legend strip below the rows, explaining the box encodings.
    legend_ax = fig.add_axes(
        [
            left_margin,
            bottom_margin_inches / figure_height,
            1.0 - left_margin - right_margin,
            legend_space_inches / figure_height,
        ]
    )
    legend_ax.set_xlim(0.0, 1.0)
    legend_ax.set_ylim(0.0, 1.0)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)

    swatch_w = 0.0135
    swatch_h = 0.52
    text_offset = 0.026
    legend_entries = [
        (0.005, GT_COLOR, "solid", "Manual reference"),
        (0.245, TP_COLOR, "solid", "Matched prediction (TP)"),
        (0.505, FP_COLOR, "cross", "Unmatched prediction (FP)"),
        (0.775, FN_COLOR, "dashed", "Missed manual target (FN)"),
    ]
    for x0, color, style, label in legend_entries:
        y0 = 0.5 - swatch_h / 2
        if style == "dashed":
            legend_ax.add_patch(
                Rectangle(
                    (x0, y0),
                    swatch_w,
                    swatch_h,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.1,
                    linestyle=(0, (3, 2)),
                )
            )
        else:
            legend_ax.add_patch(
                Rectangle((x0, y0), swatch_w, swatch_h, fill=False, edgecolor=color, linewidth=1.3)
            )
            if style == "cross":
                legend_ax.plot([x0, x0 + swatch_w], [y0, y0 + swatch_h], color=color, linewidth=0.85)
                legend_ax.plot([x0, x0 + swatch_w], [y0 + swatch_h, y0], color=color, linewidth=0.85)
        legend_ax.text(
            x0 + text_offset, 0.5, label, va="center", ha="left", fontsize=6.4, color="black"
        )

    png = FIG_DIR / f"{output_stem}.png"

    pdf = FIG_DIR / f"{output_stem}.pdf"
    fig.savefig(png, dpi=SAVEFIG_DPI, facecolor="white")
    fig.savefig(pdf, dpi=SAVEFIG_DPI, facecolor="white")
    plt.close(fig)

    gray = Image.open(png).convert("L")
    gray.save(FIG_DIR / f"{output_stem}_grayscale.png")

    selected_payload = {
        stem: {
            "selection_metrics": rows_by_frame[stem],
            "ground_truth": details[stem]["gt"],
            "models": {
                key: {
                    "predictions": details[stem]["models"][key]["predictions"],
                    "tp": details[stem]["models"][key]["tp"],
                    "fp": details[stem]["models"][key]["fp"],
                    "fn": details[stem]["models"][key]["fn"],
                    "mean_iou": details[stem]["models"][key]["mean_iou"],
                }
                for key in MODEL_LABELS
            },
        }
        for stem in selected
    }
    (SELECTED_DIR / "selected_cases.json").write_text(
        json.dumps(selected_payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return png


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    sys.path.insert(0, str(experiment_root))
    weights = weight_paths(experiment_root)
    images, aux_dir, manual_labels = validate_inputs(args.dataset_root.resolve(), weights)

    if not args.skip_inference:
        run_inference(args.dataset_root.resolve() / "images", aux_dir, weights, args.device, args.batch)

    predictions = {}
    for key in MODEL_LABELS:
        path = PREDICTION_DIR / f"{key}.json"
        if not path.exists():
            raise FileNotFoundError(f"Prediction file missing: {path}")
        predictions[key] = json.loads(path.read_text(encoding="utf-8"))

    rows, details = evaluate_frames(images, manual_labels, predictions, args.display_conf, args.match_iou)
    report = PACKAGE / "data" / "candidate_metrics.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["combined_gain"], reverse=True))

    selected = list(args.frames) if args.frames else choose_cases(rows)
    if not 3 <= len(selected) <= 4 or any(stem not in images for stem in selected):
        raise ValueError(f"Invalid selected frames: {selected}")
    rows_by_frame = {row["frame"]: row for row in rows}
    crop_top: dict[str, float] = {}
    for item in args.crop_top:
        stem, separator, fraction = item.partition(":")
        if not separator:
            raise ValueError(f"--crop-top expects STEM:FRACTION, got {item!r}")
        crop_top[stem] = float(fraction)
    output = render_figure(selected, images, manual_labels, details, rows_by_frame, crop_top)
    print(f"[selected] {', '.join(selected)}")
    print(f"[figure] {output}")


if __name__ == "__main__":
    main()
