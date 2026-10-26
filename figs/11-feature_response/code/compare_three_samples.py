"""
3 张样本 × 3 列（Input / Baseline / DPA）对比图生成 + 同步红框标注。

流程：
  1. 自动推理并拼接 3×3 矩阵（无灰边，与 V_concat 一致）
  2. 弹出标注窗口：在每一行左侧 Input 分别拖拽画框
  3. 每行红框仅同步该行 3 列（三行互不影响）；Enter/s 保存

快捷键：Enter/s 保存 | z 撤销 | c 清空 | r 重置视图 | q/Esc 退出
"""

from __future__ import annotations

import argparse
import cv2
import math
import torch
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from PIL import Image, ImageDraw
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

PACKAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PACKAGE_DIR / "fig"
IMAGE_DIR = PACKAGE_DIR / "data" / "images"
WEIGHTS_DIR = PACKAGE_DIR / "data" / "weights"
RUN_DIR = WEIGHTS_DIR
ROI_EXPORT_DIR = OUT_DIR / "roi_exports"

# 3 组样本主路图（MD2）。
# Frame identifiers are anonymised in this public archive: the `P0x-NNNN` tokens
# stand for the original dataset filenames, and the mapping is held in the
# controlled reproducibility package (Supplementary Section S2).
IMG_PATHS = [
    IMAGE_DIR / "md2" / "P03-0068.png",
    IMAGE_DIR / "md2" / "P03-0056.png",
    IMAGE_DIR / "md2" / "P03-0203.png",
]

AUX_DIR = IMAGE_DIR / "original"
BASELINE_WEIGHTS = WEIGHTS_DIR / "baseline_best.pt"
DPA_WEIGHTS = WEIGHTS_DIR / "dpa_refine_best_slim.pt"

LAYER_INDEX = 15
IMGSZ = 640
SEP = 4
COL_TITLES = [
    "Input Image",
    f"Baseline Layer {LAYER_INDEX} (Max Activation)",
    f"Ours (DPA-Refine) Layer {LAYER_INDEX} (Max Activation)",
]


# ---------------------------------------------------------------------------
# 模型 / 特征可视化
# ---------------------------------------------------------------------------
def set_model_aux_path(model, aux_path):
    for m in model.modules():
        if hasattr(m, "set_auxiliary_path"):
            m.set_auxiliary_path(aux_path)


def set_model_image_names(model, names):
    for m in model.modules():
        if hasattr(m, "set_image_names"):
            m.set_image_names(names)


def get_padding_info(original_img, target_size=640):
    h_ori, w_ori = original_img.shape[:2]
    r = min(target_size / h_ori, target_size / w_ori)
    h_new, w_new = int(round(h_ori * r)), int(round(w_ori * r))
    dh, dw = (target_size - h_new) / 2.0, (target_size - w_new) / 2.0
    return dh, dw, h_new, w_new, h_ori, w_ori


def _feat_crop_bounds(dh, dw, h_new, w_new, H_feat, W_feat, target_size=640):
    h_start = int(round(dh * H_feat / target_size))
    h_end = int(round((dh + h_new) * H_feat / target_size))
    w_start = int(round(dw * W_feat / target_size))
    w_end = int(round((dw + w_new) * W_feat / target_size))
    if dh > 0:
        h_start = min(h_start + 1, H_feat // 2)
        h_end = max(h_end - 1, H_feat // 2)
    if dw > 0:
        w_start = min(w_start + 1, W_feat // 2)
        w_end = max(w_end - 1, W_feat // 2)
    return max(0, h_start), min(H_feat, h_end), max(0, w_start), min(W_feat, w_end)


def process_feature(feat_tensor, dh, dw, h_new, w_new, target_size=640):
    heatmap = torch.max(feat_tensor, dim=0)[0].cpu().numpy()
    H_feat, W_feat = heatmap.shape
    h_start, h_end, w_start, w_end = _feat_crop_bounds(
        dh, dw, h_new, w_new, H_feat, W_feat, target_size
    )
    heatmap = heatmap[h_start:h_end, w_start:w_end]
    if heatmap.size > 0:
        vmin, vmax = np.percentile(heatmap, 2), np.percentile(heatmap, 98)
        heatmap = np.clip(heatmap, vmin, vmax)
        heatmap = (heatmap - vmin) / (vmax - vmin + 1e-8)
    return heatmap


def overlay_heatmap_dynamic_alpha(img_np, heatmap, colormap=cv2.COLORMAP_INFERNO):
    hm_color = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    alpha = heatmap[..., np.newaxis] * 0.85
    return np.uint8(img_np * (1 - alpha) + hm_color * alpha)


def _make_title_header(titles, anchor_w, sep, grid_width):
    """顶部列标题：按每列宽度居中，自动缩小字号避免互相遮挡。"""
    font = cv2.FONT_HERSHEY_DUPLEX
    colors = [(0, 0, 0), (0, 0, 0), (180, 0, 0)]
    col_starts = [0, anchor_w + sep, 2 * (anchor_w + sep)]
    max_text_w = anchor_w - 32  # 列内左右留白

    layouts = []
    base_fs = max(0.40, anchor_w / 720.0)
    min_fs = 0.32

    for title in titles:
        fs = base_fs
        thickness = max(1, int(fs * 2.0))
        tw, th = cv2.getTextSize(title, font, fs, thickness)[0]
        while fs > min_fs and tw > max_text_w:
            fs -= 0.025
            thickness = max(1, int(fs * 2.0))
            tw, th = cv2.getTextSize(title, font, fs, thickness)[0]
        layouts.append((title, fs, thickness, tw, th))

    header_height = int(max(56, max(x[4] for x in layouts) + 30))
    header = np.ones((header_height, grid_width, 3), dtype=np.uint8) * 255

    for i, (title, fs, thickness, tw, th) in enumerate(layouts):
        col_cx = col_starts[i] + anchor_w // 2
        text_x = max(col_starts[i] + 4, int(col_cx - tw // 2))
        # 不超出列右边界
        text_x = min(text_x, col_starts[i] + anchor_w - tw - 4)
        text_y = int((header_height + th) // 2)
        cv2.putText(
            header,
            title,
            (text_x, text_y),
            font,
            fs,
            colors[i],
            thickness,
            cv2.LINE_AA,
        )
    return header, header_height


def build_grid3x3(img_paths, layer_index=LAYER_INDEX, imgsz=IMGSZ):
    """推理 3 张图，返回带标题栏的最终 RGB 图及布局元数据。"""
    feat_orig_list, feat_dpa_list = [], []

    def hook_orig(module, input, output):
        feat_orig_list.append(output.detach())

    def hook_dpa(module, input, output):
        feat_dpa_list.append(output.detach())

    print("⏳ 加载 Baseline / DPA 权重...")
    model_orig = YOLO(str(BASELINE_WEIGHTS))
    device_orig = next(model_orig.model.parameters()).device
    handle_orig = model_orig.model.model[layer_index].register_forward_hook(hook_orig)

    model_dpa = YOLO(str(DPA_WEIGHTS))
    device_dpa = next(model_dpa.model.parameters()).device
    set_model_aux_path(model_dpa.model, str(AUX_DIR))
    handle_dpa = model_dpa.model.model[layer_index].register_forward_hook(hook_dpa)

    row_images = []
    anchor_h = anchor_w = None
    row_stems = []

    for idx, img_path in enumerate(img_paths):
        img_path = Path(img_path)
        print(f"\n🚀 [{idx + 1}/3] {img_path.name}")
        feat_orig_list.clear()
        feat_dpa_list.clear()

        original_img = cv2.imread(str(img_path))
        if original_img is None:
            print(f"⚠️ 跳过（未找到）: {img_path}")
            continue
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        row_stems.append(img_path.stem)

        if idx == 0:
            anchor_h, anchor_w = original_img.shape[:2]
        else:
            original_img = cv2.resize(original_img, (anchor_w, anchor_h))

        dh, dw, h_new, w_new, h_ori, w_ori = get_padding_info(original_img, imgsz)
        im_padded = LetterBox(imgsz, auto=False, stride=32)(image=original_img)
        im_tensor = np.ascontiguousarray(im_padded.transpose((2, 0, 1))[::-1])
        input_tensor = torch.from_numpy(im_tensor).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            model_orig.model(input_tensor.to(device_orig))

        set_model_image_names(model_dpa.model, [img_path.stem])
        with torch.no_grad():
            model_dpa.model(input_tensor.to(device_dpa))

        hm_orig = cv2.resize(
            process_feature(feat_orig_list[0][0], dh, dw, h_new, w_new, imgsz),
            (anchor_w, anchor_h),
            interpolation=cv2.INTER_LINEAR,
        )
        blend_orig = overlay_heatmap_dynamic_alpha(original_img, hm_orig)

        hm_dpa = cv2.resize(
            process_feature(feat_dpa_list[0][0], dh, dw, h_new, w_new, imgsz),
            (anchor_w, anchor_h),
            interpolation=cv2.INTER_LINEAR,
        )
        blend_dpa = overlay_heatmap_dynamic_alpha(original_img, hm_dpa)

        sep_v = np.ones((anchor_h, SEP, 3), dtype=np.uint8) * 255
        row_images.append(np.hstack((original_img, sep_v, blend_orig, sep_v, blend_dpa)))

    handle_orig.remove()
    handle_dpa.remove()

    if not row_images:
        raise RuntimeError("未成功处理任何图片。")

    sep_h = np.ones((SEP, row_images[0].shape[1], 3), dtype=np.uint8) * 255
    grid_rows = []
    for i, row in enumerate(row_images):
        grid_rows.append(row)
        if i < len(row_images) - 1:
            grid_rows.append(sep_h)
    grid_img = np.vstack(grid_rows)

    header, header_height = _make_title_header(COL_TITLES, anchor_w, SEP, grid_img.shape[1])
    final_combined = np.vstack((header, grid_img))
    meta = {
        "anchor_h": anchor_h,
        "anchor_w": anchor_w,
        "sep": SEP,
        "header_height": header_height,
        "n_rows": len(row_images),
        "row_stems": row_stems,
    }
    return final_combined, meta


# ---------------------------------------------------------------------------
# 3×3 红框 GUI：三行样本各自独立画框，每行内同步 3 列
# ---------------------------------------------------------------------------
class Grid3x3ROIGUI:
    """每行在左侧 Input 拖拽；红框只同步该行三列，三行互不影响。"""

    def __init__(self, image_rgb: np.ndarray, image_path: Path, output_dir: Path, meta: dict):
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image = image_rgb
        self.meta = meta

        self.anchor_h = meta["anchor_h"]
        self.anchor_w = meta["anchor_w"]
        self.sep = meta["sep"]
        self.header_h = meta["header_height"]
        self.n_rows = meta["n_rows"]

        self.panels, self.panel_origins = self._extract_panels()
        self.row_labels = meta.get("row_stems", [f"Row{i+1}" for i in range(self.n_rows)])

        # {"row": 行号, "roi": (x0, y0, x1, y1)}
        self.rois: list[dict] = []
        self.rectangles: list[Rectangle] = []
        self._last_row = 0

        self.fig = plt.figure(figsize=(16, 14), constrained_layout=True)
        gs_main = self.fig.add_gridspec(2, 1, height_ratios=[3.2, 1])
        gs_grid = gs_main[0].subgridspec(self.n_rows, 3, hspace=0.06, wspace=0.04)
        self.axes = []
        for r in range(self.n_rows):
            row_axes = [self.fig.add_subplot(gs_grid[r, c]) for c in range(3)]
            self.axes.append(row_axes)
        self.ax_preview = self.fig.add_subplot(gs_main[1])
        self._draw_grid()

        self.selectors = []
        for r in range(self.n_rows):
            sel = RectangleSelector(
                self.axes[r][0],
                lambda e1, e2, row=r: self.on_select(e1, e2, row),
                useblit=True,
                button=[1],
                minspanx=3,
                minspany=3,
                spancoords="pixels",
                interactive=False,
                props=dict(facecolor="none", edgecolor="red", linewidth=2),
            )
            self.selectors.append(sel)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        stems = " / ".join(self.row_labels)
        self.fig.suptitle(
            f"3×3 对比 | 样本: {stems}\n"
            "每行在左侧 Input 拖拽 → 仅同步该行 3 列 | Enter/s: 保存 | z: 撤销 | c: 清空 | q/Esc: 退出",
            fontsize=12,
        )

    def _extract_panels(self):
        panels, origins = [], []
        for r in range(self.n_rows):
            for c in range(3):
                gy0 = self.header_h + r * (self.anchor_h + self.sep)
                gx0 = c * (self.anchor_w + self.sep)
                panel = self.image[gy0 : gy0 + self.anchor_h, gx0 : gx0 + self.anchor_w].copy()
                panels.append(panel)
                origins.append((gx0, gy0))
        return panels, origins

    def _draw_grid(self):
        idx = 0
        for r, row_axes in enumerate(self.axes):
            for c, ax in enumerate(row_axes):
                ax.imshow(self.panels[idx])
                if r == 0:
                    ax.set_title(COL_TITLES[c], fontsize=12, color="darkred" if c == 2 else "black")
                ax.set_ylabel(self.row_labels[r] if c == 0 else "", fontsize=10)
                ax.axis("off")
                idx += 1
        self.ax_preview.axis("off")
        self.ax_preview.set_title("最新 ROI 预览（当前行三列）", fontsize=11)

    def _panel_index(self, row: int, col: int) -> int:
        return row * 3 + col

    def on_select(self, eclick, erelease, row: int):
        if eclick.xdata is None or erelease.xdata is None:
            return
        panel_idx = self._panel_index(row, 0)
        img_h, img_w = self.panels[panel_idx].shape[:2]
        x0 = int(round(min(eclick.xdata, erelease.xdata)))
        x1 = int(round(max(eclick.xdata, erelease.xdata)))
        y0 = int(round(min(eclick.ydata, erelease.ydata)))
        y1 = int(round(max(eclick.ydata, erelease.ydata)))
        x0 = max(0, min(x0, img_w - 1))
        x1 = max(x0 + 1, min(x1, img_w))
        y0 = max(0, min(y0, img_h - 1))
        y1 = max(y0 + 1, min(y1, img_h))
        roi = (x0, y0, x1, y1)
        self.rois.append({"row": row, "roi": roi})
        self._last_row = row
        self._add_roi_patches(roi, row)
        self._update_preview(roi, row)
        print(f"第 {row + 1} 行 [{self.row_labels[row]}] 添加 ROI: {roi}")
        self.fig.canvas.draw_idle()

    def _add_roi_patches(self, roi, row: int):
        x0, y0, x1, y1 = roi
        w, h = x1 - x0, y1 - y0
        for ax in self.axes[row]:
            rect = Rectangle((x0, y0), w, h, fill=False, edgecolor="red", linewidth=2.2)
            ax.add_patch(rect)
            self.rectangles.append(rect)

    def _update_preview(self, roi, row: int):
        x0, y0, x1, y1 = roi
        crops = [self.panels[self._panel_index(row, c)][y0:y1, x0:x1] for c in range(3)]
        sep = np.ones((crops[0].shape[0], SEP, 3), dtype=np.uint8) * 255
        preview = crops[0]
        for c in range(1, 3):
            preview = np.hstack((preview, sep, crops[c]))
        self.ax_preview.clear()
        self.ax_preview.imshow(preview)
        self.ax_preview.axis("off")
        n_row = sum(1 for item in self.rois if item["row"] == row)
        self.ax_preview.set_title(
            f"第 {row + 1} 行 [{self.row_labels[row]}] ROI: x={x0}:{x1}, y={y0}:{y1}  "
            f"|  该行 {n_row} 个框，总计 {len(self.rois)} 个",
            fontsize=11,
        )

    def _redraw_all(self):
        for row_axes in self.axes:
            for ax in row_axes:
                ax.clear()
                ax.axis("off")
        self.rectangles.clear()
        self._draw_grid()
        for item in self.rois:
            self._add_roi_patches(item["roi"], item["row"])
        if self.rois:
            last = self.rois[-1]
            self._update_preview(last["roi"], last["row"])
        self.fig.canvas.draw_idle()

    def save_annotated(self):
        if not self.rois:
            print("尚未画框：请在任意一行的 Input 上拖拽，再按 Enter 或 s 保存。")
            return
        annotated = Image.fromarray(self.image).convert("RGB")
        draw = ImageDraw.Draw(annotated)
        for item in self.rois:
            row = item["row"]
            x0, y0, x1, y1 = item["roi"]
            for c in range(3):
                gx0, gy0 = self.panel_origins[self._panel_index(row, c)]
                draw.rectangle(
                    [gx0 + x0, gy0 + y0, gx0 + x1, gy0 + y1],
                    outline=(255, 0, 0),
                    width=4,
                )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"{self.image_path.stem}_red_boxes_{ts}.png"
        annotated.save(out_path)
        print(f"已保存: {out_path.resolve()}")

    def on_key(self, event):
        if event.key is None:
            return
        key = event.key.lower()
        if key in ("enter", "return", "s"):
            self.save_annotated()
        elif key == "z":
            if self.rois:
                removed = self.rois.pop()
                print(f"撤销 ROI: {removed}")
                self._redraw_all()
            else:
                print("没有可撤销的框。")
        elif key == "c":
            self.rois.clear()
            self._redraw_all()
            print("已清空所有框。")
        elif key in ("q", "escape"):
            plt.close(self.fig)

    def show(self):
        plt.show()


def main(interactive: bool = False):
    ROI_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    final_img, meta = build_grid3x3(IMG_PATHS)
    grid_path = OUT_DIR / f"Grid3x3_Comparison_Layer{LAYER_INDEX}_red_boxes_20260528_204117.png"
    cv2.imwrite(str(grid_path), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    print(f"\n✅ 3×3 对比图已保存: {grid_path.resolve()}")

    if interactive:
        print("\n📌 打开标注窗口（请先点击窗口再按键）...")
        gui = Grid3x3ROIGUI(
            image_rgb=final_img,
            image_path=grid_path,
            output_dir=ROI_EXPORT_DIR,
            meta=meta,
        )
        gui.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the Figure 11 feature-response comparison")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open the optional GUI for drawing synchronized red ROI boxes",
    )
    args = parser.parse_args()
    main(interactive=args.interactive)
