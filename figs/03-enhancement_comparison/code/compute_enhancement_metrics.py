import cv2
import numpy as np
from skimage import filters
import argparse
import os


# ------------------------- Utility functions -------------------------
def to_gray(image):
    """Convert image to grayscale if it is color."""
    if image is None:
        raise ValueError("Input image is empty!")
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def normalize(img):
    """Normalize to [0, 1]."""
    img = img.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min < 1e-12:
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)


# ------------------------- SNR -------------------------
def compute_snr(image, signal_region=None, noise_region=None):

    gray = to_gray(image).astype(np.float64)

    if signal_region is not None and noise_region is not None:
        y1, y2, x1, x2 = signal_region
        signal = gray[y1:y2, x1:x2]

        ny1, ny2, nx1, nx2 = noise_region
        noise = gray[ny1:ny2, nx1:nx2]

        mean_signal = np.mean(signal)
        std_noise = np.std(noise)
    else:
        # use the whole image as signal; estimate noise from Laplacian high-frequency residual
        mean_signal = np.mean(gray)

        # estimate the smooth part with a median filter; treat the residual as noise
        smooth = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float64)
        residual = gray - smooth
        std_noise = np.std(residual)

    if std_noise < 1e-12:
        return float('inf')

    snr = 20 * np.log10(mean_signal / std_noise)
    return snr


# ------------------------- CNR -------------------------
def compute_cnr(image, roi_signal=None, roi_background=None):

    gray = to_gray(image).astype(np.float64)

    if roi_signal is not None and roi_background is not None:
        y1, y2, x1, x2 = roi_signal
        signal = gray[y1:y2, x1:x2]

        by1, by2, bx1, bx2 = roi_background
        background = gray[by1:by2, bx1:bx2]
    else:
        # automatic selection: Otsu threshold separates foreground/background
        img_u8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thresh = filters.threshold_otsu(img_u8)
        mask_bright = img_u8 > thresh
        mask_dark = ~mask_bright

        if mask_bright.sum() < 10 or mask_dark.sum() < 10:
            raise ValueError("Image contrast too low to separate foreground/background; specify the ROI manually.")

        signal = gray[mask_bright]
        background = gray[mask_dark]

    mean_s = np.mean(signal)
    mean_b = np.mean(background)
    std_s = np.std(signal)
    std_b = np.std(background)

    denom = np.sqrt(std_s**2 + std_b**2)
    if denom < 1e-12:
        return float('inf')

    cnr = abs(mean_s - mean_b) / denom
    return cnr


# ------------------------- Tenengrad -------------------------
def compute_tenengrad(image, threshold=None):

    gray = to_gray(image).astype(np.float64)

    # Sobel operators
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # sum of squared gradients
    grad_sq = gx**2 + gy**2

    if threshold is not None:
        mag = np.sqrt(grad_sq)
        mask = mag > threshold
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(grad_sq[mask]))

    return float(np.mean(grad_sq))


# ------------------------- Main routine -------------------------
def analyze_image(image_path,
                  signal_region=None,
                  noise_region=None,
                  roi_signal=None,
                  roi_background=None,
                  tenengrad_threshold=None):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # convert 16-bit images first so downstream functions can use them
    if image.dtype != np.uint8:
        image_disp = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        image_disp = image

    results = {}

    try:
        results['SNR_dB'] = compute_snr(image_disp, signal_region, noise_region)
    except Exception as e:
        results['SNR_dB'] = f"failed: {e}"

    try:
        results['CNR'] = compute_cnr(image_disp, roi_signal, roi_background)
    except Exception as e:
        results['CNR'] = f"failed: {e}"

    try:
        results['Tenengrad'] = compute_tenengrad(image_disp, tenengrad_threshold)
    except Exception as e:
        results['Tenengrad'] = f"failed: {e}"

    return results


def print_results(image_path, results):
    print("=" * 55)
    print(f"Image: {image_path}")
    print("-" * 55)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<12}: {v:.4f}")
        else:
            print(f"  {k:<12}: {v}")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Compute SNR / CNR / Tenengrad metrics of an image"
    )
    parser.add_argument("image", type=str, help="Path to the image file")
    parser.add_argument("--signal", type=int, nargs=4,
                        metavar=("Y1", "Y2", "X1", "X2"),
                        help="SNR signal ROI (y1 y2 x1 x2)")
    parser.add_argument("--noise", type=int, nargs=4,
                        metavar=("Y1", "Y2", "X1", "X2"),
                        help="SNR noise ROI (y1 y2 x1 x2)")
    parser.add_argument("--roi_signal", type=int, nargs=4,
                        metavar=("Y1", "Y2", "X1", "X2"),
                        help="CNR signal ROI (y1 y2 x1 x2)")
    parser.add_argument("--roi_bg", type=int, nargs=4,
                        metavar=("Y1", "Y2", "X1", "X2"),
                        help="CNR background ROI (y1 y2 x1 x2)")
    parser.add_argument("--tenengrad_thresh", type=float, default=None,
                        help="Tenengrad threshold (optional)")

    args = parser.parse_args()

    results = analyze_image(
        args.image,
        signal_region=tuple(args.signal) if args.signal else None,
        noise_region=tuple(args.noise) if args.noise else None,
        roi_signal=tuple(args.roi_signal) if args.roi_signal else None,
        roi_background=tuple(args.roi_bg) if args.roi_bg else None,
        tenengrad_threshold=args.tenengrad_thresh,
    )

    print_results(args.image, results)


if __name__ == "__main__":
    main()
