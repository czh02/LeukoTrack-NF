import os
import cv2
import numpy as np

from scipy.ndimage import maximum_filter, median_filter, gaussian_laplace
from scipy.fft import fft, ifft

def read_images_from_folder(folder_path, max_images=None):
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])

    if max_images is not None:
        image_files = image_files[:max_images]

    images = []
    for file in image_files:
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img.astype(np.float32) / 255.0)
    return np.stack(images, axis=0)


def mean_filter(image, kernel_size=(9, 3)):
    kernel = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])
    return cv2.filter2D(image, -1, kernel)

def process_images(images):
    num_images, height, width = images.shape
    filt_images = np.zeros_like(images)
    for i in range(num_images):
        img = images[i]
        img = apply_clahe(img)
        # img = wavelet_enhance(img)  # replaced by wavelet transform
        img = mean_filter(img)
        # img = guided_filter(img,img, radius=10, eps=0.01)
        filt_images[i] = img
    return filt_images

def apply_clahe(image, clip_limit=4.0, tile_grid_size=(7, 5)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply((image * 255).astype(np.uint8)).astype(np.float32) / 255.0

def compute_angio_md(filt_images, cut_pos=18):
    num_images, height, width = filt_images.shape
    temp_images = filt_images.copy()
    time_spectrums = fft(temp_images, axis=0)
    spectrums_abs = np.abs(time_spectrums)
    high_freq = np.mean(spectrums_abs[cut_pos + 1:, :, :], axis=0)
    low_freq = np.mean(spectrums_abs[:cut_pos, :, :], axis=0)
    angio_md = high_freq / (low_freq + 1e-8)
    angio_md = (angio_md - angio_md.min()) / (angio_md.max() - angio_md.min())
    angio_md = median_filter(angio_md, size=3)
    angio_md = np.clip((angio_md - 0.05) / (1 - 0.05), 0, 1)
    angio_md *= 255
    return angio_md.astype(np.uint8)

def ifm_processing(filt_images):
    num_images, height, width = filt_images.shape
    temp_images = filt_images.copy()

    for i in range(num_images):
        temp_images[i] = median_filter(temp_images[i], size=3)  # median-filter denoising
        # or bilateral filtering (edge-preserving)
        temp_images[i] = cv2.bilateralFilter(temp_images[i], d=5, sigmaColor=0.8, sigmaSpace=3)

    time_spectrums = fft(temp_images, axis=0)
    spectrums_signal_high = time_spectrums.copy()
    spectrums_signal_low = time_spectrums.copy()

    spectrums_signal_low[10:, :, :] = 0
    spectrums_signal_high[:4, :, :] = 0

    spectrums_signal_high[int(num_images/2):, :, :] = 0
    high = ifft(spectrums_signal_high, axis=0).real
    low = ifft(spectrums_signal_low, axis=0).real
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(high, low)
        result[~np.isfinite(result)] = 0

    result = np.true_divide(high, low + 1e-4)  # avoid amplifying background when the denominator is tiny

    return result

def post_process_ifm(ifm_images):
    num_images, height, width = ifm_images.shape
    processed_images = np.zeros_like(ifm_images)

    for i in range(num_images):
        img = ifm_images[i]

        img = cv2.GaussianBlur(img, (3, 3), sigmaX=1)
        img = median_filter(img, size=3)
        img = cv2.bilateralFilter(img, d=5, sigmaColor=0.8, sigmaSpace=3)

        processed_images[i] = img

    return processed_images


def mask_by_vessel(angio_md, threshold=0.2):
    norm = angio_md.astype(np.float32) / 255.0
    mask = (norm > threshold).astype(np.float32)
    return mask

def apply_mask_to_ifm(ifm_images, vessel_mask):
    num_images, height, width = ifm_images.shape
    masked_ifm = np.zeros_like(ifm_images)
    mask = vessel_mask.astype(np.float32)

    for i in range(num_images):
        masked_ifm[i] = ifm_images[i] * mask

    return masked_ifm


def isolate_cells(ifm_image):
    # Step 1: simple contrast stretching
    img = ifm_image.copy()
    img = (img - np.percentile(img, 5)) / (np.percentile(img, 95) - np.percentile(img, 5) + 1e-8)
    img = np.clip(img, 0, 1)

    # Step 2: adaptive binarization
    binary = (img > 0.3).astype(np.uint8)  # threshold can be tuned in 0.25-0.4

    # Step 3: connected-component filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    mask = np.zeros_like(binary)

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if 10 < area < 800:  # relaxed bounds to keep moving targets
            mask[labels == i] = 1

    # Step 4: extract original-frame content with the mask
    result = img * mask
    return result


def overlay_images(filt_images, angio_md, vessel_mask=None):
    num_images, height, width = filt_images.shape
    md2_images = np.zeros_like(filt_images)
    angio_md_normalized = angio_md.astype(np.float32) / 255.0

    if vessel_mask is None:
        vessel_mask = np.ones_like(angio_md_normalized)
    else:
        vessel_mask = vessel_mask.astype(np.float32)

    global_min = np.min(filt_images)
    global_max = np.max(filt_images)

    for i in range(num_images):
        # img = filt_images[i]
        # img = safe_normalize(img)
        img = (filt_images[i] - global_min) / (global_max - global_min + 1e-6)
        # vessel-region overlay
        overlay = img + angio_md_normalized * vessel_mask
        overlay = safe_normalize(overlay)

        overlay = overlay * vessel_mask

        # contrast stretching
        # p_low, p_high = np.percentile(overlay[vessel_mask > 0], [1, 99])  # computed within the vessel region only
        # stretch_min = p_low
        # stretch_max = p_high
        # overlay = np.clip((overlay - stretch_min) / (stretch_max - stretch_min + 1e-6), 0, 1)
        overlay = np.clip((overlay - 0.05) / (0.9 - 0.05), 0, 1)  # linear contrast stretch
        md2_images[i] = overlay
    return md2_images

def guided_filter(I, p, radius=15, eps=1e-3):
    # compute means
    mean_I = cv2.boxFilter(I, -1, (radius, radius))
    mean_p = cv2.boxFilter(p, -1, (radius, radius))

    # compute covariances
    mean_II = cv2.boxFilter(I * I, -1, (radius, radius))
    mean_Ip = cv2.boxFilter(I * p, -1, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    # compute coefficients
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # aggregate coefficients
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    # output
    q = mean_a * I + mean_b
    return q

def save_images(images, output_folder, prefix):
    os.makedirs(output_folder, exist_ok=True)
    num_images = images.shape[0]
    for i in range(num_images):
        img = safe_cast_image(images[i])
        filename = os.path.join(output_folder, f"{prefix}_{i+1:03d}.png")
        cv2.imwrite(filename, img)

def safe_normalize(img):
    min_val, max_val = np.nanmin(img), np.nanmax(img)
    if max_val - min_val < 1e-8:
        return np.zeros_like(img)
    norm = (img - min_val) / (max_val - min_val)
    return np.nan_to_num(norm)

def safe_cast_image(image):
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def main():
    input_folder = r'D:\NailfoldAnalyzer\assets\Registration_NIFM'
    output_root = r'D:\NailfoldAnalyzer\assets'
    max_images = 300

    service = PreprocessService()
    service.run(input_folder, output_root, max_images)


class PreprocessService:
    def __init__(self, log_callback=None):
        """
        log_callback: callable(str) for Qt log output
        """
        self.log = log_callback if log_callback else print

    def run(self, input_folder, output_root, max_images=300):
        self.log(f"Starting image preprocessing: {input_folder}")

        output_filt_folder = os.path.join(output_root, 'filt')
        output_angio_folder = os.path.join(output_root, 'angio')
        output_ifm_folder = os.path.join(output_root, 'ifm')
        output_md2_folder = os.path.join(output_root, 'md2')

        for folder in [
            output_filt_folder,
            output_angio_folder,
            output_ifm_folder,
            output_md2_folder,
        ]:
            os.makedirs(folder, exist_ok=True)
            self.log(f"Creating output directory: {folder}")

        # 1. Read images
        images = read_images_from_folder(input_folder, max_images=max_images)
        self.log(f"Read {images.shape[0]} images")

        # 2. Preprocessing + filtering
        filt_images = process_images(images)
        save_images(filt_images, output_filt_folder, 'filt')
        self.log("Filtering done")

        # 3. AngioMD
        angio_md = compute_angio_md(filt_images)
        cv2.imwrite(os.path.join(output_angio_folder, 'AngioMD.png'), angio_md)
        self.log("AngioMD generated")

        # 4. IFM
        ifm_images = ifm_processing(filt_images)
        vessel_mask = mask_by_vessel(angio_md, threshold=0.1)
        for i in range(ifm_images.shape[0]):
            ifm_images[i] *= vessel_mask

        processed_ifm_images = post_process_ifm(ifm_images)
        save_images(processed_ifm_images, output_ifm_folder, 'ifm')
        self.log("IFM processing done")

        # 5. Fusion overlay
        md2_images = overlay_images(processed_ifm_images, angio_md)
        save_images(md2_images, output_md2_folder, 'md2')
        self.log("MD2 generated")

        self.log("Image preprocessing complete")
