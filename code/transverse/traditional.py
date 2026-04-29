import os
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.feature import canny
from sklearn.cluster import KMeans


INPUT_DIR = "data/transverse/image"
LABEL_DIR = "data/transverse/label"
OUTPUT_DIR = "results/traditional/transverse"
DIAG_DIR = "results/diagnosis/transverse"

# -------- 创建输出文件夹 --------
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
VIS_DIR = os.path.join(OUTPUT_DIR, "vis")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(os.path.join(MASK_DIR, "lumen"), exist_ok=True)
os.makedirs(os.path.join(MASK_DIR, "vessel"), exist_ok=True)
os.makedirs(os.path.join(MASK_DIR, "plaque"), exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


# ---------------- Dice / HD95 ----------------
def calculate_dice(pred_mask, gt_mask):
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]))

    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum()

    return 2.0 * intersection / union if union > 0 else 0.0


def calculate_hd95(pred_mask, gt_mask):
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]))

    pred_points = np.argwhere(pred_mask > 0)
    gt_points = np.argwhere(gt_mask > 0)

    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('inf')

    hd1 = directed_hausdorff(pred_points, gt_points)[0]
    hd2 = directed_hausdorff(gt_points, pred_points)[0]
    return max(hd1, hd2)


# ---------------- 预处理 ----------------
def preprocess_carotid_image(img):
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)

    denoised_img = cv2.medianBlur(img, 5)
    smoothed = cv2.GaussianBlur(denoised_img, (5, 5), 1.5)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)

    gaussian_blur = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian_blur, -0.5, 0)

    return sharpened


# ---------------- 主体分割 ----------------
def initialize_contour_with_label(gray, label_img, num_points=100):
    coords = np.argwhere(label_img > 0)
    yc, xc = coords.mean(axis=0)
    radius = np.sqrt(coords.shape[0] / np.pi)

    angles = np.linspace(0, 2*np.pi, num_points)
    x = xc + radius * np.cos(angles)
    y = yc + radius * np.sin(angles)

    return np.stack([y, x], axis=1)


def segment_with_active_contour_label(img, label_img):
    gray = img.astype(np.float32) / 255.0
    init = initialize_contour_with_label(gray, label_img)

    smoothed = gaussian(gray, sigma=2)
    edges = canny(smoothed, sigma=2)
    edge_img = 1.0 - edges.astype(float)

    snake = active_contour(
        edge_img, init,
        alpha=0.01, beta=1.0, gamma=0.02,
        w_line=0, w_edge=3.0,
        max_num_iter=700, convergence=0.05
    )

    mask = np.zeros_like(gray, dtype=np.uint8)
    pts = np.array([np.fliplr(snake)], np.int32)
    cv2.fillPoly(mask, pts, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = (mask > 0).astype(np.uint8) * 255

    return mask


# ---------------- 粥样化判断 ----------------
def detect_atherosclerosis(gray, mask_lumen, mask_vessel, area_threshold=100):
    wall_pixels = gray[mask_vessel > 0]

    if len(wall_pixels) < 30:
        return 0, 0, "too_few_pixels", np.zeros_like(gray)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(wall_pixels.reshape(-1, 1))

    centers = km.cluster_centers_.flatten()
    labels = km.labels_
    plaque_clu = int(np.argmax(centers))

    wall_coords = np.argwhere(mask_vessel > 0)

    plaque_mask = np.zeros_like(gray)
    for idx, (y, x) in enumerate(wall_coords):
        if labels[idx] == plaque_clu:
            plaque_mask[y, x] = 255

    area = int(np.sum(plaque_mask > 0))
    return (1 if area >= area_threshold else 0), area, f"area={area}", plaque_mask


# ---------------- 可视化 ----------------
def make_overlay(gray, lumen, plaque):
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    overlay[lumen > 0] = [0, 255, 0]      # green
    overlay[plaque > 0] = [0, 0, 255]     # red
    return cv2.addWeighted(base, 0.6, overlay, 0.4, 0)


# ---------------- 批处理 ----------------
def batch_segment_transverse():

    results = []

    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img = cv2.imread(os.path.join(INPUT_DIR, fname), 0)
        gt = cv2.imread(os.path.join(LABEL_DIR, fname), 0)

        pre = preprocess_carotid_image(img)
        seg_mask = segment_with_active_contour_label(pre, gt)

        dice = calculate_dice(seg_mask, gt)
        hd95 = calculate_hd95(seg_mask, gt)

        # ------------ 生成血管壁 ------------
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_lumen = seg_mask
        mask_vessel = cv2.dilate(mask_lumen, kernel) - mask_lumen
        mask_vessel = (mask_vessel > 0).astype(np.uint8) * 255

        # ------------ 粥样化检测 ------------
        athero, area, debug, plaque_mask = detect_atherosclerosis(img, mask_lumen, mask_vessel)

        print(f"[处理 {fname}] Dice={dice:.2f}, HD95={hd95:.2f}, plaque_area={area}")

        # ------------ 保存所有 mask ------------
        base = fname.rsplit(".", 1)[0]

        cv2.imwrite(os.path.join(MASK_DIR, "lumen", f"{base}_lumen.png"), mask_lumen)
        cv2.imwrite(os.path.join(MASK_DIR, "vessel", f"{base}_vessel.png"), mask_vessel)
        cv2.imwrite(os.path.join(MASK_DIR, "plaque", f"{base}_plaque.png"), plaque_mask)

        # ------------ 保存可视化 ------------
        vis = make_overlay(img, mask_lumen, plaque_mask)
        cv2.imwrite(os.path.join(VIS_DIR, f"{base}_overlay.png"), vis)

        # ------------ CSV ------------
        results.append({
            "filename": fname,
            "dice": round(dice, 2),
            "hd95": round(hd95, 2),
            "plaque_area": area,
            "athero": int(athero)
        })

    pd.DataFrame(results).to_csv(os.path.join(DIAG_DIR, "evaluation_results.csv"), index=False)
    print("\n已写入 evaluation_results.csv")
    print("分割结果已输出到:", OUTPUT_DIR)


if __name__ == "__main__":
    batch_segment_transverse()
