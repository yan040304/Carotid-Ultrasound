import os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage
from sklearn.cluster import KMeans
import traceback

INPUT_DIR = "data/longitudinal/image"
LABEL_DIR = "data/longitudinal/label"
OUTPUT_DIR = "results/traditional/longitudinal"
DIAG_DIR = "results/diagnosis/longitudinal"

# subfolders for outputs
OUT_MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
OUT_VIS_DIR = os.path.join(OUTPUT_DIR, "vis")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_MASK_DIR, "lumen"), exist_ok=True)
os.makedirs(os.path.join(OUT_MASK_DIR, "vessel"), exist_ok=True)
os.makedirs(os.path.join(OUT_MASK_DIR, "plaque"), exist_ok=True)
os.makedirs(OUT_VIS_DIR, exist_ok=True)


# ------------ Dice / HD95 ------------
def calculate_dice(pred_mask, gt_mask):
    if pred_mask is None or gt_mask is None:
        return None
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()
    return (2 * inter / union) if union > 0 else 0.0


def calculate_hd95(pred_mask, gt_mask):
    if pred_mask is None or gt_mask is None:
        return None
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_pts = np.argwhere(pred_mask > 0)
    gt_pts = np.argwhere(gt_mask > 0)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")
    # directed_hausdorff returns (distance, idx1, idx2)
    a = directed_hausdorff(pred_pts, gt_pts)[0]
    b = directed_hausdorff(gt_pts, pred_pts)[0]
    return max(a, b)


# ------------ 分割 ------------
class LongitudinalCarotidSegmenter:

    def preprocess(self, img):
        # img: single-channel uint8
        clahe = cv2.createCLAHE(2.0, (8, 8))
        blurred = cv2.GaussianBlur(img, (7, 7), 2)
        return clahe.apply(blurred)

    def segment(self, img):
        pre = self.preprocess(img)
        edges = cv2.Canny(pre, 30, 120)
        linked = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((8, 8), np.uint8))

        num, lbl, stats, _ = cv2.connectedComponentsWithStats(linked, connectivity=8)
        mask = np.zeros_like(linked)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= 20:
                mask[lbl == i] = 255

        filled = ndimage.binary_fill_holes(mask).astype(np.uint8) * 255
        return filled


# ------------ 粥样化检测 ------------
def detect_atherosclerosis(gray, mask_lumen, mask_vessel, area_threshold=50):

    # ensure vessel mask provided
    if mask_vessel is None or np.sum(mask_vessel > 0) == 0:
        return 0, 0, np.zeros_like(gray), "no_vessel_pixels"

    wall_pixels = gray[mask_vessel > 0]
    if wall_pixels.size < 20:
        return 0, 0, np.zeros_like(gray), "too_few_pixels"

    # KMeans requires 2D array
    try:
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        km.fit(wall_pixels.reshape(-1, 1))
    except Exception as e:
        return 0, 0, np.zeros_like(gray), f"kmeans_error:{e}"

    centers = km.cluster_centers_.flatten()
    labels = km.labels_
    plaque_cluster = int(np.argmax(centers))

    wall_coords = np.argwhere(mask_vessel > 0)
    plaque_mask = np.zeros_like(gray)
    # safety check: labels length must match wall_coords length
    if labels.shape[0] != wall_coords.shape[0]:
        # Defensive: try to map using min length
        minlen = min(labels.shape[0], wall_coords.shape[0])
        for idx in range(minlen):
            y, x = wall_coords[idx]
            if labels[idx] == plaque_cluster:
                plaque_mask[y, x] = 255
    else:
        for idx, (y, x) in enumerate(wall_coords):
            if labels[idx] == plaque_cluster:
                plaque_mask[y, x] = 255

    area = int(np.sum(plaque_mask > 0))
    return (1 if area >= area_threshold else 0), area, plaque_mask, f"area={area}"


# ------------ 批处理 ------------
def save_mask(path, mask):
    # ensure mask is uint8
    if mask is None:
        return
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(path, mask)


def make_overlay(gray, lumen_mask, plaque_mask):
    # create BGR image from gray
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = bgr.copy()
    # lumen: draw in green, plaque in red (on top)
    # apply with some alpha
    alpha = 0.03
    lumen_area = (lumen_mask > 0)
    plaque_area = (plaque_mask > 0)

    overlay[lumen_area, 1] = 255  # green channel
    overlay[plaque_area, 2] = 255  # red channel

    vis = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)
    return vis


def batch_segment_longitudinal():

    seg = LongitudinalCarotidSegmenter()
    results = []

    filelist = sorted(os.listdir(INPUT_DIR))
    if len(filelist) == 0:
        print(f"[警告] 输入文件夹为空: {INPUT_DIR}")
        return

    for fname in filelist:
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        in_path = os.path.join(INPUT_DIR, fname)
        label_path = os.path.join(LABEL_DIR, fname)
        try:
            img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[跳过] 无法读取图像: {in_path}")
                continue

            gt = None
            if os.path.exists(label_path):
                gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    print(f"[注意] 找到标签文件但无法读取（可能非图像）: {label_path}; 将在 CSV 中记为 null")
            else:
                # 标签不存在时仍然继续生成分割与可视化
                gt = None

            # segmentation
            lumen = seg.segment(img)
            # vessel = dilate(lumen) - lumen
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            vessel = cv2.dilate(lumen, kernel) - lumen
            vessel = (vessel > 0).astype(np.uint8) * 255

            # detect plaque
            athero, area, plaque_mask, debug = detect_atherosclerosis(img, lumen, vessel)

            # calc metrics (if gt exists)
            dice = calculate_dice(lumen, gt) if gt is not None else None
            hd95 = calculate_hd95(lumen, gt) if gt is not None else None

            # print summary
            dice_str = f"{dice:.2f}" if dice is not None else "null"
            hd95_str = f"{hd95:.2f}" if (hd95 is not None and np.isfinite(hd95)) else ("inf" if hd95 == float("inf") else "null")
            print(f"[处理] {fname} | Dice={dice_str} | HD95={hd95_str} | plaque_area={area} | athero={athero} | {debug}")

            # save masks and visualization
            base = os.path.splitext(fname)[0]
            save_mask(os.path.join(OUT_MASK_DIR, "lumen", f"{base}_lumen.png"), lumen)
            save_mask(os.path.join(OUT_MASK_DIR, "vessel", f"{base}_vessel.png"), vessel)
            save_mask(os.path.join(OUT_MASK_DIR, "plaque", f"{base}_plaque.png"), plaque_mask)

            vis = make_overlay(img, lumen, plaque_mask)
            cv2.imwrite(os.path.join(OUT_VIS_DIR, f"{base}_overlay.png"), vis)

            results.append({
                "filename": fname,
                "dice": round(dice, 2) if dice is not None else None,
                "hd95": (round(hd95, 2) if (hd95 is not None and np.isfinite(hd95)) else (None if hd95 is None else float("inf"))),
                "athero": int(athero),
                "plaque_area": int(area),
                "debug": debug
            })

        except Exception as e:
            print(f"[错误] 处理 {fname} 时发生异常: {e}")
            traceback.print_exc()
            results.append({
                "filename": fname,
                "dice": None,
                "hd95": None,
                "athero": None,
                "plaque_area": None,
                "debug": f"exception:{str(e)}"
            })

    df = pd.DataFrame(results)
    csv_path = os.path.join(DIAG_DIR, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n已写入 {csv_path}")
    print(f"分割 masks 保存在: {OUT_MASK_DIR}")
    print(f"可视化 overlay 保存在: {OUT_VIS_DIR}")


if __name__ == "__main__":
    batch_segment_longitudinal()
