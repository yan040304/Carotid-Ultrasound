import os
import glob
import csv
import numpy as np
from PIL import Image
from typing import Tuple, List


def calculate_lumen_area(mask_arr: np.ndarray) -> float:
	"""Calculate lumen area from binary mask (pixels > 0)."""
	return float((mask_arr > 0).sum())


def calculate_lumen_diameter(mask_arr: np.ndarray) -> float:
	"""Calculate lumen diameter from binary mask using equivalent circle diameter."""
	area = (mask_arr > 0).sum()
	if area == 0:
		return 0.0
	# Equivalent circle diameter: d = 2 * sqrt(area / pi)
	diameter = 2.0 * np.sqrt(area / np.pi)
	return diameter


def find_minimal_lumen_area(mask_paths: List[str]) -> Tuple[float, str]:
	"""Find the minimal lumen area across all transverse masks."""
	min_area = float('inf')
	min_file = ""
	
	for mask_path in mask_paths:
		mask = Image.open(mask_path).convert('L')
		mask_arr = np.array(mask, dtype=np.uint8)
		area = calculate_lumen_area(mask_arr)
		if area < min_area and area > 0:
			min_area = area
			min_file = os.path.basename(mask_path)
	
	return min_area if min_area != float('inf') else 0.0, min_file


def find_reference_lumen_area(mask_paths: List[str]) -> Tuple[float, str]:
	"""Find the reference lumen area (largest area) across all transverse masks."""
	max_area = 0.0
	max_file = ""
	
	for mask_path in mask_paths:
		mask = Image.open(mask_path).convert('L')
		mask_arr = np.array(mask, dtype=np.uint8)
		area = calculate_lumen_area(mask_arr)
		if area > max_area:
			max_area = area
			max_file = os.path.basename(mask_path)
	
	return max_area, max_file


def calculate_transverse_csr(mask_dir: str, output_csv: str) -> None:
	"""Calculate Carotid Stenosis Rate for transverse views."""
	os.makedirs(os.path.dirname(output_csv), exist_ok=True)
	
	# Get all mask files
	mask_paths = []
	for ext in ("png", "jpg", "jpeg", "bmp", "tiff"):
		mask_paths.extend(glob.glob(os.path.join(mask_dir, f"*_mask.{ext}")))
		mask_paths.extend(glob.glob(os.path.join(mask_dir, f"*_mask.{ext.upper()}")))
	
	if not mask_paths:
		print(f"未在 {mask_dir} 找到掩码文件")
		return
	
	# Find minimal and reference areas
	min_area, min_file = find_minimal_lumen_area(mask_paths)
	ref_area, ref_file = find_reference_lumen_area(mask_paths)
	
	# Calculate CSR for each mask
	rows = [("filename", "lumen_area", "reference_area", "csr_percent")]
	
	for mask_path in sorted(mask_paths):
		fname = os.path.basename(mask_path)
		mask = Image.open(mask_path).convert('L')
		mask_arr = np.array(mask, dtype=np.uint8)
		area = calculate_lumen_area(mask_arr)
		
		# CSR = (1 - (current_area / reference_area)) × 100%
		if ref_area > 0:
			csr = (1.0 - (area / ref_area)) * 100.0
		else:
			csr = 0.0
		
		rows.append((fname, f"{area:.2f}", f"{ref_area:.2f}", f"{csr:.2f}"))
	
	# Write results
	with open(output_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerows(rows)
	
	print(f"横切CSR计算完成: {output_csv}")
	print(f"最小管腔面积: {min_area:.2f} ({min_file})")
	print(f"参考管腔面积: {ref_area:.2f} ({ref_file})")
	print(f"最大狭窄率: {(1.0 - (min_area / ref_area)) * 100.0:.2f}%" if ref_area > 0 else "无法计算CSR")


def calculate_longitudinal_csr(mask_dir: str, output_csv: str) -> None:
	"""Calculate Carotid Stenosis Rate for longitudinal views using diameter."""
	os.makedirs(os.path.dirname(output_csv), exist_ok=True)
	
	# Get all mask files
	mask_paths = []
	for ext in ("png", "jpg", "jpeg", "bmp", "tiff"):
		mask_paths.extend(glob.glob(os.path.join(mask_dir, f"*_mask.{ext}")))
		mask_paths.extend(glob.glob(os.path.join(mask_dir, f"*_mask.{ext.upper()}")))
	
	if not mask_paths:
		print(f"未在 {mask_dir} 找到掩码文件")
		return
	
	# Calculate diameter for each mask (lumen = label 100)
	rows = [("filename", "lumen_diameter", "reference_diameter", "csr_percent")]
	
	diameters = []
	for mask_path in sorted(mask_paths):
		fname = os.path.basename(mask_path)
		mask = Image.open(mask_path).convert('L')
		mask_arr = np.array(mask, dtype=np.uint8)
		
		# Extract lumen region (label 100)
		lumen_mask = (mask_arr == 100)
		diameter = calculate_lumen_diameter(lumen_mask.astype(np.uint8))
		diameters.append((diameter, fname))
	
	# Find reference diameter (maximum)
	ref_diameter = max(diameters, key=lambda x: x[0])[0] if diameters else 0.0
	
	# Calculate CSR for each mask
	for mask_path in sorted(mask_paths):
		fname = os.path.basename(mask_path)
		mask = Image.open(mask_path).convert('L')
		mask_arr = np.array(mask, dtype=np.uint8)
		
		lumen_mask = (mask_arr == 100)
		diameter = calculate_lumen_diameter(lumen_mask.astype(np.uint8))
		
		# CSR = (1 - (current_diameter / reference_diameter)) × 100%
		if ref_diameter > 0:
			csr = (1.0 - (diameter / ref_diameter)) * 100.0
		else:
			csr = 0.0
		
		rows.append((fname, f"{diameter:.2f}", f"{ref_diameter:.2f}", f"{csr:.2f}"))
	
	# Write results
	with open(output_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerows(rows)
	
	print(f"纵切CSR计算完成: {output_csv}")
	print(f"参考管腔直径: {ref_diameter:.2f}")
	print(f"各图像CSR已分别计算")


if __name__ == "__main__":
	# Calculate CSR for both views
	transverse_mask_dir = "results/deep_learning/transverse"
	longitudinal_mask_dir = "results/deep_learning/longitudinal"
	
	transverse_csv = "results/diagnosis/transverse/csr.csv"
	longitudinal_csv = "results/diagnosis/longitudinal/csr.csv"
	
	print("=== 颈动脉狭窄率 (CSR) 计算 ===")
	print("\n1. 横切面CSR计算:")
	calculate_transverse_csr(transverse_mask_dir, transverse_csv)
	
	print("\n2. 纵切面CSR计算:")
	calculate_longitudinal_csr(longitudinal_mask_dir, longitudinal_csv)
	
	print("\nCSR计算完成！")
