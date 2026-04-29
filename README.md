# Carotid Ultrasound Segmentation Project

This repository implements both a traditional rule-based pipeline and a deep learning pipeline to segment longitudinal and transverse carotid ultrasound images, evaluate overlap metrics (Dice, HD95), estimate carotid stenosis rate (CSR), and diagnose plaque formation through thresholding.

## Repository Layout

- `code/longitudinal/traditional.py` – Classic longitudinal segmentation, assessment, and determination of lesion occurrence.
- `code/transverse/traditional.py` – Classic transverse segmentation, assessment, and determination of lesion occurrence.
- `code/longitudinal/infer_v.py`, `code/transverse/infer.py` – UNet inference scripts that generate deep learning masks.
- `code/longitudinal/metrics_evaluation.py`, `code/transverse/metrics_evaluation.py` – Script to calculate Dice, HD95, and CSR.
- `code/csr_calculation.py` – CSR calculator.
- `data/` – input ultrasound images (`image/`) and ground truth (`label/`).
- `results/` – automatically created folders for masks, diagnostics, CSR exports, and visualisations.
- `report.tex` – LaTeX report describing the methodology and current results.


## Traditional Longitudinal Pipeline

1. Place the source images in `data/longitudinal/image` and masks in `data/longitudinal/label`.
2. Run:

   ```bash
   python code/longitudinal/traditional.py
   ```

3. Outputs:
   - Binary masks saved to `results/traditional/longitudinal/`.
   - `results/diagnosis/longitudinal/evaluation_results.csv` with Dice and HD95 per case 
   - `results/diagnosis/longitudinal/csr.csv` summarising CSR per slice
   - `results/diagnosis/longitudinal/method_metrics.csv` with classification metrics

## Deep Learning Inference

1. Download the pretrained checkpoints (e.g., `code/transverse/carotid_unet.pth`, `code/longitudinal/carotid_unet_longitudinal.pth`).
2. Run inference (example for longitudinal):

   ```bash
   python code/longitudinal/infer_v.py
   ```

   Masks are written to `results/deep_learning/longitudinal/` using the `_mask` suffix. Repeat for transverse images via `code/transverse/infer.py`.

3. Optional: Create visual overlays in `results/deep_learning/longitudinal_visualizations/`.

The traditional script automatically consumes the deep learning masks to perform plaque thresholding once both prediction and label folders exist.

## CSR Only

If you only need CSR from existing masks, run:

```bash
python code/csr_calculation.py
```

This script reads masks from `results/deep_learning/{view}` and writes CSV summaries to `results/diagnosis/{view}/`.

## Report Generation

`report.tex` documents the full workflow, quantitative metrics (Dice/HD95/CSR ranges, plaque classification results), and discussion. Update numerical values by re-running the traditional script and re-compiling the LaTeX:

```bash
pdflatex report.tex
```