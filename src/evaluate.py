"""Evaluate denoised results against clean ground truth.

Calculates per-image PSNR/SSIM and the competition final score.

Usage:
    python src/evaluate.py \
        --result_dir results/val \
        --clean_dir data/train/clean \
        --noisy_dir data/train/noisy
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate denoising results")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory of denoised images.")
    parser.add_argument("--clean_dir", type=str, required=True,
                        help="Directory of clean ground truth images.")
    parser.add_argument("--noisy_dir", type=str, default=None,
                        help="Directory of noisy images (optional, for baseline comparison).")
    return parser.parse_args()


def load_img(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim_channel(img1: np.ndarray, img2: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean())


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.ndim == 2:
        return ssim_channel(img1, img2)
    return float(np.mean([ssim_channel(img1[:, :, i], img2[:, :, i]) for i in range(3)]))


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    clean_dir = Path(args.clean_dir)

    # Find matching image pairs
    exts = {".png", ".jpg", ".jpeg"}
    result_files = sorted(
        [p for p in result_dir.iterdir() if p.suffix.lower() in exts],
        key=lambda p: p.name,
    )

    if not result_files:
        print(f"No result images found in {result_dir}")
        return 1

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    rows = []

    for res_path in result_files:
        # Map noisy filename to clean filename
        clean_name = res_path.name.replace("_noisy", "_clean")
        clean_path = clean_dir / clean_name

        if not clean_path.exists():
            print(f"  SKIP: {res_path.name} (no matching clean image)")
            continue

        pred = load_img(res_path)
        clean = load_img(clean_path)

        if pred.shape != clean.shape:
            print(f"  ERROR: Shape mismatch for {res_path.name}: "
                  f"pred={pred.shape}, clean={clean.shape}")
            continue

        psnr = calculate_psnr(pred, clean)
        ssim_val = calculate_ssim(pred, clean)
        total_psnr += psnr
        total_ssim += ssim_val
        count += 1

        rows.append({
            "filename": res_path.name,
            "psnr": f"{psnr:.4f}",
            "ssim": f"{ssim_val:.6f}",
        })

    if count == 0:
        print("No valid image pairs found!")
        return 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    score = 0.7 * avg_psnr + 0.3 * (avg_ssim * 40.0)

    print(f"\n{'='*50}")
    print(f"  Evaluation Results ({count} images)")
    print(f"{'='*50}")
    print(f"  Avg PSNR:    {avg_psnr:.6f} dB")
    print(f"  Avg SSIM:    {avg_ssim:.8f}")
    print(f"  Final Score: {score:.6f}")
    print(f"{'='*50}")

    # Save CSV
    csv_path = result_dir / "eval_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-image metrics saved to: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
