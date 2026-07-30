"""Fine-tune Restormer on competition data for PRCV2026 Challenge.

Optimized for low-VRAM GPUs (GTX 1650 Ti, 4GB):
- Small patch size (128x128)
- Batch size 1 with gradient accumulation
- Mixed precision (fp16) training
- Memory-efficient gradient checkpointing

Usage:
    E:\\Anaconda_env\\envs\\prcv2026\\python.exe src/train.py ^
        --pretrained pretrained_models/gaussian_color_denoising_sigma50.pth ^
        --train_noisy data/train/noisy ^
        --train_clean data/train/clean ^
        --output_dir experiments/finetune_v1 ^
        --patch_size 128 ^
        --batch_size 1 ^
        --grad_accum 4 ^
        --lr 2e-5 ^
        --total_iters 50000 ^
        --save_every 5000 ^
        --eval_every 2500
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_arch import Restormer, count_parameters


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DenoisingDataset(Dataset):
    """Paired noisy-clean dataset with on-the-fly augmentation.

    When use_synthetic_noise=True (Direction 3 / Domain-Gap fix):
      - Ignores the noisy images on disk.
      - Loads only the clean image and synthesizes fresh Gaussian noise
        with sigma randomly sampled from [sigma_min, sigma_max] every time.
      - Forces the model to generalise to the full range of noise levels
        instead of memorising the fixed-sigma training distribution.
    """

    def __init__(
        self,
        noisy_dir: str,
        clean_dir: str,
        patch_size: int = 128,
        augment: bool = True,
        use_synthetic_noise: bool = False,
        sigma_min: float = 40.0,
        sigma_max: float = 60.0,
    ):
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.use_synthetic_noise = use_synthetic_noise
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Find all matching pairs
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        noisy_files = sorted(
            [p for p in self.noisy_dir.iterdir() if p.suffix.lower() in exts],
            key=lambda p: p.name,
        )

        self.pairs = []
        for nf in noisy_files:
            cf = self.clean_dir / nf.name.replace("_noisy", "_clean")
            if cf.exists():
                self.pairs.append((nf, cf))

        mode = f"synthetic noise σ~U[{sigma_min},{sigma_max}]" if use_synthetic_noise else "paired (disk)"
        print(f"Dataset: {len(self.pairs)} pairs from {noisy_dir} | mode={mode}")

    def __len__(self):
        return len(self.pairs)

    def _load_img(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _random_crop(self, noisy: np.ndarray, clean: np.ndarray):
        h, w = noisy.shape[:2]
        ps = self.patch_size

        if h < ps or w < ps:
            # Pad if image is smaller than patch
            pad_h = max(ps - h, 0)
            pad_w = max(ps - w, 0)
            noisy = np.pad(noisy, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            clean = np.pad(clean, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            h, w = noisy.shape[:2]

        y = random.randint(0, h - ps)
        x = random.randint(0, w - ps)
        return noisy[y:y+ps, x:x+ps], clean[y:y+ps, x:x+ps]

    def _augment(self, noisy: np.ndarray, clean: np.ndarray):
        """Random flip and rotation (training-time only, allowed by rules)."""
        # Random horizontal flip
        if random.random() > 0.5:
            noisy = np.fliplr(noisy).copy()
            clean = np.fliplr(clean).copy()

        # Random vertical flip
        if random.random() > 0.5:
            noisy = np.flipud(noisy).copy()
            clean = np.flipud(clean).copy()

        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            noisy = np.rot90(noisy, k).copy()
            clean = np.rot90(clean, k).copy()

        return noisy, clean

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        clean = self._load_img(clean_path)

        if self.use_synthetic_noise:
            # --- Direction 3: Dynamic noise synthesis ---
            # Only crop from clean; synthesize noise AFTER crop to save memory
            # Crop clean image first
            h, w = clean.shape[:2]
            ps = self.patch_size
            if h < ps or w < ps:
                pad_h = max(ps - h, 0)
                pad_w = max(ps - w, 0)
                clean = np.pad(clean, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                h, w = clean.shape[:2]
            y = random.randint(0, h - ps)
            x = random.randint(0, w - ps)
            clean_patch = clean[y:y+ps, x:x+ps].copy()

            # Augment
            if self.augment:
                noisy_patch, clean_patch = self._augment(clean_patch.copy(), clean_patch)

            # Synthesize fresh Gaussian noise with random sigma
            sigma = random.uniform(self.sigma_min, self.sigma_max)
            clean_f = clean_patch.astype(np.float32) / 255.0
            noise = np.random.randn(*clean_f.shape).astype(np.float32) * (sigma / 255.0)
            noisy_f = np.clip(clean_f + noise, 0.0, 1.0)

            noisy_t = torch.from_numpy(noisy_f).permute(2, 0, 1)
            clean_t = torch.from_numpy(clean_f).permute(2, 0, 1)
        else:
            # --- Original: load fixed paired images from disk ---
            noisy = self._load_img(noisy_path)
            noisy, clean = self._random_crop(noisy, clean)
            if self.augment:
                noisy, clean = self._augment(noisy, clean)
            noisy_t = torch.from_numpy(noisy.astype(np.float32) / 255.0).permute(2, 0, 1)
            clean_t = torch.from_numpy(clean.astype(np.float32) / 255.0).permute(2, 0, 1)

        return noisy_t, clean_t


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smoother than L1, better for image restoration)."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class PSNRLoss(nn.Module):
    """PSNR-oriented loss (directly optimizes for PSNR metric)."""
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        return mse  # minimizing MSE = maximizing PSNR


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss (1 - SSIM). Pure PyTorch, no external deps.

    Computes SSIM over sliding windows using depthwise convolution with
    a Gaussian kernel. Supports multi-channel images (e.g. RGB).
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = 0.01 ** 2  # stabilizer for luminance
        self.C2 = 0.03 ** 2  # stabilizer for contrast
        # Pre-compute 2D Gaussian kernel
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        kernel_1d = g / g.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        # Shape: (1, 1, window_size, window_size)
        self.register_buffer("kernel", kernel_2d.unsqueeze(0).unsqueeze(0))

    def _ssim_map(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = pred.shape[1]
        # Expand kernel to (channels, 1, ws, ws) for depthwise conv
        # Move kernel to same device as input (fixes CUDA/CPU mismatch)
        kernel = self.kernel.to(device=pred.device, dtype=pred.dtype).expand(channels, -1, -1, -1)
        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, kernel, padding=pad, groups=channels)
        mu_target = F.conv2d(target, kernel, padding=pad, groups=channels)

        mu_pred_sq = mu_pred * mu_pred
        mu_target_sq = mu_target * mu_target
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_cross

        # Clamp variances to avoid negative values from numerical errors
        sigma_pred_sq = torch.clamp(sigma_pred_sq, min=0)
        sigma_target_sq = torch.clamp(sigma_target_sq, min=0)

        ssim_num = (2 * mu_cross + self.C1) * (2 * sigma_cross + self.C2)
        ssim_den = (mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2)

        return ssim_num / ssim_den

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_map = self._ssim_map(pred, target)
        return 1.0 - ssim_map.mean()


class HybridLoss(nn.Module):
    """Combined Charbonnier + SSIM loss for jointly optimizing PSNR and SSIM.

    total_loss = (1 - alpha) * Charbonnier + alpha * (1 - SSIM)

    Default alpha=0.84 follows the MS-SSIM paper recommendation.
    """
    def __init__(self, alpha: float = 0.84, eps: float = 1e-3):
        super().__init__()
        self.alpha = alpha
        self.charbonnier = CharbonnierLoss(eps=eps)
        self.ssim_loss = SSIMLoss(window_size=11, sigma=1.5)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate loss in FP32 to prevent float16 overflow in SSIM variances
        with torch.cuda.amp.autocast(enabled=False):
            pred = pred.float()
            target = target.float()
            l_char = self.charbonnier(pred, target)
            l_ssim = self.ssim_loss(pred, target)
            return (1 - self.alpha) * l_char + self.alpha * l_ssim


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calc_psnr_torch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate PSNR on tensors [0, 1]."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model weights.
    Smooths out training fluctuations and generally provides a 0.1~0.2 dB PSNR boost.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(param.device)

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Restormer")
    # Data
    p.add_argument("--train_noisy", type=str, default="data/train/noisy")
    p.add_argument("--train_clean", type=str, default="data/train/clean")
    # Model
    p.add_argument("--pretrained", type=str,
                   default="pretrained_models/gaussian_color_denoising_sigma50.pth")
    # Training
    p.add_argument("--patch_size", type=int, default=128,
                   help="Training patch size (128 for 4GB VRAM)")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size per step (1 for 4GB VRAM)")
    p.add_argument("--grad_accum", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--lr", type=float, default=2e-5,
                   help="Learning rate (small for fine-tuning)")
    p.add_argument("--total_iters", type=int, default=50000)
    p.add_argument("--warmup_iters", type=int, default=1000)
    p.add_argument("--loss", type=str, default="hybrid",
                   choices=["l1", "charbonnier", "mse", "hybrid"])
    p.add_argument("--ssim_weight", type=float, default=0.84,
                   help="Weight of SSIM in hybrid loss (0=pure Charbonnier, 1=pure SSIM)")
    # Mixed precision
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Use mixed precision training (saves ~50%% VRAM)")
    p.add_argument("--no_fp16", action="store_true", default=False)
    # Save & eval
    p.add_argument("--output_dir", type=str, default="experiments/finetune_v1")
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--eval_every", type=int, default=2500)
    p.add_argument("--eval_samples", type=int, default=10,
                   help="Number of training samples to evaluate PSNR on")
    # Workers
    p.add_argument("--num_workers", type=int, default=2)
    # Noise augmentation (Direction 3 - Domain Gap fix)
    p.add_argument("--synthetic_noise", action="store_true", default=False,
                   help="Re-synthesize noise from clean images instead of loading from disk")
    p.add_argument("--sigma_min", type=float, default=40.0,
                   help="Min noise sigma for synthetic noise (default: 40)")
    p.add_argument("--sigma_max", type=float, default=60.0,
                   help="Max noise sigma for synthetic noise (default: 60)")
    return p.parse_args()


def build_scheduler(optimizer, total_iters: int, warmup_iters: int):
    """Cosine annealing with linear warmup."""
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return current_iter / max(1, warmup_iters)
        progress = (current_iter - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, dataset, device, num_samples=10):
    """Quick evaluation on a few training samples (full image PSNR)."""
    model.eval()
    total_psnr = 0.0
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        noisy_path, clean_path = dataset.pairs[idx]
        noisy = cv2.imread(str(noisy_path), cv2.IMREAD_UNCHANGED)
        clean = cv2.imread(str(clean_path), cv2.IMREAD_UNCHANGED)
        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)

        # Crop to manageable size for evaluation (avoid OOM)
        h, w = noisy.shape[:2]
        max_eval_size = 512
        if h > max_eval_size or w > max_eval_size:
            cy = (h - max_eval_size) // 2 if h > max_eval_size else 0
            cx = (w - max_eval_size) // 2 if w > max_eval_size else 0
            eh = min(max_eval_size, h)
            ew = min(max_eval_size, w)
            noisy = noisy[cy:cy+eh, cx:cx+ew]
            clean = clean[cy:cy+eh, cx:cx+ew]

        noisy_t = torch.from_numpy(noisy.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        clean_t = torch.from_numpy(clean.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        # Pad to factor of 8
        _, _, h, w = noisy_t.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            noisy_t = F.pad(noisy_t, (0, pad_w, 0, pad_h), mode="reflect")

        # Evaluation is safer in FP32 to avoid attention overflow on large images
        with torch.no_grad():
            pred = model(noisy_t)

        if pad_h > 0 or pad_w > 0:
            pred = pred[:, :, :h, :w]

        pred = torch.clamp(pred, 0, 1)
        psnr = calc_psnr_torch(pred, clean_t)
        total_psnr += psnr

    model.train()
    return total_psnr / len(indices)


def main():
    args = parse_args()
    use_fp16 = args.fp16 and not args.no_fp16

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ----- Model -----
    model = Restormer(layer_norm_type="BiasFree")
    n_params = count_parameters(model)
    print(f"Model params: {n_params:,} ({n_params / 1e6:.2f}M)")

    # Load pretrained weights
    ckpt = torch.load(args.pretrained, map_location="cpu")
    state_dict = ckpt.get("params", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print(f"[OK] Loaded pretrained: {args.pretrained}")

    model.to(device)
    model.train()

    # Enable gradient checkpointing for memory saving
    if hasattr(torch.utils, 'checkpoint'):
        print("[OK] Gradient checkpointing available")

    # ----- Dataset -----
    dataset = DenoisingDataset(
        args.train_noisy, args.train_clean,
        patch_size=args.patch_size, augment=True,
        use_synthetic_noise=args.synthetic_noise,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # ----- Loss -----
    if args.loss == "l1":
        criterion = nn.L1Loss()
    elif args.loss == "charbonnier":
        criterion = CharbonnierLoss()
    elif args.loss == "mse":
        criterion = PSNRLoss()
    elif args.loss == "hybrid":
        criterion = HybridLoss(alpha=args.ssim_weight)
        print(f"  Hybrid loss: Charbonnier weight={1-args.ssim_weight:.2f}, SSIM weight={args.ssim_weight:.2f}")
    print(f"Loss: {args.loss}")

    # ----- Optimizer & Scheduler -----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, args.total_iters, args.warmup_iters)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    ema = EMA(model, decay=0.999)

    print(f"LR: {args.lr}, Effective batch: {args.batch_size * args.grad_accum}")
    print(f"Patch: {args.patch_size}x{args.patch_size}, FP16: {use_fp16}")
    print(f"Total iters: {args.total_iters}, Save every: {args.save_every}")
    print(f"{'='*60}")

    # ----- Training Loop -----
    current_iter = 0
    best_psnr = 0.0
    log_rows = []
    running_loss = 0.0
    start_time = time.time()

    data_iter = iter(loader)
    optimizer.zero_grad()

    while current_iter < args.total_iters:
        # Get batch (loop dataset)
        try:
            noisy, clean = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            noisy, clean = next(data_iter)

        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        # Forward with mixed precision
        with torch.cuda.amp.autocast(enabled=use_fp16):
            pred = model(noisy)
            loss = criterion(pred, clean) / args.grad_accum

        # Backward
        scaler.scale(loss).backward()
        running_loss += loss.item() * args.grad_accum

        # Optimizer step (with gradient accumulation)
        if (current_iter + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update()  # Update EMA weights after optimizer step
            scheduler.step()

        current_iter += 1

        # ----- Logging -----
        if current_iter % 100 == 0:
            avg_loss = running_loss / 100
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]
            its_per_sec = current_iter / elapsed
            eta_sec = (args.total_iters - current_iter) / max(its_per_sec, 1e-6)
            eta_min = eta_sec / 60

            print(f"[{current_iter:>6}/{args.total_iters}] "
                  f"loss={avg_loss:.6f} lr={lr:.2e} "
                  f"speed={its_per_sec:.1f}it/s ETA={eta_min:.0f}min")
            running_loss = 0.0

        # ----- Evaluation -----
        if current_iter % args.eval_every == 0:
            torch.cuda.empty_cache()
            ema.apply_shadow()
            eval_psnr = evaluate(model, dataset, device, args.eval_samples)
            ema.restore()
            print(f"  >> Eval PSNR: {eval_psnr:.4f} dB (best: {best_psnr:.4f} dB)")

            log_rows.append({
                "iter": current_iter,
                "loss": f"{avg_loss:.6f}",
                "psnr": f"{eval_psnr:.4f}",
                "lr": f"{lr:.2e}",
            })

            if eval_psnr > best_psnr:
                best_psnr = eval_psnr
                save_path = output_dir / "best_model.pth"
                ema.apply_shadow()
                torch.save({"params": model.state_dict()}, save_path)
                ema.restore()
                print(f"  >> NEW BEST! Saved to {save_path}")

            model.train()

        # ----- Save checkpoint -----
        if current_iter % args.save_every == 0:
            save_path = output_dir / f"model_iter{current_iter}.pth"
            ema.apply_shadow()
            torch.save({
                "params": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "iter": current_iter,
                "best_psnr": best_psnr,
            }, save_path)
            ema.restore()
            print(f"  >> Checkpoint saved: {save_path}")

    # ----- Final save -----
    final_path = output_dir / "final_model.pth"
    torch.save({"params": model.state_dict()}, final_path)
    print(f"\n{'='*60}")
    print(f"Training complete! Best PSNR: {best_psnr:.4f} dB")
    print(f"Final model: {final_path}")
    print(f"Best model:  {output_dir / 'best_model.pth'}")

    # Save log
    log_path = output_dir / "training_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "loss", "psnr", "lr"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
