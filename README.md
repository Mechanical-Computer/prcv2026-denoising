# PRCV2026 Gaussian Image Denoising — TinyLight Solution

**Competition:** PRCV2026 垃圾图像高斯噪声去噪挑战赛  
**Final Rank:** 13th / ~35 teams  
**Final Score:** PSNR 32.984 | SSIM 0.8920 | Final 33.792

---

## Model

[**Restormer**](https://github.com/swz30/Restormer) (Restoration Transformer), self-implemented in a single standalone file — no need to install the full Restormer repo.

| Config | Value |
|:---|:---:|
| Parameters | ~26.1M |
| `dim` | 48 |
| `num_blocks` | [4, 6, 6, 8] |
| Attention | MDTA (Multi-Dconv Head Transposed Attention) |
| FFN | GDFN (Gated-Dconv Feed-Forward Network) |
| Layer Norm | BiasFree |

## Pretrained Weights

Download from the official Restormer release:

```
gaussian_color_denoising_sigma50.pth
```

Place in `pretrained_models/`:

```bash
mkdir pretrained_models
# Download from: https://github.com/swz30/Restormer/releases
```

---

## Environment

```bash
conda create -n denoising python=3.10
conda activate denoising
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python einops timm tqdm
```

---

## Data Structure

```
data/
├── train/
│   ├── noisy/      # *_noisy.png
│   └── clean/      # *_clean.png  (same filename with _noisy → _clean)
└── noisy/          # Test images (no labels)
```

---

## Training

### Single GPU (e.g. RTX 4090 24GB)

```bash
python src/train.py \
    --pretrained pretrained_models/gaussian_color_denoising_sigma50.pth \
    --output_dir experiments/finetune_v1 \
    --synthetic_noise --sigma_min 35 --sigma_max 65 \
    --loss hybrid --ssim_weight 0.1 \
    --patch_size 256 --batch_size 3 --grad_accum 5 \
    --lr 2e-5 --total_iters 60000
```

### Dual GPU (e.g. 2× RTX 4090 48GB, auto-detected)

```bash
python src/train.py \
    --pretrained pretrained_models/gaussian_color_denoising_sigma50.pth \
    --output_dir experiments/finetune_dual \
    --synthetic_noise --sigma_min 35 --sigma_max 65 \
    --loss hybrid --ssim_weight 0.1 \
    --patch_size 384 --batch_size 6 --grad_accum 4 \
    --lr 1e-5 --total_iters 60000
```

### Key Training Arguments

| Argument | Default | Description |
|:---|:---:|:---|
| `--pretrained` | — | Path to pretrained `.pth` checkpoint |
| `--loss` | `hybrid` | `l1` / `charbonnier` / `hybrid` |
| `--ssim_weight` | 0.84 | Weight of SSIM in hybrid loss (0=pure Charbonnier) |
| `--patch_size` | 128 | Training crop size |
| `--batch_size` | 1 | Per-step batch size |
| `--grad_accum` | 4 | Gradient accumulation steps |
| `--synthetic_noise` | False | Re-synthesize noise from clean images |
| `--sigma_min/max` | 40/60 | Noise sigma range for synthetic noise |
| `--fp16` | True | Mixed precision training |

---

## Inference

```bash
python src/inference.py \
    --weights experiments/finetune_dual/best_model.pth \
    --input_dir data/noisy \
    --result_dir results/output \
    --tile_size 384 --tile_overlap 256
```

**Tile inference** is used to handle high-resolution images within VRAM limits.  
Each tile uses soft cosine blending at borders to eliminate seam artifacts.

| `tile_size` | `tile_overlap` | Notes |
|:---:|:---:|:---|
| 384 | 128 | Fast, matches training patch |
| 384 | 256 | Higher quality, 2× slower |
| 256 | 128 | For lower VRAM (< 8GB) |

---

## Code Structure

```
├── src/
│   ├── model_arch.py   # Restormer architecture (standalone)
│   ├── train.py        # Fine-tuning script
│   │                   #   - Hybrid Loss (Charbonnier + SSIM)
│   │                   #   - EMA (Exponential Moving Average)
│   │                   #   - Dynamic noise synthesis (sigma aug)
│   │                   #   - Auto multi-GPU (DataParallel)
│   │                   #   - Mixed precision (AMP)
│   └── inference.py    # Tile inference with soft blending
├── pretrained_models/
├── data/
├── experiments/
└── results/
```

---

## Key Techniques

| Technique | Effect |
|:---|:---|
| **Hybrid Loss** (Charbonnier + SSIM) | SSIM improved from 0.885 → 0.892 |
| **EMA** (decay=0.999) | Smooths weight oscillations, improves final model stability |
| **Dynamic σ augmentation** (σ∈[35,65]) | Reduces domain gap between train and test noise distributions |
| **Large patch training** (384×384) | Larger receptive field → better PSNR |
| **Tile inference with overlap** | Handles any resolution within VRAM limits |
| **FP16 inference** | ~2× faster inference, negligible quality loss |

---

## Results

| Experiment | PSNR (online) | SSIM | Notes |
|:---|:---:|:---:|:---|
| v1 baseline | 33.67 | 0.885 | Charbonnier only, patch=128 |
| v3 hybrid | 33.75 | 0.890 | SSIM weight=0.5 |
| 4090 dual + sigma aug | **32.98** | **0.892** | Final submission |

*(Online scores use a harder test set than local validation)*

---

## License

MIT
