# PRCV2026 垃圾图像高斯噪声去噪挑战赛

## 队伍信息
- **队伍编号**: PRCV2026-0025
- **队伍名称**: TinyLight

## 目录结构

```
competition/
├── data/                       # 📂 数据集（不要修改）
│   ├── train/
│   │   ├── noisy/              # 训练集 noisy 图片 (2387张)
│   │   └── clean/              # 训练集 clean 图片 (2387张)
│   ├── val/
│   │   └── noisy/              # 验证集 noisy 图片 (100张)
│   ├── train_list.csv          # 训练集列表
│   └── val_list.csv            # 验证集列表
│
├── src/                        # 📂 源代码
│   ├── model_arch.py           # Restormer 模型架构定义
│   ├── inference.py            # 推理脚本（单模型单次前向传播）
│   ├── evaluate.py             # 评估脚本（PSNR/SSIM）
│   ├── prepare_submit.py       # 提交文件打包脚本
│   └── train.py                # 训练脚本（后续创建）
│
├── configs/                    # 📂 训练/推理配置文件
│
├── pretrained_models/          # 📂 预训练权重文件
│
├── results/                    # 📂 推理输出结果
│   └── val/                    # 验证集去噪结果
│
├── submit/                     # 📂 提交文件
│   └── Results_TinyLight.zip   # 打包好的提交文件
│
├── validate_results.py         # 官方验证脚本（保留原位）
├── requirements.txt            # Python 依赖
├── info.txt                    # 竞赛信息
└── README.md                   # 本文件
```

## 快速开始

```bash
# 1. 激活环境
conda activate prcv2026

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载预训练权重到 pretrained_models/ 目录

# 4. 推理
python src/inference.py \
    --weights pretrained_models/restormer_gaussian_sigma50.pth \
    --input_dir data/val/noisy \
    --result_dir results/val \
    --tile_size 256 --tile_overlap 32

# 5. 打包提交
python src/prepare_submit.py --result_dir results/val

# 6. 评估（需要 clean ground truth）
python src/evaluate.py \
    --result_dir results/val \
    --clean_dir data/train/clean

# 7. 本地 Fine-tune 训练 (针对 4GB 显存优化)
python src/train.py --pretrained pretrained_models/gaussian_color_denoising_sigma50.pth --train_noisy data/train/noisy --train_clean data/train/clean --output_dir experiments/finetune_v1 --patch_size 128 --batch_size 1 --grad_accum 4 --lr 2e-5 --total_iters 50000 --eval_every 5000 --save_every 10000 --eval_samples 20 --num_workers 2
```

## 竞赛规则要点
- ❌ 禁止自集成 / TTA（必须单模型单次前向传播）
- ⚠️ 模型参数量 ≤ 50M（超限总分 ×85%）
- ✅ 允许使用公开预训练模型和公开数据集
- ✅ 训练阶段允许数据增强
