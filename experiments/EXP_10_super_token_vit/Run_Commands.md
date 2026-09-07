# EXP_10 Execution Commands — Macro-Patch Geometric ViT (Way A)

This guide provides exact, copy-pasteable terminal commands to run **EXP_10: Macro-Patch Geometric ViT (Way A: 64px Grid)** on local macOS and Kaggle CUDA environments.

---

## 1. Local Smoke Test (Offline CPU / macOS)

Verify macro-patch architecture, 4x4 spatial merge, inter-macro relational transformer, loss, and coordinate shifting:
```bash
python experiments/EXP_10_super_token_vit/scripts/smoke_test_macro_vit.py
```

---

## 2. Training on Kaggle GPU (CUDA: T4 / P100 / A100)

### Setup & Directory Navigation
```bash
cd /kaggle/working/surgical_ai
git pull origin main

export PYTHONPATH="/kaggle/working/surgical_ai:/kaggle/working/surgical_ai/experiments/EXP_10_super_token_vit:$PYTHONPATH"
```

### Full Training Run (ViT-Base, 64px Macro-Patches, 8x8 Grid, 80 Epochs)
```bash
python experiments/EXP_10_super_token_vit/scripts/train_macro_vit.py \
    --dataset_dir /kaggle/working/L3D \
    --backbone vit_base_patch16_224 \
    --macro_patch_size 64 \
    --epochs 80 \
    --batch_size 16 \
    --lr 1e-4 \
    --backbone_lr_mult 0.1 \
    --amp \
    --use_depth \
    --save_dir checkpoints/EXP_10 \
    --wandb
```

### Fast Prototyping Run (20 Epochs, ViT-Tiny)
```bash
python experiments/EXP_10_super_token_vit/scripts/train_macro_vit.py \
    --dataset_dir /kaggle/working/L3D \
    --backbone vit_tiny_patch16_224 \
    --macro_patch_size 64 \
    --epochs 20 \
    --batch_size 32 \
    --lr 2e-4 \
    --amp \
    --use_depth \
    --save_dir checkpoints/EXP_10_tiny
```

---

## 3. Official Benchmark Evaluation (Validation Set)

Evaluate official Pixel Dice score, IoU, and Control Point Error on the 122 validation frames:
```bash
python experiments/EXP_10_super_token_vit/scripts/evaluate_macro_vit.py \
    --checkpoint checkpoints/EXP_10/best_model.pth \
    --dataset_dir /kaggle/working/L3D \
    --thresh 0.30 \
    --stroke_px 2
```

---

## 4. Generate 4-Panel Diagnostic Visualizations (All 122 Frames)

Produces:
1. RGB with predicted continuous 64px macro-splines
2. Ground truth mask (Cyan) vs Predicted raster (Red) with per-frame Dice
3. $8\times 8$ Macro-Grid Activation Map showing active $64\times 64$ blocks
4. Depth map with 3D anatomical trajectory
```bash
python experiments/EXP_10_super_token_vit/scripts/visualize_macro_vit.py \
    --checkpoint checkpoints/EXP_10/best_model.pth \
    --dataset_dir /kaggle/working/L3D \
    --output_dir outputs/EXP_10/val_visualizations \
    --max_samples 122
```
