# EXP_10 Execution Commands — Super-Token Geometric ViT

This guide provides exact, copy-pasteable terminal commands to run **EXP_10: Super-Token Geometric ViT** on local macOS and Kaggle CUDA environments.

---

## 1. Local Smoke Test (Offline CPU / macOS)

Verify model architecture, tensor shapes, dual-domain loss, and gradient backpropagation:
```bash
python experiments/EXP_10_super_token_vit/scripts/smoke_test_exp10.py
```

---

## 2. Training on Kaggle GPU (CUDA: T4 / P100 / A100)

### Setup & Directory Navigation
```bash
cd /kaggle/working/surgical_ai
export PYTHONPATH="/kaggle/working/surgical_ai:/kaggle/working/surgical_ai/experiments/EXP_10_super_token_vit:$PYTHONPATH"
```

### Full Training Run (ViT-Base, K=6 Control Points, Dual-Domain Loss)
```bash
python experiments/EXP_10_super_token_vit/scripts/train_super_token_vit.py \
    --dataset_dir /kaggle/working/L3D \
    --backbone vit_base_patch16_224 \
    --num_ctrl_points 6 \
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
python experiments/EXP_10_super_token_vit/scripts/train_super_token_vit.py \
    --dataset_dir /kaggle/working/L3D \
    --backbone vit_tiny_patch16_224 \
    --num_ctrl_points 6 \
    --epochs 20 \
    --batch_size 32 \
    --lr 2e-4 \
    --amp \
    --use_depth \
    --save_dir checkpoints/EXP_10_tiny
```

---

## 3. Official Benchmark Evaluation (Validation Set)

Evaluate official Pixel Dice score, IoU, and Control Point Error against `masks_gt/`:
```bash
python experiments/EXP_10_super_token_vit/scripts/evaluate_super_token.py \
    --checkpoint checkpoints/EXP_10/best_model.pth \
    --dataset_dir /kaggle/working/L3D \
    --thresh 0.35 \
    --stroke_px 2
```

---

## 4. Generate 4-Panel Diagnostic Visualizations (All 122 Frames)

Produces:
1. RGB with continuous $K=6$ predicted splines
2. Ground truth mask (Cyan) vs Predicted raster (Red) with per-frame Dice
3. $32\times 32$ Super-Token cross-attention heatmaps
4. Depth map with 3D anatomical trajectory
```bash
python experiments/EXP_10_super_token_vit/scripts/visualize_val_predictions.py \
    --checkpoint checkpoints/EXP_10/best_model.pth \
    --dataset_dir /kaggle/working/L3D \
    --output_dir outputs/EXP_10/val_visualizations \
    --max_samples 122
```
