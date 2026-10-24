# Execution Commands: EXP_13 — Mask2Former + TopoNet Loss + BCRNet

Commands for running training, evaluation, and visual reporting locally or on Kaggle GPU.

---

## 1. Kaggle CUDA Environment (Recommended)

### Step 1: Sync Codebase
```bash
%cd /kaggle/working/surgical_ai
!git pull origin main
```

### Step 2: Launch Training
```bash
%cd /kaggle/working/surgical_ai

!python3 experiments/EXP_13_mask2former_bcrnet_toponet/scripts/train_exp13.py \
    --dataset_dir /kaggle/working/L3D \
    --save_dir /kaggle/working/checkpoints/EXP_13 \
    --epochs 60 \
    --batch_size 4 \
    --lr 6e-5
```

### Step 3: Run Full Benchmark Evaluation & Visualizations
```bash
%cd /kaggle/working/surgical_ai

!python3 experiments/EXP_13_mask2former_bcrnet_toponet/scripts/evaluate_exp13.py \
    --dataset_dir /kaggle/working/L3D \
    --checkpoint /kaggle/working/checkpoints/EXP_13/best_model.pth \
    --output_dir /kaggle/working/eval_plots_exp13 \
    --max_plots 25
```

---

## 2. Local macOS / CPU / MPS Environment

### Fast Sanity Run:
```bash
python3 experiments/EXP_13_mask2former_bcrnet_toponet/scripts/train_exp13.py \
    --epochs 2 \
    --batch_size 2 \
    --no_amp
```
