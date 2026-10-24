# EXP_12: Execution Commands (macOS & Kaggle CUDA)

## 1. Local macOS Development / Sanity Run
```bash
# Run mathematical verification of the master architecture
python3 shared/math_lab/test_19_latent_space_analysis.py
python3 shared/math_lab/test_20_iterative_architecture_dice.py

# Quick local sanity check on CPU / MPS
python3 experiments/EXP_12_surgical_curve_former_v2/scripts/train_exp12.py \
    --epochs 2 \
    --batch_size 2 \
    --lr 5e-5 \
    --device cpu \
    --save_dir experiments/EXP_12_surgical_curve_former_v2/outputs/checkpoints
```

---

## 2. Kaggle CUDA Production Training (Copy-Paste Ready)
```bash
# Clone and prepare environment
cd /kaggle/working
git pull origin main

# Launch production training with AMP and W&B logging
python3 experiments/EXP_12_surgical_curve_former_v2/scripts/train_exp12.py \
    --dataset_dir /kaggle/working/L3D \
    --epochs 80 \
    --batch_size 4 \
    --lr 5e-5 \
    --backbone_lr_mult 0.1 \
    --weight_decay 1e-4 \
    --hold_epochs 15 \
    --lambda_d_min 0.05 \
    --acpi_top_k 10 \
    --amp \
    --use_depth \
    --save_dir /kaggle/working/checkpoints/EXP_12 \
    --wandb \
    --wandb_key 83f4544a22543e319c6009abceaac90b634c68a3
```


---

## 3. Evaluation & 4-Panel Clinical Visualizer
```bash
# Run validation evaluation and save 4-panel visual overlays
python3 experiments/EXP_12_surgical_curve_former_v2/scripts/evaluate_exp12.py \
    --dataset_dir /kaggle/working/L3D \
    --checkpoint /kaggle/working/checkpoints/EXP_12/best_model.pth \
    --output_dir /kaggle/working/eval_plots_exp12 \
    --max_plots 20
```
