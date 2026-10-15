# EXPERIMENT_6: Execution Commands & Run Guide

This document provides exact, copy-pasteable execution commands for training and evaluating **EXPERIMENT_6 (Heatmap-Guided Junction-Steered Mask2Former)** across local macOS and CUDA server environments.

---

## 1. Local Smoke Test (Fast Pipeline Verification)

Verify dataset loading, continuous Gaussian heatmap synthesis, model forward pass, dynamic dot-product heatmap generation, `delta_q` query steering, and Gaussian focal loss backward autograd:

```bash
/opt/anaconda3/envs/surgical_ai/bin/python experiments/EXPERIMENT_6/scripts/test_local_forward.py
```

Or run 1 step of full data training:
```bash
/opt/anaconda3/envs/surgical_ai/bin/python experiments/EXPERIMENT_6/scripts/train.py \
    --smoke_test \
    --device cpu \
    --batch_size 1 \
    --out_dir experiments/EXPERIMENT_6/results/smoke_test
```

---

## 2. Server Training (NVIDIA A100 / RTX 3090 / Kaggle CUDA)

Execute full 60-epoch multi-task training with AMP bfloat16:

```bash
# Standard 60-Epoch Training (Effective Batch Size: 4)
python3 experiments/EXPERIMENT_6/scripts/train.py \
    --epochs 60 \
    --batch_size 2 \
    --accum_steps 2 \
    --lr_head 1e-4 \
    --lr_backbone 1e-5 \
    --lambda_heatmap 1.0 \
    --device cuda \
    --out_dir /data/khoalq/checkpoints/exp6_heatmap_m2f
```

---

## 3. Standalone Evaluation (Validation & Test Sets)

Run comprehensive benchmark evaluation using the trained checkpoint:

```bash
python3 experiments/EXPERIMENT_6/scripts/evaluate.py \
    --checkpoint /data/khoalq/checkpoints/exp6_heatmap_m2f/best_model.pth \
    --device cuda \
    --out_dir /data/khoalq/checkpoints/exp6_heatmap_m2f/eval
```

Outputs produced in `--out_dir`:
1. `metrics_summary.json`: High-level summary of Macro Dice, IoU, ASSD, class Dices, Patient 40 metrics, and heatmap-decoded Junction errors.
2. `val_predictions.csv`: 122 validation frames with per-frame metrics.
3. `test_predictions.csv`: 109 test frames with per-frame metrics.
4. `patient_40_diagnostics/`: 101 four-panel diagnostic montages for Patient 40 validation frames.

---

## 4. Slurm Cluster Submission (`gpu-a240`)

The cluster submission script is located at [`experiments/EXPERIMENT_6/scripts/run_server.sbatch`](file:///Users/khoale/Downloads/Surgical%20AI/experiments/EXPERIMENT_6/scripts/run_server.sbatch).

### One-line submission command on server:
```bash
cd /data/khoalq/surgical_ai && git pull && sbatch experiments/EXPERIMENT_6/scripts/run_server.sbatch
```

### Live log monitoring command:
```bash
tail -f /data/khoalq/logs/exp6_heatmap_m2f_*.log
```
