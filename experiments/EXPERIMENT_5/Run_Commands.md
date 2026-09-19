# EXPERIMENT_5: Execution Commands & Run Guide

This document provides exact, copy-pasteable execution commands for training and evaluating **EXPERIMENT_5 (Junction-Steered Mask2Former)** across local macOS and CUDA environments.

---

## 1. Local Smoke Test (Fast Pipeline Verification)

Verify dataset loading, deterministic junction extraction, forward pass, query steering, and loss backward pass in 1 iteration:

```bash
python3 experiments/EXPERIMENT_5/scripts/train.py \
    --smoke_test \
    --device cpu \
    --batch_size 1 \
    --out_dir experiments/EXPERIMENT_5/results/smoke_test
```

---

## 2. Server Training (NVIDIA A100 / RTX 3090 / Kaggle CUDA)

Execute full 60-epoch multi-task training:

```bash
# Standard 60-Epoch Training (Effective Batch Size: 4)
python3 experiments/EXPERIMENT_5/scripts/train.py \
    --epochs 60 \
    --batch_size 2 \
    --accum_steps 2 \
    --lr_head 1e-4 \
    --lr_backbone 1e-5 \
    --lambda_coord 5.0 \
    --lambda_vis 1.0 \
    --device cuda \
    --out_dir experiments/EXPERIMENT_5/results/run_01
```

---

## 3. Standalone Evaluation (Validation & Test Sets)

Run comprehensive benchmark evaluation using the trained checkpoint:

```bash
python3 experiments/EXPERIMENT_5/scripts/evaluate.py \
    --checkpoint experiments/EXPERIMENT_5/results/run_01/best_model.pth \
    --device cuda \
    --out_dir experiments/EXPERIMENT_5/results/run_01_eval
```

Outputs produced in `--out_dir`:
1. `metrics_summary.json`: High-level summary of Macro Dice, IoU, ASSD, class Dices, Patient 40 metrics, and Junction errors.
2. `val_predictions.csv`: 122 validation frames with per-frame metrics.
3. `test_predictions.csv`: 109 test frames with per-frame metrics.
4. `patient_40_diagnostics/`: 101 four-panel diagnostic montages for Patient 40 validation frames.

---

## 4. Slurm Cluster Submission (`gpu-a240`)

The standard cluster submission script is located at [`experiments/EXPERIMENT_5/scripts/run_server.sbatch`](file:///Users/khoale/Downloads/Surgical%20AI/experiments/EXPERIMENT_5/scripts/run_server.sbatch).

### One-line submission command on server:
```bash
cd /data/khoalq/surgical_ai && git pull && sbatch experiments/EXPERIMENT_5/scripts/run_server.sbatch
```

### Live log monitoring command:
```bash
tail -f /data/khoalq/logs/exp5_junction_m2f_*.log
```
