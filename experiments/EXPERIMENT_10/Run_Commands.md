# EXPERIMENT_10: Execution & Run Commands

This document contains exact, verified CLI commands for training, evaluating, and packaging **EXPERIMENT_10 (Depth-Geometric Junction-Steered Mask2Former)** across local macOS and CUDA server environments.

---

## 1. Local Environment (macOS MPS / CPU Verification)

Use this to verify data loading, tensor dimensions, forward passes, and loss computation without running a full training loop:

```bash
# Navigate to workspace root
cd "/Users/khoale/Downloads/Surgical AI"

# 1. Smoke test forward pass & tensor shapes (1 step dry run)
python3 -m experiments.EXPERIMENT_10.scripts.train --smoke-test --device cpu

# 2. Verify metric computation on 2 validation frames
python3 -m experiments.EXPERIMENT_10.scripts.evaluate --data-dir data/L3D --device cpu --max-samples 2
```

---

## 2. Server Cluster Execution (`gpu-a240` / SLURM)

For training on NVIDIA A100 MIG 2g.10gb / 40GB GPU:

```bash
# 1. Submit SLURM training job
sbatch experiments/EXPERIMENT_10/scripts/run_server.sbatch

# 2. Monitor job status
squeue -u $USER

# 3. Stream real-time training log
tail -f experiments/EXPERIMENT_10/results/train_exp10.log

# 4. Run standalone evaluation on best checkpoint
python3 -m experiments.EXPERIMENT_10.scripts.evaluate \
    --checkpoint checkpoints/exp10_best_model.pth \
    --data-dir /data/khoalq/data/L3D \
    --device cuda
```

---

## 3. Kaggle Environment Execution

When running inside a Kaggle notebook with GPU (T4 / P100):

```python
# Cell 1: Environment & Repository Paths
import sys, os
WORKSPACE_DIR = "/kaggle/working/surgical_ai"
sys.path.insert(0, WORKSPACE_DIR)

# Cell 2: Launch Full Training
!python3 -m experiments.EXPERIMENT_10.scripts.train \
    --data-dir /kaggle/input/laparoscopic-liver-landmark-dataset/L3D \
    --epochs 60 \
    --batch-size 2 \
    --accum-steps 2 \
    --lr-backbone 1e-5 \
    --lr-head 1e-4 \
    --device cuda

# Cell 3: Evaluate Best Model
!python3 -m experiments.EXPERIMENT_10.scripts.evaluate \
    --checkpoint /kaggle/working/surgical_ai/checkpoints/exp10_best_model.pth \
    --data-dir /kaggle/input/laparoscopic-liver-landmark-dataset/L3D \
    --device cuda
```

---

## 4. Packaging Deliverables

After evaluation completes, package all outputs for documentation parity:

```bash
cd experiments/EXPERIMENT_10
zip -r results/results.zip results/metrics_summary.json results/val_predictions.csv results/test_predictions.csv
```
