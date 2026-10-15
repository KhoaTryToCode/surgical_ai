# EXPERIMENT_7: Execution Commands & Run Guide

This document provides exact, copy-pasteable execution commands for training, evaluating, and downloading results for **EXPERIMENT_7 (Manifold-Steered Mask2Former)** across local environments, Slurm server clusters (`gpu-a240`), and Kaggle GPU notebooks.

---

## 1. Local Smoke Test (Fast Pipeline Verification)

Verify dataset loading, 11-landmark canonical atlas extraction, continuous $(u, v)$ manifold generation, forward pass, visibility-gated cross-attention steering, and autograd backward pass in 1 iteration:

```bash
python3 experiments/EXPERIMENT_7/scripts/train.py \
    --smoke_test \
    --device cpu \
    --batch_size 1 \
    --out_dir experiments/EXPERIMENT_7/results/smoke_test
```

---

## 2. Remote Server Submission via Slurm (`gpu-a240`)

The server script is preconfigured for an NVIDIA A100 MIG 3g.20gb slice at [`experiments/EXPERIMENT_7/scripts/run_server.sbatch`](file:///Users/khoale/Downloads/Surgical%20AI/experiments/EXPERIMENT_7/scripts/run_server.sbatch).

### One-line submission command on server:
```bash
cd /data/khoalq/surgical_ai && git pull && sbatch experiments/EXPERIMENT_7/scripts/run_server.sbatch
```

### Live log monitoring command:
```bash
tail -f /data/khoalq/logs/exp7_manifold_m2f_*.log
```

### Check Slurm queue status:
```bash
squeue -u khoalq
```

---

## 3. Direct Server Training & Standalone Evaluation (CUDA)

If running in an interactive CUDA shell or tmux session on the server or workstation:

### Standard 60-Epoch Training (Effective Batch Size: 4)
```bash
python3 -u experiments/EXPERIMENT_7/scripts/train.py \
    --data_dir /data/khoalq/data/L3D \
    --out_dir /data/khoalq/checkpoints/exp7_manifold_m2f \
    --epochs 60 \
    --batch_size 2 \
    --accum_steps 2 \
    --lr_head 1e-4 \
    --lr_backbone 1e-5 \
    --lambda_coord 5.0 \
    --lambda_vis 1.0 \
    --lambda_uv 1.0 \
    --num_workers 8 \
    --eval_splits both \
    --device cuda
```

### Standalone Benchmark Evaluation
Run comprehensive validation (122 frames) and test (109 frames) evaluation using the best checkpoint:

```bash
python3 -u experiments/EXPERIMENT_7/scripts/evaluate.py \
    --checkpoint /data/khoalq/checkpoints/exp7_manifold_m2f/best_model.pth \
    --data_dir /data/khoalq/data/L3D \
    --out_dir /data/khoalq/checkpoints/exp7_manifold_m2f/eval_results \
    --eval_splits both \
    --device cuda
```

Outputs produced in the evaluation directory:
1. `metrics_summary.json`: Macro Dice, Mean IoU, ASSD, class Dices (Ridge, Silhouette, Falciform), Patient 40 diagnostics, and 11-query atlas pixel errors.
2. `val_predictions.csv`: 122 validation frames with per-frame metrics.
3. `test_predictions.csv`: 109 test frames with per-frame metrics.
4. `patient_40_diagnostics/`: 101 four-panel diagnostic montages for Patient 40 validation frames.
5. `results.zip`: Compressed archive containing all predictions, logs, summaries, and diagnostic plots.

---

## 4. Kaggle 1-Click Execution

For training and evaluating on Kaggle (NVIDIA P100 / T4 x2):

1. Upload [`experiments/EXPERIMENT_7/notebooks/EXP7_ManifoldSteered_Mask2Former_Kaggle.ipynb`](file:///Users/khoale/Downloads/Surgical%20AI/experiments/EXPERIMENT_7/notebooks/EXP7_ManifoldSteered_Mask2Former_Kaggle.ipynb) as a new notebook on Kaggle.
2. Attach the L3D dataset (`khoale/laparoscopic-liver-3d` or path `/kaggle/input/laparoscopic-liver-3d`).
3. Set Accelerator to **GPU P100** or **GPU T4 x2**.
4. Click **Run All**.
5. At the end of execution, `results_exp7.zip` is automatically created and presented via an interactive HTML download link in cell output.

---

## 5. Syncing Results from Remote Server to Local Machine

Once the Slurm job completes, copy `results.zip` from the cluster to your local machine:

```bash
rsync -avzP khoalq@100.82.42.48:/data/khoalq/checkpoints/exp7_manifold_m2f/results.zip experiments/EXPERIMENT_7/results/results.zip
```

To extract and inspect the results locally:
```bash
unzip -o experiments/EXPERIMENT_7/results/results.zip -d experiments/EXPERIMENT_7/results/
```
