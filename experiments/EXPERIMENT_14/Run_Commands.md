# Execution Instructions for EXPERIMENT_14

## Option 1: GPU Cluster (SLURM Server Execution) — Recommended

Run directly from the root repository on the GPU server (`gpu-a240`):

```bash
cd /data/khoalq/surgical_ai
sbatch experiments/EXPERIMENT_14/scripts/run_server.sbatch
```

### Monitoring the Job:
```bash
# Check queue status
squeue -u khoalq

# Tail output logs
tail -f /data/khoalq/logs/exp14_stratified_m2f_*.out

# Tail training metrics
tail -f /data/khoalq/logs/exp14_stratified_m2f_*.log
```

### Output Deliverables:
At completion, `/data/khoalq/checkpoints/exp14_stratified_m2f/` contains:
* `best_model.pth`
* `training_log.csv`
* `val_predictions.csv` (120 frames: Patients 38, 32, 18)
* `test_predictions.csv` (109 frames: Patients 41, 31, 21, 51)
* `metrics_summary.json` (global & per-patient breakdown)
* `diagnostics_val/` & `diagnostics_test/` (2x2 montages)
* **`results.zip`** (packaged archive matching EXP_5 standard)

---

## Option 2: Kaggle CUDA GPU (Alternative Execution)

1. Open [Kaggle](https://www.kaggle.com/) and create a new Python Notebook.
2. Under **Accelerator**, select **GPU T4 x2** or **GPU P100**.
3. Under **Input Data**, attach your L3D dataset (`laparoscopic-liver-landmark-dataset` or `l3d-dataset`).
4. Upload and open the self-contained notebook:
   `experiments/EXPERIMENT_14/notebooks/EXP14_Stratified_JunctionSteered_Mask2Former_Kaggle.ipynb`
5. Click **Run All**.
6. At completion, download `/kaggle/working/results_exp14/results.zip`.

---

## Option 3: Local CLI Verification (macOS / Local GPU)

### 1. Fast Smoke Test (CPU / Local)
```bash
HF_HUB_OFFLINE=1 python3 experiments/EXPERIMENT_14/scripts/train.py \
  --smoke_test \
  --device cpu
```

### 2. Standalone Evaluation on Checkpoint
```bash
python3 experiments/EXPERIMENT_14/scripts/evaluate.py \
  --checkpoint /path/to/best_model.pth \
  --out_dir experiments/EXPERIMENT_14/results \
  --device cuda
```
