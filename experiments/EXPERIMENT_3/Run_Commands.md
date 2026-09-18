# Run Commands — EXPERIMENT_3: Mask2Former Backbone + Patch Bézier Decoder

## 1. Local macOS Smoke Test (CPU/MPS, ~30 seconds)

```bash
cd /path/to/surgical_ai  # workspace root

python experiments/EXPERIMENT_3/scripts/train.py \
    --train_dir data/L3D/Train \
    --val_dir   data/L3D/Val \
    --smoke_test
```

Expected output:
- `Pred class shape: torch.Size([1, 64, 4])`
- `Pred bezier shape: torch.Size([1, 64, 4, 2])`
- `Backward pass successful.`

---

## 2. Kaggle GPU Run (T4 ~16GB — batch_size=1, accum=4)

> Use the **Kaggle Notebook**: `notebooks/EXP3_PatchBezier_Kaggle.ipynb`
> Upload to Kaggle, add the L3D dataset, enable GPU T4, run all cells.
>
> The notebook auto-discovers the dataset from `/kaggle/input` via scored heuristics.
> Results are zipped to `/kaggle/working/EXPERIMENT_3_PatchBezier_RESULTS.zip`.

If running via script instead of notebook:
```bash
python experiments/EXPERIMENT_3/scripts/train.py \
    --train_dir /kaggle/input/laparoscopic-liver-3d/Train/images \
    --val_dir   /kaggle/input/laparoscopic-liver-3d/Val/images \
    --test_dir  /kaggle/input/laparoscopic-liver-3d/Test/images \
    --epochs 60 \
    --batch_size 1 \
    --accumulation_steps 4 \
    --lr 8e-5 \
    --save_dir /kaggle/working/EXP3_results
```

---

## 3. HPC Server (A100-40GB — batch_size=4, no accumulation)

> Use the **Slurm script**: `scripts/run_server.sbatch`

```bash
# Submit job
sbatch experiments/EXPERIMENT_3/scripts/run_server.sbatch

# Monitor
squeue -u $USER
tail -f /data/khoalq/logs/exp3_patch_bezier_<JOB_ID>.out

# Or run directly (interactive session)
python -u /data/khoalq/surgical_ai/experiments/EXPERIMENT_3/scripts/train.py \
    --train_dir /data/khoalq/data/L3D/Train \
    --val_dir   /data/khoalq/data/L3D/Val \
    --test_dir  /data/khoalq/data/L3D/Test \
    --epochs 60 \
    --batch_size 4 \
    --accumulation_steps 1 \
    --lr 8e-5 \
    --num_workers 8 \
    --save_dir /data/khoalq/checkpoints/exp3_patch_bezier_60ep \
    --eval_splits both
```

---

## Key Differences: Kaggle vs Server

| Setting | Kaggle T4 (~16GB) | Server A100-40GB |
|---|---|---|
| `batch_size` | 1 | **4** |
| `accumulation_steps` | 4 | **1** (no accumulation) |
| Effective batch | 4 | 4 |
| `num_workers` | 2 | **8** |
| Est. training time | ~6-8 hrs | ~2-3 hrs |

---

## Output Artifacts (both environments)

| File | Description |
|---|---|
| `best_model.pth` | Checkpoint at best Val Macro Dice |
| `metrics_summary.json` | Full val + test benchmark results |
| `val_per_frame_metrics.csv` | 122-frame Val breakdown |
| `test_per_frame_metrics.csv` | 109-frame Test breakdown |
| `patient_40_diagnostics/` | 4-panel RGB\|GT\|Pred\|Error montages |
| `training_log.csv` | Epoch-by-epoch loss + metric curves |
| `EXPERIMENT_3_PatchBezier_RESULTS.zip` | One-click download archive |
