# Run Commands — EXPERIMENT_4: Landmark Master Tokens + Patch Bézier Decoder

## 1. Local macOS Smoke Test (CPU/MPS, ~30-60 seconds)

Test that the 67-token architecture, forward pass, loss calculation, backward pass, and rasterizer execute without errors:

```bash
cd /Users/khoale/Downloads/Surgical\ AI

# Run the local unit verification test
python3 experiments/EXPERIMENT_4/scripts/test_local_forward.py
```

Expected output:
- `pred_class shape: torch.Size([2, 64, 4])`
- `pred_bezier shape: torch.Size([2, 64, 4, 2])`
- `pred_presence shape: torch.Size([2, 3])`
- `pred_centroid shape: torch.Size([2, 3, 2])`
- `Total Loss: ...`
- `Backward pass successful. Gradients present on landmark_tokens.`
- `Rasterized canvas shape: (2, 1024, 1024)`

---

## 2. HPC Server (A100-40GB — batch_size=4, no accumulation)

Target server: `gpu-a240` (100.82.42.48), user `khoalq`.

### A. Deploy Changes to Server
```bash
# On local machine:
git add experiments/EXPERIMENT_4/
git commit -m "feat(exp4): implement Landmark Master Tokens + Patch Bézier model"
git push

# On server (ssh khoalq@100.82.42.48):
cd /data/khoalq/surgical_ai
git pull
```

### B. Submit Training Job via Slurm
```bash
# Submit job
sbatch experiments/EXPERIMENT_4/scripts/run_server.sbatch

# Check status
squeue -u $USER

# Follow live output logs
tail -f /data/khoalq/logs/exp4_landmark_bezier_*.out
```

### C. Direct Interactive Session (Alternative)
```bash
conda activate surgical_ai
export PYTHONPATH=/data/khoalq/surgical_ai:$PYTHONPATH
export PYTHONUNBUFFERED=1

python -u /data/khoalq/surgical_ai/experiments/EXPERIMENT_4/scripts/train.py \
    --train_dir     /data/khoalq/data/L3D/Train \
    --val_dir       /data/khoalq/data/L3D/Val \
    --test_dir      /data/khoalq/data/L3D/Test \
    --epochs        60 \
    --batch_size    4 \
    --accumulation_steps 1 \
    --lr            8e-5 \
    --weight_decay  3e-5 \
    --grid_size     8 \
    --single_scale \
    --save_dir      /data/khoalq/checkpoints/exp4_landmark_bezier_60ep \
    --num_workers   8 \
    --eval_splits   both
```

---

## 3. Kaggle GPU Run (T4 ~16GB — batch_size=1, accum=4)

> Use the **Kaggle Notebook**: `experiments/EXPERIMENT_4/notebooks/EXP4_LandmarkBezier_Kaggle.ipynb`
> 1. Upload this notebook to Kaggle.
> 2. Attach the `laparoscopic-liver-3d` (L3D) dataset to `/kaggle/input`.
> 3. Turn on GPU accelerator (Tesla T4).
> 4. Run All Cells.
> 
> The notebook handles dataset auto-discovery, training with mixed precision, evaluation on Val + Test, and zips the results to `/kaggle/working/EXPERIMENT_4_LandmarkBezier_RESULTS.zip`.

If executing via command-line in a Kaggle container:
```bash
python experiments/EXPERIMENT_4/scripts/train.py \
    --train_dir /kaggle/input/laparoscopic-liver-3d/Train/images \
    --val_dir   /kaggle/input/laparoscopic-liver-3d/Val/images \
    --test_dir  /kaggle/input/laparoscopic-liver-3d/Test/images \
    --epochs 60 \
    --batch_size 1 \
    --accumulation_steps 4 \
    --lr 8e-5 \
    --single_scale \
    --save_dir /kaggle/working/EXP4_results
```

---

## 4. Standalone Evaluation (Evaluate Existing Checkpoint)

Evaluate `best_model.pth` across both Val & Test sets and generate diagnostic montages with landmark crosshairs:

```bash
# On server:
bash experiments/EXPERIMENT_4/scripts/run_eval_server.sh

# Or directly:
python -u /data/khoalq/surgical_ai/experiments/EXPERIMENT_4/scripts/evaluate.py \
    --checkpoint /data/khoalq/checkpoints/exp4_landmark_bezier_60ep/best_model.pth \
    --val_dir /data/khoalq/data/L3D/Val \
    --test_dir /data/khoalq/data/L3D/Test \
    --save_dir /data/khoalq/checkpoints/exp4_landmark_bezier_60ep \
    --batch_size 4 \
    --eval_splits both
```

---

## Output Artifacts & Metrics Comparison

| Artifact | Description |
|---|---|
| `best_model.pth` | Weights saved at highest Val Macro Dice |
| `metrics_summary.json` | Val & Test Macro Dice, ASSD, IoU, per-class breakdown |
| `val_predictions.csv` | Frame-by-frame metrics for all 122 validation frames |
| `test_predictions.csv` | Frame-by-frame metrics for all 109 test frames |
| `patient_40_diagnostics_final/` | 4-panel RGB \| GT \| Pred \| Error montages with centroid markers |
| `training_log.csv` | Per-epoch loss and validation metrics across 60 epochs |
| `results.zip` | Complete bundled archive of model, logs, and visualizations |
