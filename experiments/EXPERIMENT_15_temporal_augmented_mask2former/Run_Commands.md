# Run Commands: EXPERIMENT_15 (Temporal-Augmented Mask2Former, $T=3$)

## 1. Local Smoke Test (macOS CPU / Apple Silicon MPS)

Verify the pipeline end-to-end with a 1-step test:

```bash
python experiments/EXPERIMENT_15_temporal_augmented_mask2former/scripts/train.py \
    --data_dir data/L3D \
    --image_size 512 \
    --batch_size 1 \
    --epochs 1 \
    --smoke_test \
    --device cpu
```

---

## 2. Remote Cluster Execution (SLURM on `gpu-a240`)

### 20GB VRAM Profile (A100 MIG 3g.20gb Partition — Recommended):
```bash
sbatch experiments/EXPERIMENT_15_temporal_augmented_mask2former/scripts/run_server_20gb.sbatch
```

### 10GB VRAM Profile (A100 MIG 2g.10gb Partition):
```bash
sbatch experiments/EXPERIMENT_15_temporal_augmented_mask2former/scripts/run_server_10gb.sbatch
```

### Full A100 GPU Profile:
```bash
sbatch experiments/EXPERIMENT_15_temporal_augmented_mask2former/scripts/run_server.sbatch
```

### Monitor SLURM Jobs:
```bash
squeue -u khoalq
tail -f /data/khoalq/logs/exp15_temporal_m2f_20gb_<JOB_ID>.log
```

---

## 3. Kaggle GPU Execution (Tesla T4 / P100)

1. Open a new Kaggle Notebook with GPU accelerator enabled (T4 or P100).
2. Attach the L3D dataset mounts:
   - `khoatrytopublish/l3d-train`
   - `khoatrytopublish/l3d-val`
   - `khoatrytopublish/l3d-test`
3. Upload or import the generated self-contained notebook:
   `experiments/EXPERIMENT_15_temporal_augmented_mask2former/notebooks/EXP15_Temporal_Mask2Former_Kaggle.ipynb`
4. Run all cells. Outputs and the `results.zip` bundle will be saved to `/kaggle/working/results_exp15/results.zip`.

---

## 4. Standalone Evaluation & Diagnostics Command

To re-evaluate a trained checkpoint and regenerate the top 15 worst failure cases:

```bash
python experiments/EXPERIMENT_15_temporal_augmented_mask2former/scripts/evaluate.py \
    --ckpt experiments/EXPERIMENT_15_temporal_augmented_mask2former/results/best_model.pth \
    --data_dir data/L3D \
    --clip_len 3 \
    --image_size 1024 \
    --device cuda
```
