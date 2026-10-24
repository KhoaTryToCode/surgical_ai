# EXPERIMENT 12: EXECUTION COMMANDS

## 1. Local Smoke Test (macOS / CPU)
Run a single-step verification pass to ensure modules, loss balancing, and data pipeline function without error:

```bash
python3 experiments/EXPERIMENT_12/scripts/train.py \
    --data_dir      data/L3D \
    --cholec_dir    data/cholecseg8k \
    --out_dir       experiments/EXPERIMENT_12/results \
    --epochs        1 \
    --batch_size    1 \
    --accum_steps   1 \
    --smoke_test \
    --device        cpu
```

---

## 2. Server Cluster Execution (`gpu-a240`)

### Step A: Sync Repository to Server
From your local terminal:
```bash
git add experiments/EXPERIMENT_12
git commit -m "feat(exp12): Dual-Decoder Mask2Former co-training L3D and CholecSeg8k"
git push origin main
```

On server (`gpu-a240`):
```bash
cd /data/khoalq/surgical_ai
git pull origin main
```

### Step B: Submit SLURM Job
```bash
sbatch experiments/EXPERIMENT_12/scripts/run_server.sbatch
```

### Step C: Monitor Job Progress
Check queue status:
```bash
squeue -u khoalq
```

Stream live training logs:
```bash
tail -f /data/khoalq/logs/exp12_dual_decoder_m2f-*.out
```

Grep validation dice progress across epochs:
```bash
grep -E "NEW BEST CHECKPOINT|Validating on L3D|ValDice:" /data/khoalq/logs/exp12_dual_decoder_m2f-*.out
```

---

## 3. Post-Training Artifact Verification (Trap 14)
Verify that the checkpoint and packaged archive were created:
```bash
ls -lh /data/khoalq/checkpoints/exp12_dual_decoder_m2f/best_model.pth
ls -lh /data/khoalq/checkpoints/exp12_dual_decoder_m2f/results.zip
```
