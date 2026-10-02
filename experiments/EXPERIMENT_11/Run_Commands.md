# EXPERIMENT_11 Run Commands: Targeted Inversion-Weighted Junction-Steered Mask2Former

## 1. Quick Pipeline Verification (Local CPU / Mac Smoke Test)

Run 1 step of training and validation to verify models, loss computation, WeightedRandomSampler, and metric evaluation:

```bash
cd "/Users/khoale/Downloads/Surgical AI"
python3 experiments/EXPERIMENT_11/scripts/train.py \
    --data_dir   "data/L3D" \
    --out_dir    "experiments/EXPERIMENT_11/results" \
    --batch_size 2 \
    --smoke_test \
    --device     cpu
```

---

## 2. Server Cluster Execution (SLURM on gpu-a240)

### Step 1: Git Push from Local Machine
```bash
git add experiments/EXPERIMENT_11/
git commit -m "feat(exp11): implement targeted inversion re-weighting and photometric jitter on Junction-Steered Mask2Former"
git push origin master
```

### Step 2: Launch Job on Cluster (`gpu-a240`)
```bash
ssh khoalq@gpu-a240
cd /data/khoalq/surgical_ai
git pull origin master
sbatch experiments/EXPERIMENT_11/scripts/run_server.sbatch
```

### Step 3: Monitor Live Training Output
```bash
# Check queue
squeue -u khoalq

# Tail stdout / stderr logs
tail -f /data/khoalq/logs/exp11_deformed_weighted_m2f_*.out

# Or view training execution log
tail -f /data/khoalq/logs/exp11_deformed_weighted_m2f_*.log
```

---

## 3. Results Retrieval (Trap 14 Standard)

Once training completes, download the complete packaged results archive (`results.zip`) directly from the unified server checkpoint directory:

```bash
rsync -avzP khoalq@gpu-a240:/data/khoalq/checkpoints/exp11_deformed_weighted_m2f/results.zip ./experiments/EXPERIMENT_11/results/exp11_results.zip
```

To extract diagnostic montages and CSV summaries locally:
```bash
unzip -o ./experiments/EXPERIMENT_11/results/exp11_results.zip -d ./experiments/EXPERIMENT_11/results/
```
