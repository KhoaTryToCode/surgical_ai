# EXPERIMENT_13: Execution Guide & Run Commands

## 1. Prerequisites: Dataset Setup on Server `gpu-a240`

To download or link the L3D-2K dataset on the remote cluster (`gpu-a240`):

```bash
# 1. SSH into the GPU cluster
ssh khoalq@gpu-a240

# 2. Configure Kaggle Token (if not already done)
mkdir -p ~/.kaggle
echo "KGAT_68b57a10ef21ac15ecf07c9d6939a8bf" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

# 3. Download L3D-2K via KaggleHub and create symlink to /data/khoalq/data/L3D-2K
KAGGLE_API_TOKEN="KGAT_68b57a10ef21ac15ecf07c9d6939a8bf" \
KAGGLEHUB_CACHE=/data/khoalq/data/kagglehub_cache \
/data/khoalq/miniconda3/envs/surgical_ai/bin/python -c '
import os, kagglehub
print("Downloading khoale05/l3d-2k...")
downloaded_path = kagglehub.dataset_download("khoale05/l3d-2k")
print("Downloaded path:", downloaded_path)
target_link = "/data/khoalq/data/L3D-2K"
if not os.path.exists(target_link):
    os.symlink(downloaded_path, target_link)
    print(f"Created symlink: {target_link} -> {downloaded_path}")
else:
    print(f"Dataset already exists at: {target_link}")
'

# 4. Verify directory contents
ls -la /data/khoalq/data/L3D-2K
```

---

## 2. Server SLURM Training Submission

Submit the training job directly via SLURM:

```bash
cd /data/khoalq/surgical_ai
git pull origin main

sbatch experiments/EXPERIMENT_13/scripts/run_server.sbatch
```

### Real-Time Monitoring
```bash
# Check queue status
squeue -u khoalq

# Tail standard output log
tail -f /data/khoalq/logs/exp13_l3d2k_junction_m2f-*.out

# Tail training run log
tail -f /data/khoalq/logs/exp13_l3d2k_junction_m2f_*.log
```

---

## 3. Local Verification & Smoke Test

To verify code integrity on local macOS before full execution:

```bash
cd "/Users/khoale/Downloads/Surgical AI"

python3 experiments/EXPERIMENT_13/scripts/train.py \
    --data_dir "/Users/khoale/Downloads/L3D-2K" \
    --out_dir experiments/EXPERIMENT_13/results \
    --batch_size 1 \
    --smoke_test \
    --device cpu
```

---

## 4. Standalone Evaluation Command

To re-run evaluation on an existing checkpoint with both L3D-2K and original L3D test sets:

```bash
python experiments/EXPERIMENT_13/scripts/evaluate.py \
    --checkpoint /data/khoalq/checkpoints/exp13_l3d2k_junction_m2f/best_model.pth \
    --data_dir /data/khoalq/data/L3D-2K \
    --eval_original_l3d /data/khoalq/data/L3D \
    --out_dir /data/khoalq/checkpoints/exp13_l3d2k_junction_m2f \
    --device cuda
```
