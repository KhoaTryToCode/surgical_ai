# EXPERIMENT_9: Run Commands & Execution Guide

## 1. Local Smoke Test (macOS / CPU Verification)

```bash
# Run a quick 1-epoch / 4-sample smoke test locally to verify 4-channel RGB-D forward/backward:
python3 experiments/EXPERIMENT_9/scripts/train.py \
    --smoke_test \
    --device cpu \
    --num_workers 0
```

---

## 2. Server Sync to Cluster (`gpu-a240`)

```bash
# Push EXPERIMENT_9 directory to server:
rsync -avzP --exclude '__pycache__' --exclude '*.pth' \
    experiments/EXPERIMENT_9/ \
    khoalq@gpu-a240:/data/khoalq/surgical_ai/experiments/EXPERIMENT_9/
```

---

## 3. Launch SLURM Job on Server (`gpu-a240`)

```bash
# Submit the training job to the A100 MIG partition:
ssh khoalq@gpu-a240 "sbatch /data/khoalq/surgical_ai/experiments/EXPERIMENT_9/scripts/run_server.sbatch"
```

---

## 4. Monitor Server Job Execution

```bash
# Check queue status:
ssh khoalq@gpu-a240 "squeue -u khoalq"

# Tail live output logs:
ssh khoalq@gpu-a240 "tail -f /data/khoalq/logs/exp9_rgbd_human_anchor_m2f_*.out"

# Check GPU utilization & VRAM consumption:
ssh khoalq@gpu-a240 "nvidia-smi"
```

---

## 5. Download Results & Checkpoint to Local Machine

```bash
# Once training completes, download the packaged results and best checkpoint:
mkdir -p checkpoints/exp9_rgbd_human_anchor_m2f
rsync -avzP \
    khoalq@gpu-a240:/data/khoalq/checkpoints/exp9_rgbd_human_anchor_m2f/results.zip \
    checkpoints/exp9_rgbd_human_anchor_m2f/

rsync -avzP \
    khoalq@gpu-a240:/data/khoalq/checkpoints/exp9_rgbd_human_anchor_m2f/best_model.pth \
    checkpoints/exp9_rgbd_human_anchor_m2f/
```

---

## 6. Standalone Local Evaluation

```bash
# Run full evaluation on the downloaded checkpoint:
python3 experiments/EXPERIMENT_9/scripts/evaluate.py \
    --checkpoint checkpoints/exp9_rgbd_human_anchor_m2f/best_model.pth \
    --split Val \
    --out_dir experiments/EXPERIMENT_9/results
```
