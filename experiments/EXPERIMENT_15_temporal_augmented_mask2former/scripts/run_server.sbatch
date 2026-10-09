#!/usr/bin/env bash
#SBATCH --job-name=exp15_temporal_m2f
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/data/khoalq/logs/%x-%j.out
#SBATCH --error=/data/khoalq/logs/%x-%j.err

# ==============================================================================
# SLURM: EXPERIMENT_15 — Temporal-Augmented Mask2Former (Full GPU Profile)
# Target Server: gpu-a240 (100.82.42.48)
# Dedicated Hardware: Full NVIDIA A100 (40GB/80GB VRAM)
# Memory Strategy: Micro-batch 2 clips (T=3), Accumulation 2 (Effective: 4 clips / 12 frames)
# ==============================================================================

set -e
set -o pipefail

mkdir -p /data/khoalq/tmp /data/khoalq/logs /data/khoalq/checkpoints/exp15_temporal_m2f
export TMPDIR=/data/khoalq/tmp
export MPLCONFIGDIR=/data/khoalq/tmp/mpl
export HF_HOME=/data/khoalq/tmp/hf

source "/data/khoalq/miniconda3/etc/profile.d/conda.sh"
conda activate surgical_ai

export PYTHONPATH=/data/khoalq/surgical_ai:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="/data/khoalq/surgical_ai"
DATA_ROOT="/data/khoalq/data/L3D"
SAVE_DIR="/data/khoalq/checkpoints/exp15_temporal_m2f"

python -u "${REPO_ROOT}/experiments/EXPERIMENT_15_temporal_augmented_mask2former/scripts/train.py" \
    --data_dir      "$DATA_ROOT" \
    --out_dir       "$SAVE_DIR"  \
    --epochs        60           \
    --clip_len      3            \
    --max_gap       240          \
    --image_size    1024         \
    --batch_size    2            \
    --accum_steps   2            \
    --lr_head       1e-4         \
    --lr_backbone   1e-5         \
    --num_workers   4            \
    --eval_splits   both         \
    --device        cuda         \
    2>&1 | tee "/data/khoalq/logs/exp15_temporal_m2f_${SLURM_JOB_ID}.log"

python -u "${REPO_ROOT}/experiments/EXPERIMENT_15_temporal_augmented_mask2former/scripts/evaluate.py" \
    --ckpt          "$SAVE_DIR/best_model.pth" \
    --data_dir      "$DATA_ROOT" \
    --out_dir       "$SAVE_DIR"  \
    --clip_len      3            \
    --image_size    1024         \
    --device        cuda         \
    2>&1 | tee -a "/data/khoalq/logs/exp15_temporal_m2f_${SLURM_JOB_ID}.log"

if [ ! -f "$SAVE_DIR/results.zip" ]; then
    python3 -c "
import os, zipfile
source_dir = '$SAVE_DIR'
zip_path = os.path.join(source_dir, 'results.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f != 'results.zip':
                p = os.path.join(root, f)
                zipf.write(p, os.path.relpath(p, source_dir))
print('Packaged results.zip:', os.path.getsize(zip_path)/(1024*1024), 'MB')
"
fi
