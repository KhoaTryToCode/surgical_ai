#!/usr/bin/env bash
#SBATCH --job-name=exp13_l3d2k_junction_m2f
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=/data/khoalq/logs/%x-%j.out
#SBATCH --error=/data/khoalq/logs/%x-%j.err

# ==============================================================================
# SLURM: EXPERIMENT_13 — Junction-Steered Mask2Former on L3D-2K
# Controlled Single-Variable Replication of EXPERIMENT_5
# Target Server: gpu-a240 (100.82.42.48)
# Primary Hardware: NVIDIA A100-PCIE-40GB (or MIG a100_3g.20gb:1 fallback)
# Hypothesis:
#   Controlled ablation testing if clinical patient cohort scale (39 -> 47 patients,
#   921 -> 1,532 train frames) breaks through the 68.13% Val Dice ceiling.
# Trap 14 Packaging:
#   SAVE_DIR contains best_model.pth and results.zip
# ==============================================================================

set -e
set -o pipefail

echo "=================================================================="
echo "🚀 EXPERIMENT_13: Junction-Steered Mask2Former on L3D-2K Training"
echo "   Job ID   : $SLURM_JOB_ID"
echo "   Node     : $(hostname)"
echo "   GPU(s)   : $CUDA_VISIBLE_DEVICES"
echo "   Start    : $(date)"
echo "=================================================================="

# ── Redirect Temporary Directories to prevent /tmp disk exhaustion crash ───────
mkdir -p /data/khoalq/tmp /data/khoalq/logs /data/khoalq/checkpoints/exp13_l3d2k_junction_m2f
export TMPDIR=/data/khoalq/tmp
export TEMP=/data/khoalq/tmp
export TMP=/data/khoalq/tmp
export MPLCONFIGDIR=/data/khoalq/tmp/mpl
export HF_HOME=/data/khoalq/tmp/hf

# ── Environment & Conda ────────────────────────────────────────────────────────
source "/data/khoalq/miniconda3/etc/profile.d/conda.sh"
conda activate surgical_ai

export PYTHONPATH=/data/khoalq/surgical_ai:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="/data/khoalq/surgical_ai"
DATA_ROOT="/data/khoalq/data/L3D-2K"
ORIG_L3D_ROOT="/data/khoalq/data/L3D"
SAVE_DIR="/data/khoalq/checkpoints/exp13_l3d2k_junction_m2f"

# ── Verify or Auto-Link L3D-2K Dataset ──────────────────────────────────────────
if [ ! -d "${DATA_ROOT}" ] || { [ ! -d "${DATA_ROOT}/train" ] && [ ! -d "${DATA_ROOT}/Train" ]; }; then
    echo "⚠️ L3D-2K not found at ${DATA_ROOT}. Checking kagglehub download cache..."
    export KAGGLEHUB_CACHE=/data/khoalq/data/kagglehub_cache
    python -c '
import os, kagglehub
print("Checking/Downloading khoale05/l3d-2k...")
path = kagglehub.dataset_download("khoale05/l3d-2k")
print("Downloaded path:", path)
target = "/data/khoalq/data/L3D-2K"
if os.path.islink(target) or os.path.exists(target):
    os.remove(target) if os.path.islink(target) else None
if not os.path.exists(target):
    os.symlink(path, target)
    print("Created symlink /data/khoalq/data/L3D-2K ->", path)
'
fi

echo ""
echo "Dataset root       : $DATA_ROOT"
echo "Original L3D root  : $ORIG_L3D_ROOT"
echo "Checkpoint out dir : $SAVE_DIR"
echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Training Execution ─────────────────────────────────────────────────────────
# Exact hyperparameter parity with EXPERIMENT_5:
# batch_size=2, accum_steps=2 (effective batch_size=4), 60 epochs, AdamW (1e-5 backbone, 1e-4 head)
python -u "${REPO_ROOT}/experiments/EXPERIMENT_13/scripts/train.py" \
    --data_dir           "$DATA_ROOT" \
    --out_dir            "$SAVE_DIR"  \
    --epochs             60           \
    --batch_size         2            \
    --accum_steps        2            \
    --lr_head            1e-4         \
    --lr_backbone        1e-5         \
    --lambda_coord       5.0          \
    --lambda_vis         1.0          \
    --num_workers        8            \
    --eval_splits        both         \
    --eval_original_l3d  "$ORIG_L3D_ROOT" \
    --device             cuda         \
    2>&1 | tee "/data/khoalq/logs/exp13_l3d2k_junction_m2f_${SLURM_JOB_ID}.log"

echo ""
echo "=================================================================="
echo "✅ EXPERIMENT_13 COMPLETED AT $(date)"
echo "   Checkpoint in   : $SAVE_DIR/best_model.pth"
echo "   Results Archive : $SAVE_DIR/results.zip"
echo "=================================================================="
