#!/usr/bin/env bash
#SBATCH --job-name=exp4_landmark_bezier
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/data/khoalq/logs/%x-%j.out
#SBATCH --error=/data/khoalq/logs/%x-%j.err

# ==============================================================================
# SLURM: EXPERIMENT_4 — Landmark Master Tokens + Patch Bézier Decoder
# Target Server: gpu-a240 (100.82.42.48)
# Primary Hardware: NVIDIA A100-PCIE-40GB (20 GB+ VRAM — batch_size=4, no accum)
# ==============================================================================

set -e
set -o pipefail

echo "=================================================================="
echo "🚀 EXPERIMENT_4: Landmark Master Tokens + Patch Bézier Training"
echo "   Job ID   : $SLURM_JOB_ID"
echo "   Node     : $(hostname)"
echo "   GPU(s)   : $CUDA_VISIBLE_DEVICES"
echo "   Start    : $(date)"
echo "=================================================================="

# ── Environment ────────────────────────────────────────────────────────────────
mkdir -p /data/khoalq/logs /data/khoalq/checkpoints

source "/data/khoalq/miniconda3/etc/profile.d/conda.sh"
conda activate surgical_ai

export PYTHONPATH=/data/khoalq/surgical_ai:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="/data/khoalq/surgical_ai"
DATA_ROOT="/data/khoalq/data/L3D"
TRAIN_DIR="${DATA_ROOT}/Train"
VAL_DIR="${DATA_ROOT}/Val"
TEST_DIR="${DATA_ROOT}/Test"
SAVE_DIR="/data/khoalq/checkpoints/exp4_landmark_bezier_60ep"

# ── Sanity Checks ──────────────────────────────────────────────────────────────
if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: Train directory not found: $TRAIN_DIR"
    exit 1
fi

echo ""
echo "Dataset root  : $DATA_ROOT"
echo "Checkpoint out: $SAVE_DIR"
echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Training ───────────────────────────────────────────────────────────────────
# A100-40GB: batch_size=4, no gradient accumulation needed (effective batch=4 in one step)
python -u "${REPO_ROOT}/experiments/EXPERIMENT_4/scripts/train.py" \
    --train_dir     "$TRAIN_DIR"  \
    --val_dir       "$VAL_DIR"    \
    --test_dir      "$TEST_DIR"   \
    --epochs        60            \
    --batch_size    4             \
    --accumulation_steps 1        \
    --lr            8e-5          \
    --weight_decay  3e-5          \
    --grid_size     8             \
    --single_scale                \
    --save_dir      "$SAVE_DIR"   \
    --num_workers   8             \
    --eval_splits   both          \
    2>&1 | tee "/data/khoalq/logs/exp4_landmark_bezier_${SLURM_JOB_ID}.log"

echo ""
echo "=================================================================="
echo "✅ EXPERIMENT_4 COMPLETED AT $(date)"
echo "   Results in: $SAVE_DIR"
echo "=================================================================="
