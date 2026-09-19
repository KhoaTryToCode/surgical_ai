#!/usr/bin/env bash
#SBATCH --job-name=exp5_junction_m2f
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/data/khoalq/logs/%x-%j.out
#SBATCH --error=/data/khoalq/logs/%x-%j.err

# ==============================================================================
# SLURM: EXPERIMENT_5 — Junction-Steered Mask2Former
# Target Server: gpu-a240 (100.82.42.48)
# Primary Hardware: NVIDIA A100-PCIE-40GB (batch_size=4, 8 workers, no accum)
# ==============================================================================

set -e
set -o pipefail

echo "=================================================================="
echo "🚀 EXPERIMENT_5: Junction-Steered Mask2Former Training"
echo "   Job ID   : $SLURM_JOB_ID"
echo "   Node     : $(hostname)"
echo "   GPU(s)   : $CUDA_VISIBLE_DEVICES"
echo "   Start    : $(date)"
echo "=================================================================="

# ── Environment & Directories ──────────────────────────────────────────────────
mkdir -p /data/khoalq/logs /data/khoalq/checkpoints/exp5_junction_m2f

source "/data/khoalq/miniconda3/etc/profile.d/conda.sh"
conda activate surgical_ai

export PYTHONPATH=/data/khoalq/surgical_ai:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="/data/khoalq/surgical_ai"
DATA_ROOT="/data/khoalq/data/L3D"
SAVE_DIR="/data/khoalq/checkpoints/exp5_junction_m2f"

# ── Sanity Checks ──────────────────────────────────────────────────────────────
if [ ! -d "${DATA_ROOT}/Train" ]; then
    echo "ERROR: Train directory not found: ${DATA_ROOT}/Train"
    exit 1
fi

echo ""
echo "Dataset root  : $DATA_ROOT"
echo "Checkpoint out: $SAVE_DIR"
echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Training Execution ─────────────────────────────────────────────────────────
# A100-40GB: batch_size=4, num_workers=8, no accumulation needed
python -u "${REPO_ROOT}/experiments/EXPERIMENT_5/scripts/train.py" \
    --data_dir      "$DATA_ROOT" \
    --out_dir       "$SAVE_DIR"  \
    --epochs        60           \
    --batch_size    4            \
    --accum_steps   1            \
    --lr_head       1e-4         \
    --lr_backbone   1e-5         \
    --lambda_coord  5.0          \
    --lambda_vis    1.0          \
    --num_workers   8            \
    --eval_splits   both         \
    --device        cuda         \
    2>&1 | tee "/data/khoalq/logs/exp5_junction_m2f_${SLURM_JOB_ID}.log"

echo ""
echo "=================================================================="
echo "✅ EXPERIMENT_5 COMPLETED AT $(date)"
echo "   Results in: $SAVE_DIR"
echo "=================================================================="
