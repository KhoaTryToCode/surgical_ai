#!/usr/bin/env bash
#SBATCH --job-name=exp10_depth_junction_m2f
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100_2g.10gb_mps:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/data/khoalq/logs/%x-%j.out
#SBATCH --error=/data/khoalq/logs/%x-%j.err

# ==============================================================================
# SLURM: EXPERIMENT_10 — Depth-Geometric Junction-Steered Mask2Former
# Target Server: gpu-a240 (100.82.42.48)
# Hardware: NVIDIA A100 MIG 2g.10gb MPS (10 GB Isolated Dedicated VRAM)
# ==============================================================================

set -e
set -o pipefail

echo "=================================================================="
echo "🚀 EXPERIMENT_10: Depth-Geometric Junction-Steered Mask2Former Training"
echo "   Job ID   : $SLURM_JOB_ID"
echo "   Node     : $(hostname)"
echo "   GPU(s)   : $CUDA_VISIBLE_DEVICES"
echo "   Start    : $(date)"
echo "=================================================================="

# ── Environment & Directories ──────────────────────────────────────────────────
mkdir -p /data/khoalq/logs /data/khoalq/checkpoints/exp10_depth_junction_m2f

source "/data/khoalq/miniconda3/etc/profile.d/conda.sh"
conda activate surgical_ai

export PYTHONPATH=/data/khoalq/surgical_ai:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="/data/khoalq/surgical_ai"
DATA_ROOT="/data/khoalq/data/L3D"
SAVE_DIR="/data/khoalq/checkpoints/exp10_depth_junction_m2f"
OUTPUT_DIR="${REPO_ROOT}/experiments/EXPERIMENT_10/results"

mkdir -p "$SAVE_DIR" "$OUTPUT_DIR"

# ── Sanity Checks ──────────────────────────────────────────────────────────────
if [ ! -d "${DATA_ROOT}/Train" ]; then
    echo "ERROR: Train directory not found: ${DATA_ROOT}/Train"
    exit 1
fi

echo ""
echo "Dataset root  : $DATA_ROOT"
echo "Checkpoint out: $SAVE_DIR"
echo "Results out   : $OUTPUT_DIR"
echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Training Execution ─────────────────────────────────────────────────────────
python -u "${REPO_ROOT}/experiments/EXPERIMENT_10/scripts/train.py" \
    --data-dir        "$DATA_ROOT"   \
    --output-dir      "$OUTPUT_DIR"  \
    --checkpoint-dir  "$SAVE_DIR"    \
    --epochs          60             \
    --batch-size      1              \
    --accum-steps     4              \
    --lr-head         1e-4           \
    --lr-backbone     1e-5           \
    --lambda-coord    5.0            \
    --lambda-vis      1.0            \
    --device          cuda           \
    2>&1 | tee "/data/khoalq/logs/exp10_depth_junction_m2f_${SLURM_JOB_ID}.log"

echo ""
echo "=================================================================="
echo "✅ EXPERIMENT_10 COMPLETED AT $(date)"
echo "   Results in      : $OUTPUT_DIR"
echo "   Packaged Archive: $OUTPUT_DIR/results.zip"
echo "=================================================================="
