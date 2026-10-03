#!/usr/bin/env bash
#SBATCH --job-name=exp12_dual_decoder_m2f
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
# SLURM: EXPERIMENT_12 — Dual-Decoder Multi-Task Mask2Former (L3D + CholecSeg8k)
# Target Server: gpu-a240 (100.82.42.48)
# Primary Hardware: NVIDIA A100 (40 GB VRAM)
# Note: If A100 full is busy, fallback to 20 GB MIG: --gres=gpu:a100_3g.20gb:1
# Multi-Task Foundation Learning:
#   - Shared Swin-Tiny Backbone + MSDeformAttn Pixel Decoder
#   - Scene Decoder trained on CholecSeg8k (8,080 frames, 13 classes)
#   - Landmark Decoder trained on L3D (921 frames, 3 lines + 4 continuous junctions)
#   - Trap 14 Server Packaging Standard: SAVE_DIR and results.zip
# ==============================================================================

set -e
set -o pipefail

echo "=================================================================="
echo "🚀 EXPERIMENT_12: Dual-Decoder Multi-Task Mask2Former Training"
echo "   Job ID   : $SLURM_JOB_ID"
echo "   Node     : $(hostname)"
echo "   GPU(s)   : $CUDA_VISIBLE_DEVICES"
echo "   Start    : $(date)"
echo "=================================================================="

# ── Redirect Temporary Directories to prevent /tmp disk exhaustion crash ───────
mkdir -p /data/khoalq/tmp /data/khoalq/logs /data/khoalq/checkpoints/exp12_dual_decoder_m2f
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
DATA_ROOT="/data/khoalq/data/L3D"
CHOLEC_ROOT="/data/khoalq/data/cholecseg8k"
SAVE_DIR="/data/khoalq/checkpoints/exp12_dual_decoder_m2f"

# ── Sanity Checks ──────────────────────────────────────────────────────────────
if [ ! -d "${DATA_ROOT}/Train" ]; then
    echo "ERROR: L3D Train directory not found: ${DATA_ROOT}/Train"
    exit 1
fi

if [ ! -d "${CHOLEC_ROOT}" ]; then
    echo "ERROR: CholecSeg8k directory not found: ${CHOLEC_ROOT}"
    exit 1
fi

echo ""
echo "L3D Dataset root     : $DATA_ROOT"
echo "CholecSeg8k root     : $CHOLEC_ROOT"
echo "Checkpoint out dir   : $SAVE_DIR"
echo "lambda_cholec        : 0.5"
echo "Tier 1 Weight        : 15.0x"
echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Training Execution (Trap 14 Compliant) ──────────────────────────────────────
python -u "${REPO_ROOT}/experiments/EXPERIMENT_12/scripts/train.py" \
    --data_dir      "$DATA_ROOT"   \
    --cholec_dir    "$CHOLEC_ROOT" \
    --out_dir       "$SAVE_DIR"    \
    --epochs        60             \
    --batch_size    2              \
    --accum_steps   2              \
    --lambda_cholec 0.5            \
    --tier1_weight  15.0           \
    --lr_head       1e-4           \
    --lr_backbone   1e-5           \
    --lambda_coord  5.0            \
    --lambda_vis    1.0            \
    --num_workers   4              \
    --eval_splits   both           \
    --device        cuda           \
    2>&1 | tee "/data/khoalq/logs/exp12_dual_decoder_m2f_${SLURM_JOB_ID}.log"

echo ""
echo "=================================================================="
echo "✅ EXPERIMENT_12 COMPLETED AT $(date)"
echo "   Results in      : $SAVE_DIR"
echo "   Packaged Archive: $SAVE_DIR/results.zip"
echo "=================================================================="
