#!/usr/bin/env bash
#SBATCH --job-name=exp11_deformed_weighted_m2f
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100_2g.10gb_mps:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=14:00:00
#SBATCH --output=/data/khoalq/logs/%x-%j.out
#SBATCH --error=/data/khoalq/logs/%x-%j.err

# ==============================================================================
# SLURM: EXPERIMENT_11 — Targeted Inversion-Weighted Junction-Steered Mask2Former
# Target Server: gpu-a240 (100.82.42.48)
# Primary Hardware: NVIDIA A100 MIG 2g.10gb MPS (10 GB Dedicated Isolated VRAM)
# Clean Controlled Ablation on EXP_5: Identical 2D continuous junction-steered
# architecture, inflating rare inverted/retracted training frames via
# WeightedRandomSampler (Tier 1: 15.0x, Tier 2: 5.0x) + Photometric ColorJitter.
# ==============================================================================

set -e
set -o pipefail

echo "=================================================================="
echo "🚀 EXPERIMENT_11: Targeted Inversion-Weighted Mask2Former Training"
echo "   Job ID   : $SLURM_JOB_ID"
echo "   Node     : $(hostname)"
echo "   GPU(s)   : $CUDA_VISIBLE_DEVICES"
echo "   Start    : $(date)"
echo "=================================================================="

# ── Redirect Temporary Directories to prevent /tmp disk exhaustion crash ───────
mkdir -p /data/khoalq/tmp /data/khoalq/logs /data/khoalq/checkpoints/exp11_deformed_weighted_m2f
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
SAVE_DIR="/data/khoalq/checkpoints/exp11_deformed_weighted_m2f"

# ── Sanity Checks ──────────────────────────────────────────────────────────────
if [ ! -d "${DATA_ROOT}/Train" ]; then
    echo "ERROR: Train directory not found: ${DATA_ROOT}/Train"
    exit 1
fi

echo ""
echo "Dataset root   : $DATA_ROOT"
echo "Checkpoint out : $SAVE_DIR"
echo "Tier 1 Weight  : 15.0x (All 24 Deformed/Inverted/Traction frames)"
echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Training Execution (Trap 14 Compliant) ──────────────────────────────────────
# Same as EXP_5: batch_size=2, accum_steps=2 (effective batch_size=4, AMP enabled)
python -u "${REPO_ROOT}/experiments/EXPERIMENT_11/scripts/train.py" \
    --data_dir      "$DATA_ROOT"   \
    --out_dir       "$SAVE_DIR"    \
    --epochs        60             \
    --batch_size    2              \
    --accum_steps   2              \
    --tier1_weight  15.0           \
    --lr_head       1e-4           \
    --lr_backbone   1e-5           \
    --lambda_coord  5.0            \
    --lambda_vis    1.0            \
    --num_workers   4              \
    --eval_splits   both           \
    --device        cuda           \
    2>&1 | tee "/data/khoalq/logs/exp11_deformed_weighted_m2f_${SLURM_JOB_ID}.log"

echo ""
echo "=================================================================="
echo "✅ EXPERIMENT_11 COMPLETED AT $(date)"
echo "   Results in      : $SAVE_DIR"
echo "   Packaged Archive: $SAVE_DIR/results.zip"
echo "=================================================================="
