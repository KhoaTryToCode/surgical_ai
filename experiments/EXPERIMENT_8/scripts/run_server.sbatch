#!/usr/bin/env bash
#SBATCH --job-name=exp8_human_anchor_m2f
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100_2g.10gb_mps:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/data/khoalq/logs/%x-%j.out
#SBATCH --error=/data/khoalq/logs/%x-%j.err

# ==============================================================================
# SLURM: EXPERIMENT_8 — Human-Anchor-Steered Mask2Former
# Target Server: gpu-a240 (100.82.42.48)
# Primary Hardware: NVIDIA A100 MIG 2g.10gb MPS (10 GB Dedicated Isolated VRAM)
# ==============================================================================

set -e
set -o pipefail

echo "=================================================================="
echo "🚀 EXPERIMENT_8: Human-Anchor-Steered Mask2Former Training"
echo "   Job ID   : $SLURM_JOB_ID"
echo "   Node     : $(hostname)"
echo "   GPU(s)   : $CUDA_VISIBLE_DEVICES"
echo "   Start    : $(date)"
echo "=================================================================="

# ── Environment & Directories ──────────────────────────────────────────────────
mkdir -p /data/khoalq/logs /data/khoalq/checkpoints/exp8_human_anchor_m2f

source "/data/khoalq/miniconda3/etc/profile.d/conda.sh"
conda activate surgical_ai

export PYTHONPATH=/data/khoalq/surgical_ai:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="/data/khoalq/surgical_ai"
DATA_ROOT="/data/khoalq/data/L3D"

# Auto-detect anchor JSON location (supports /data/khoalq/data/L3D/ or repo location)
if [ -f "${DATA_ROOT}/train_biological_anchors_human.json" ]; then
    ANCHOR_JSON="${DATA_ROOT}/train_biological_anchors_human.json"
else
    ANCHOR_JSON="${REPO_ROOT}/data/llm_annotate/train_biological_anchors_human.json"
fi

SAVE_DIR="/data/khoalq/checkpoints/exp8_human_anchor_m2f"

# ── Sanity Checks ──────────────────────────────────────────────────────────────
if [ ! -d "${DATA_ROOT}/Train" ]; then
    echo "ERROR: Train directory not found: ${DATA_ROOT}/Train"
    exit 1
fi

if [ ! -f "${ANCHOR_JSON}" ]; then
    echo "ERROR: Human anchor file not found: ${ANCHOR_JSON}"
    exit 1
fi

echo ""
echo "Dataset root  : $DATA_ROOT"
echo "Anchor JSON   : $ANCHOR_JSON"
echo "Checkpoint out: $SAVE_DIR"
echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Training Execution ─────────────────────────────────────────────────────────
# Training with batch_size=1, accum_steps=4 (effective batch_size=4, AMP enabled, 10GB VRAM optimized)
python -u "${REPO_ROOT}/experiments/EXPERIMENT_8/scripts/train.py" \
    --data_dir      "$DATA_ROOT"   \
    --anchor_json   "$ANCHOR_JSON" \
    --out_dir       "$SAVE_DIR"    \
    --epochs        60             \
    --batch_size    1              \
    --accum_steps   4              \
    --lr_head       1e-4           \
    --lr_backbone   1e-5           \
    --lambda_coord  5.0            \
    --lambda_vis    1.0            \
    --num_workers   8              \
    --eval_splits   both           \
    --device        cuda           \
    2>&1 | tee "/data/khoalq/logs/exp8_human_anchor_m2f_${SLURM_JOB_ID}.log"

echo ""
echo "=================================================================="
echo "✅ EXPERIMENT_8 COMPLETED AT $(date)"
echo "   Results in      : $SAVE_DIR"
echo "   Packaged Archive: $SAVE_DIR/results.zip"
echo "=================================================================="
