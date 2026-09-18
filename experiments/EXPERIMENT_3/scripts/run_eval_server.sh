#!/usr/bin/env bash
# ==============================================================================
# Run Standalone Evaluation for EXPERIMENT_3 on gpu-a240 Server
# ==============================================================================
set -e

source "/data/khoalq/miniconda3/etc/profile.d/conda.sh"
conda activate surgical_ai

export PYTHONPATH=/data/khoalq/surgical_ai:$PYTHONPATH
export PYTHONUNBUFFERED=1

CHECKPOINT="/data/khoalq/checkpoints/exp3_patch_bezier_60ep/best_model.pth"
DATA_ROOT="/data/khoalq/data/L3D"
SAVE_DIR="/data/khoalq/checkpoints/exp3_patch_bezier_60ep"

echo "=================================================================="
echo "🚀 Running Standalone Evaluation for EXPERIMENT_3"
echo "   Checkpoint: $CHECKPOINT"
echo "   Output Dir: $SAVE_DIR"
echo "=================================================================="

python -u /data/khoalq/surgical_ai/experiments/EXPERIMENT_3/scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --val_dir "${DATA_ROOT}/Val" \
    --test_dir "${DATA_ROOT}/Test" \
    --save_dir "$SAVE_DIR" \
    --batch_size 4 \
    --num_workers 4 \
    --eval_splits both

echo ""
echo "=================================================================="
echo "✅ Evaluation complete! Results saved in: $SAVE_DIR"
echo "=================================================================="
