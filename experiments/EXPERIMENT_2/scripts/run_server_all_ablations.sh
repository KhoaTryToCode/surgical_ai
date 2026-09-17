#!/bin/bash
# ==============================================================================
# Mask2Former Ablation Study - Dedicated Server Runner
# ==============================================================================
# Usage:
#   bash run_server_all_ablations.sh [MODE]
#
# Examples:
#   # Run specific mode:
#   bash run_server_all_ablations.sh Swin_MaskedAttn
#
#   # Run all 4 ablation runs sequentially:
#   bash run_server_all_ablations.sh all
#
# Custom environment overrides:
#   GPU_ID=0 BATCH_SIZE=4 EPOCHS=60 bash run_server_all_ablations.sh Swin_MaskedAttn
# ==============================================================================

set -e

GPU_ID=${GPU_ID:-0}
BATCH_SIZE=${BATCH_SIZE:-4}
ACCUMULATION=${ACCUMULATION:-1}
EPOCHS=${EPOCHS:-60}
LR=${LR:-8e-5}
DATA_PATH=${DATA_PATH:-""}
SAVE_DIR=${SAVE_DIR:-"./experiments/EXPERIMENT_2/results"}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/train_server.py"

cd "$REPO_ROOT"

echo "========================================================================"
echo "🚀 SURGICAL AI: EXPERIMENT_2 SERVER RUNNER"
echo "========================================================================"
echo "CUDA Device        : $GPU_ID"
echo "Batch Size         : $BATCH_SIZE (Accumulation: $ACCUMULATION)"
echo "Epochs             : $EPOCHS"
echo "Learning Rate      : $LR"
echo "Save Directory     : $SAVE_DIR"
if [ -n "$DATA_PATH" ]; then
  echo "Dataset Path       : $DATA_PATH"
fi
echo "========================================================================"

run_ablation() {
    local mode=$1
    echo ""
    echo "▶️ Starting Ablation Mode: [$mode] on CUDA:$GPU_ID..."
    
    EXTRA_ARGS=""
    if [ -n "$DATA_PATH" ]; then
        EXTRA_ARGS="--data_path $DATA_PATH"
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID python "$PYTHON_SCRIPT" \
        --mode "$mode" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --accumulation_steps "$ACCUMULATION" \
        --lr "$LR" \
        --save_dir "$SAVE_DIR" \
        $EXTRA_ARGS

    echo "✅ Completed: [$mode]"
}

TARGET_MODE=${1:-"all"}

if [ "$TARGET_MODE" == "all" ]; then
    echo "📋 Queueing All 4 Ablation Runs sequentially..."
    run_ablation "Swin_MaskedAttn"
    run_ablation "Swin_FullAttn"
    run_ablation "ResNet_MaskedAttn"
    run_ablation "ResNet_FullAttn"
    echo "🎉 All 4 Ablations Completed Successfully!"
else
    run_ablation "$TARGET_MODE"
fi
