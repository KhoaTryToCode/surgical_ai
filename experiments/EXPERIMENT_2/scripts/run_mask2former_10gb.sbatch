#!/usr/bin/env bash
#SBATCH --job-name=mask2former_10gb
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/data/khoalq/logs/%x-%j.out
#SBATCH --error=/data/khoalq/logs/%x-%j.err

# ==============================================================================
# SLURM SUITE: MASK2FORMER 10GB GPU RUNNER (MIG 2g.10gb on GPU-A240)
# Conservative Profile: Micro-batch 1, Accumulation 4 (Effective Batch: 4)
# ==============================================================================

set -e
set -o pipefail

echo "=== INITIALIZING MASK2FORMER 10GB EXECUTION ON $(hostname) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPU (CUDA_VISIBLE_DEVICES): $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"

# 1. Directories & Environment Activation (Strict Cluster Standard)
mkdir -p /data/khoalq/logs /data/khoalq/checkpoints

source "/data/khoalq/miniconda3/etc/profile.d/conda.sh"
conda activate surgical_ai
export PYTHONPATH=/data/khoalq/surgical_ai:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_ROOT="/data/khoalq/data/L3D"
TRAIN_DIR="${DATA_ROOT}/Train"
VAL_DIR="${DATA_ROOT}/Val"
TEST_DIR="${DATA_ROOT}/Test"

# Verify dataset directory
if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: Train dataset directory $TRAIN_DIR not found!"
    exit 1
fi

echo "Dataset confirmed at: $DATA_ROOT"
nvidia-smi

# Target mode selector: defaults to 'all_server' (runs all 4 sequentially)
# Or can be called directly with a specific mode: sbatch run_mask2former_10gb.sbatch baseline
TARGET_MODE="${1:-all_server}"

run_ablation () {
    local mode=$1
    local out_dir="/data/khoalq/checkpoints/mask2former_${mode}_60ep"
    local log_file="/data/khoalq/logs/mask2former_${mode}.log"

    echo ""
    echo "======================================================================"
    echo "🚀 STARTING MASK2FORMER ABLATION ON 10GB GPU: [${mode^^}]"
    echo "   Output Directory: $out_dir"
    echo "   Log File        : $log_file"
    echo "   Start Timestamp : $(date)"
    echo "======================================================================"

    python -u /data/khoalq/surgical_ai/experiments/EXPERIMENT_2/scripts/train_server.py \
        --ablation "$mode" \
        --epochs 60 \
        --batch_size 1 \
        --accumulation_steps 4 \
        --train_dir "$TRAIN_DIR" \
        --val_dir "$VAL_DIR" \
        --test_dir "$TEST_DIR" \
        --save_dir "$out_dir" \
        --eval_splits both 2>&1 | tee "$log_file"

    echo "✅ FINISHED ABLATION: [${mode^^}] AT $(date)"
}

if [ "$TARGET_MODE" = "baseline" ] || [ "$TARGET_MODE" = "all_server" ]; then
    run_ablation "baseline"
fi

if [ "$TARGET_MODE" = "wo_masked_attn" ] || [ "$TARGET_MODE" = "all_server" ]; then
    run_ablation "wo_masked_attn"
fi

if [ "$TARGET_MODE" = "wo_multiscale" ] || [ "$TARGET_MODE" = "all_server" ]; then
    run_ablation "wo_multiscale"
fi

if [ "$TARGET_MODE" = "wo_self_attn" ] || [ "$TARGET_MODE" = "all_server" ]; then
    run_ablation "wo_self_attn"
fi

echo ""
echo "======================================================================"
echo "🎉 ALL REQUESTED MASK2FORMER ABLATIONS COMPLETED SUCCESSFULLY AT $(date)"
echo "======================================================================"
