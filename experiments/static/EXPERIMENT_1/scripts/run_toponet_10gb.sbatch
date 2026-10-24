#!/usr/bin/env bash
#SBATCH --job-name=toponet_10gb
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
# SLURM SUITE: TOPONET 10GB GPU RUNNER (MIG 2g.10gb on GPU-A240)
# Peak VRAM: 7.51 GB / 9.75 GB (with 2.2 GB safety margin)
# ==============================================================================

set -e
set -o pipefail

echo "=== INITIALIZING TOPONET 10GB EXECUTION ON $(hostname) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPU (CUDA_VISIBLE_DEVICES): $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"

# 1. Directories & Environment Activation (Per Cluster Manual Standard)
mkdir -p /data/khoalq/logs /data/khoalq/checkpoints

source "/data/khoalq/miniconda3/etc/profile.d/conda.sh"
conda activate surgical_ai
export PYTHONPATH=/data/khoalq/surgical_ai:/data/khoalq/surgical_ai/repos/TopoNet:$PYTHONPATH
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

# Target mode selector: defaults to 'all_server' (runs full, wo_lper, wo_btf sequentially)
TARGET_MODE="${1:-all_server}"

run_ablation () {
    local mode=$1
    local out_dir="/data/khoalq/checkpoints/toponet_${mode}_50ep"
    local log_file="/data/khoalq/logs/toponet_${mode}.log"

    echo ""
    echo "======================================================================"
    echo "🚀 STARTING TOPONET ABLATION ON 10GB GPU: [${mode^^}]"
    echo "   Output Directory: $out_dir"
    echo "   Log File        : $log_file"
    echo "   Start Timestamp : $(date)"
    echo "======================================================================"

    python -u /data/khoalq/surgical_ai/experiments/EXPERIMENT_1/scripts/train_toponet.py \
        --ablation "$mode" \
        --epochs 50 \
        --batch_size 1 \
        --accumulation_steps 4 \
        --cl_size 512 \
        --train_dir "$TRAIN_DIR" \
        --val_dir "$VAL_DIR" \
        --test_dir "$TEST_DIR" \
        --save_dir "$out_dir" \
        --eval_splits both 2>&1 | tee "$log_file"

    echo "✅ FINISHED ABLATION: [${mode^^}] AT $(date)"
}

if [ "$TARGET_MODE" = "full" ] || [ "$TARGET_MODE" = "all_server" ]; then
    # [4/6] Full TopoNet (Dice + clDice + Perceptual + BTF)
    run_ablation "full"
fi

if [ "$TARGET_MODE" = "wo_lper" ] || [ "$TARGET_MODE" = "all_server" ]; then
    # [5/6] TopoNet w/o Lper (Dice + clDice + BTF)
    run_ablation "wo_lper"
fi

if [ "$TARGET_MODE" = "wo_btf" ] || [ "$TARGET_MODE" = "all_server" ]; then
    # [6/6] TopoNet w/o BTF (Dice + clDice + Perceptual)
    run_ablation "wo_btf"
fi

echo ""
echo "======================================================================"
echo "🎉 ALL REQUESTED TOPONET ABLATIONS COMPLETED SUCCESSFULLY AT $(date)"
echo "======================================================================"
