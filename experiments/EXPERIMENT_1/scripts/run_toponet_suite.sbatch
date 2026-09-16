#!/usr/bin/env bash
#SBATCH --job-name=toponet_suite
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/data/khoalq/logs/%x-%j.out
#SBATCH --error=/data/khoalq/logs/%x-%j.err

# ==============================================================================
# SLURM SUITE: TOPONET REPLICATION & SYSTEMATIC ABLATION STUDY (L3D DATASET)
# Target Server: gpu-a240 (100.82.42.48)
# Storage Location: /data/khoalq/
# ==============================================================================

# Fail immediately on unhandled error or piped error
set -e
set -o pipefail

echo "=== INITIALIZING TOPONET EXECUTION ON $(hostname) ==="
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

# ==============================================================================
# SECTION A: RUNNING ON KAGGLE (Non-clDice Runs)
# Kept documented for record; commented out to avoid duplicate compute on server
# ==============================================================================
# [1/6] TopoNet Baseline (ResNet34 + Decoder, Soft Dice only)
# Location: experiments/EXPERIMENT_1/notebooks/TopoNet_Run_2_Baseline.ipynb
# python -u /data/khoalq/surgical_ai/experiments/EXPERIMENT_1/scripts/train_toponet.py \
#     --ablation baseline \
#     --epochs 100 \
#     --batch_size 4 \
#     --accumulation_steps 1 \
#     --train_dir "$TRAIN_DIR" --val_dir "$VAL_DIR" --test_dir "$TEST_DIR" \
#     --save_dir /data/khoalq/checkpoints/toponet_baseline_100ep \
#     --eval_splits both

# [2/6] TopoNet w/o Lcl (Dice + Perceptual, without Centerline Dice)
# Location: experiments/EXPERIMENT_1/notebooks/TopoNet_Run_4_wo_Lcl.ipynb
# python -u /data/khoalq/surgical_ai/experiments/EXPERIMENT_1/scripts/train_toponet.py \
#     --ablation wo_lcl \
#     --epochs 100 \
#     --batch_size 4 \
#     --accumulation_steps 1 \
#     --train_dir "$TRAIN_DIR" --val_dir "$VAL_DIR" --test_dir "$TEST_DIR" \
#     --save_dir /data/khoalq/checkpoints/toponet_wo_lcl_100ep \
#     --eval_splits both

# [3/6] TopoNet w/o Lper w/o Lcl (Soft Dice only, with Boundary-Topology Fusion)
# Location: experiments/EXPERIMENT_1/notebooks/TopoNet_Run_5_wo_Lper_Lcl.ipynb
# python -u /data/khoalq/surgical_ai/experiments/EXPERIMENT_1/scripts/train_toponet.py \
#     --ablation wo_lper_lcl \
#     --epochs 100 \
#     --batch_size 4 \
#     --accumulation_steps 1 \
#     --train_dir "$TRAIN_DIR" --val_dir "$VAL_DIR" --test_dir "$TEST_DIR" \
#     --save_dir /data/khoalq/checkpoints/toponet_wo_lper_lcl_100ep \
#     --eval_splits both

# ==============================================================================
# SECTION B: ACTIVE RUNS ON GPU-A240 HPC SERVER (Topological clDice Ablations)
# ==============================================================================

# Target mode selector: defaults to 'all_server' (runs full, wo_lper, wo_btf sequentially)
# Or can be called directly with a specific mode: sbatch run_toponet_suite.sbatch wo_lper
TARGET_MODE="${1:-all_server}"

run_ablation () {
    local mode=$1
    local out_dir="/data/khoalq/checkpoints/toponet_${mode}_50ep"
    local log_file="/data/khoalq/logs/toponet_${mode}.log"

    echo ""
    echo "======================================================================"
    echo "🚀 STARTING TOPONET ABLATION: [${mode^^}]"
    echo "   Output Directory: $out_dir"
    echo "   Log File        : $log_file"
    echo "   Start Timestamp : $(date)"
    echo "======================================================================"

    python -u /data/khoalq/surgical_ai/experiments/EXPERIMENT_1/scripts/train_toponet.py \
        --ablation "$mode" \
        --epochs 50 \
        --batch_size 2 \
        --accumulation_steps 2 \
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
