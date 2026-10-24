#!/usr/bin/env bash
#SBATCH --job-name=exp14_stratified_m2f
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
# SLURM: EXPERIMENT_14 — Stratified Junction-Steered Mask2Former
# Patient Re-stratification (Option 1: Patient_40 <-> Patient_38 Swap)
# Target Server: gpu-a240 (100.82.42.48)
# Primary Hardware: NVIDIA A100-PCIE-40GB (or MIG a100_3g.20gb:1)
# Objective:
#   Test whether moving Patient 40 (101 frames, 15 flipped) into Train and Patient 38
#   (99 frames, 6 flipped) into Val enables the model to learn severe surgical traction,
#   and measure generalization on the frozen Test set (Patient 41).
# Deliverables:
#   SAVE_DIR contains best_model.pth, training_log.csv, val_predictions.csv,
#   test_predictions.csv, metrics_summary.json, diagnostics, and results.zip.
# ==============================================================================

set -e
set -o pipefail

echo "=================================================================="
echo "🚀 EXPERIMENT_14: Stratified Junction-Steered Mask2Former Training"
echo "   Job ID   : $SLURM_JOB_ID"
echo "   Node     : $(hostname)"
echo "   GPU(s)   : $CUDA_VISIBLE_DEVICES"
echo "   Start    : $(date)"
echo "=================================================================="

# ── Redirect Temporary Directories to prevent /tmp disk exhaustion crash ───────
mkdir -p /data/khoalq/tmp /data/khoalq/logs /data/khoalq/checkpoints/exp14_stratified_m2f
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
SAVE_DIR="/data/khoalq/checkpoints/exp14_stratified_m2f"

# ── Sanity Checks ──────────────────────────────────────────────────────────────
if [ ! -d "${DATA_ROOT}/Train" ]; then
    echo "ERROR: Train directory not found: ${DATA_ROOT}/Train"
    exit 1
fi
if [ ! -d "${DATA_ROOT}/Val" ]; then
    echo "ERROR: Val directory not found: ${DATA_ROOT}/Val"
    exit 1
fi

echo ""
echo "Dataset root       : $DATA_ROOT"
echo "Checkpoint out dir : $SAVE_DIR"
echo "Stratification     : Option 1 (Patient 40 -> Train, Patient 38 -> Val)"
echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Training Execution ─────────────────────────────────────────────────────────
# Exact hyperparameter parity with EXPERIMENT_5:
# batch_size=2, accum_steps=2 (effective batch_size=4), 60 epochs, AdamW (1e-5 backbone, 1e-4 head)
python -u "${REPO_ROOT}/experiments/EXPERIMENT_14/scripts/train.py" \
    --data_dir      "$DATA_ROOT" \
    --out_dir       "$SAVE_DIR"  \
    --epochs        60           \
    --batch_size    2            \
    --accum_steps   2            \
    --lr_head       1e-4         \
    --lr_backbone   1e-5         \
    --lambda_coord  5.0          \
    --lambda_vis    1.0          \
    --num_workers   8            \
    --eval_splits   both         \
    --device        cuda         \
    2>&1 | tee "/data/khoalq/logs/exp14_stratified_m2f_${SLURM_JOB_ID}.log"

# ── Verify Results Archive ─────────────────────────────────────────────────────
if [ ! -f "$SAVE_DIR/results.zip" ]; then
    echo "⚠️ Packaging results.zip manually as fallback..."
    python3 -c "
import os, zipfile
source_dir = '$SAVE_DIR'
zip_path = os.path.join(source_dir, 'results.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f != 'results.zip':
                p = os.path.join(root, f)
                zipf.write(p, os.path.relpath(p, source_dir))
print('Manually packaged results.zip:', os.path.getsize(zip_path)/(1024*1024), 'MB')
"
fi

echo ""
echo "=================================================================="
echo "✅ EXPERIMENT_14 COMPLETED AT $(date)"
echo "   Checkpoint in   : $SAVE_DIR/best_model.pth"
echo "   Results Archive : $SAVE_DIR/results.zip"
echo "=================================================================="
