#!/bin/bash
#SBATCH --job-name=exp5_junction_m2f
#SBATCH --output=experiments/EXPERIMENT_5/logs/train_%j.log
#SBATCH --error=experiments/EXPERIMENT_5/logs/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

mkdir -p experiments/EXPERIMENT_5/logs
mkdir -p /data/khoalq/checkpoints/exp5_junction_m2f

source activate surgical_ai
cd /data/khoalq/surgical_ai

python3 -u experiments/EXPERIMENT_5/scripts/train.py \
    --epochs 60 \
    --batch_size 2 \
    --accum_steps 2 \
    --lr_head 1e-4 \
    --lr_backbone 1e-5 \
    --lambda_coord 5.0 \
    --lambda_vis 1.0 \
    --out_dir /data/khoalq/checkpoints/exp5_junction_m2f \
    --device cuda
