#!/bin/bash
# One-click environment setup script for EXP_12 (BCRNet Replication) on Kaggle / Colab
set -e

echo "🚀 [1/3] Installing Python dependencies..."
pip install -q \
    einops \
    scipy \
    shapely \
    timm \
    segment-anything \
    medpy \
    wandb \
    opencv-python-headless

echo "📥 [2/3] Checking / Downloading SAM ViT-B Weights..."
mkdir -p checkpoints
if [ ! -f "checkpoints/sam_vit_b_01ec64.pth" ]; then
    echo "Downloading sam_vit_b_01ec64.pth (375 MB)..."
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b_01ec64.pth
else
    echo "SAM ViT-B checkpoint already exists in checkpoints/."
fi

echo "⚙️ [3/3] Compiling adet._C MSDeformAttn CUDA extension..."
python experiments/EXP_12_bcrnet_replication/scripts/build_adet_ext.py

echo "✅ Environment setup complete! Ready to run prepare_data.py, train_bcrnet.py, or evaluate_bcrnet.py."
