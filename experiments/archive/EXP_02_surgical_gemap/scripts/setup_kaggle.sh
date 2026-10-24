#!/bin/bash
# One-click environment & dependency setup script for Kaggle / Colab
set -e

echo "Installing requirements..."
pip install pybind11 pytorch-metric-learning medpy surface-distance timm einops

if [ ! -d "betti_match" ]; then
    echo "Cloning and building Betti-Matching-3D..."
    git clone https://github.com/nstucki/Betti-Matching-3D.git betti_match
    cd betti_match && mkdir -p build && cd build
    cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. && make
    cd ../..
    touch betti_match/__init__.py
    mkdir -p betti_match/Betti_Matching
    touch betti_match/Betti_Matching/__init__.py
    cp betti_match/build/betti_matching*.so betti_match/Betti_Matching/ 2>/dev/null || true
    echo "from betti_match.Betti_Matching import betti_matching" > betti_match/Betti_Matching/betti_build.py
fi

echo "Setup complete! Ready to run train.py or test.py."
