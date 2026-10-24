# Execution Commands & Environment Cheat-Sheet (Layer 3)

Master reference of standard CLI commands for running Mask2Former pixel-wise baselines and Surgical GeMap landmark detection locally (macOS MPS/CPU) and on Kaggle (Linux CUDA/GPU).

---

## 1. Setup & Environment Activation

### Local Development (macOS)
```bash
conda activate TopoNet
```

### Kaggle Notebook Setup (Linux Session)
```bash
cd /kaggle/working
git clone <your_repository_url>
cd Surgical_AI
bash experiments/EXP_02_surgical_gemap/scripts/setup_kaggle.sh
```

---

## 2. Execution Commands for EXP_01 (Mask2Former Baseline & Topo Verification)

### Local Mask2Former Ablation Run
```bash
python experiments/EXP_01_mask2former_pixelwise/scripts/ablation_mask2former.py
```

### Attention Visualizer Run
```bash
python experiments/EXP_01_mask2former_pixelwise/scripts/visualize_attention.py
```

### TopoNet Paper Metric Reconstruction Run (Sub-Experiment 01)
```bash
python experiments/EXP_01_mask2former_pixelwise/sub_experiments/SUB_01_toponet_paper_reconstruction/scripts/train.py --config experiments/EXP_02_surgical_gemap/configs/toponet.yaml
```

---

## 3. Execution Commands for EXP_02 (Surgical GeMap — Main Contribution)

### Local macOS Execution (Surgical GeMap Training)
```bash
python experiments/EXP_02_surgical_gemap/scripts/train_surgical_gemap.py
```

### Kaggle Single GPU Run (CUDA)
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/EXP_02_surgical_gemap/scripts/train_surgical_gemap.py
```

### Kaggle Multi-GPU Run (PyTorch DDP — Dual T4 GPUs)
```bash
torchrun --nproc_per_node=2 experiments/EXP_02_surgical_gemap/scripts/train_surgical_gemap.py
```

### Cross-Model Benchmark Evaluation
```bash
python experiments/EXP_02_surgical_gemap/scripts/evaluate_all_models.py
```
