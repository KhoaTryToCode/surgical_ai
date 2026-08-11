# Execution Manual: EXP_02 — Surgical GeMap

Copy-pasteable execution commands for local (macOS) and cloud (Kaggle) runs of Surgical GeMap.

---

## 1. Setup & Environment Activation

```bash
# Local macOS setup
conda activate TopoNet

# Kaggle Session setup
cd /kaggle/working
git clone <your_repository_url>
cd Surgical_AI
bash experiments/EXP_02_surgical_gemap/scripts/setup_kaggle.sh
```

---

## 2. Model Training Execution Commands

### Local macOS Run (MPS / CPU)
```bash
python experiments/EXP_02_surgical_gemap/scripts/train_surgical_gemap.py
```

### Kaggle Single GPU Run (CUDA)
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/EXP_02_surgical_gemap/scripts/train_surgical_gemap.py
```

### Kaggle Multi-GPU Run (PyTorch DDP - Dual T4 GPUs)
```bash
torchrun --nproc_per_node=2 experiments/EXP_02_surgical_gemap/scripts/train_surgical_gemap.py
```

---

## 3. Evaluation & Benchmark Analysis

### Cross-Model Benchmark Evaluation
```bash
python experiments/EXP_02_surgical_gemap/scripts/evaluate_all_models.py
```

### Statistical Result Analysis
```bash
python experiments/EXP_02_surgical_gemap/scripts/analyze_results.py
```
