# Execution Manual: EXP_01 — Mask2Former Pixel-Wise Baseline

Copy-pasteable execution commands for local (macOS) and cloud (Kaggle) runs of Mask2Former pixel-wise baseline and attention visualizations.

---

## 1. Setup & Environment Activation

```bash
# Local macOS setup
conda activate TopoNet

# Kaggle Session setup
cd /kaggle/working
git clone <your_repository_url>
cd Surgical_AI
```

---

## 2. Dataset Preparation & Precomputations

### Precompute Depth Anything V2 (Train, Val, Test)
```bash
python shared/utils/precompute_depth_anything.py \
  --data_path /kaggle/working/L3D \
  --batch_size 8 \
  --model_id depth-anything/Depth-Anything-V2-Base-hf
```

---

## 3. Model Execution Commands

### Mask2Former Baseline Ablation Run
```bash
python experiments/EXP_01_mask2former_pixelwise/scripts/ablation_mask2former.py
```

### Attention Map Visualizer (9 Layers)
```bash
python experiments/EXP_01_mask2former_pixelwise/scripts/visualize_attention.py
```

### Validation Failure Mode Analysis (Ranked Worst to Best)
```bash
python experiments/EXP_01_mask2former_pixelwise/scripts/evaluate_worst_cases.py \
  --mode Swin_MaskedAttn \
  --data_path /kaggle/working/L3D \
  --output_dir /kaggle/working/results_ablation/worst_cases \
  --top_k_save 15
```

---

## 3. Sub-Experiment Execution Commands (TopoNet Paper Metric Reconstruction)

```bash
python experiments/EXP_01_mask2former_pixelwise/sub_experiments/SUB_01_toponet_paper_reconstruction/scripts/train.py --config experiments/EXP_02_surgical_gemap/configs/toponet.yaml
```
