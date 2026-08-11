# 🏥 Surgical AI — Laparoscopic Liver Landmark Detection & GeMap

This repository implements a **5-Layer Context Architecture** for Laparoscopic Liver Landmark Detection using Mask2Former pixel-wise baselines, topological constraint verification (TopoNet), and novel geometric & depth prompt learning (**Surgical GeMap**).

---

## 📁 Workspace Layout

```text
Surgical AI/
├── GEMINI.md                                  # Layer 0: Global Identity & Rules for Surgical AI
├── CONTEXT.md                                 # Layer 1: Workspace Master Router
├── README.md                                  # Workspace Navigation & Command Manual
├── _config/                                   # Layer 3: System Specifications & Standards
│   ├── execution_commands.md                  # Local macOS (MPS) & Kaggle Dual-GPU Run Manual
│   ├── environment_setup.md                   # Path Registries for Surgical Datasets & Depth Maps
│   └── code_conventions.md                    # PyTorch Standards & Clean Cloning Protocol
├── papers/                                    # Literature Knowledge Base
│   ├── inbox/                                 # 📥 Drop-zone for raw paper PDFs
│   ├── 2024_toponet_liver_landmarks/          # TopoNet MICCAI 2025 paper
│   ├── 2024_gemap_depth_prompt/               # GeMap ECCV 2024 paper
│   ├── 2022_mask2former/                      # Mask2Former paper
│   └── references/                            # AR Surgery & 3D-2D Registration Survey PDFs
├── repos/                                     # Immutable External Codebases
│   ├── GeMap/                                 # Clean original GeMap repository (stripped of .git)
│   └── TopoNet/                               # Clean original TopoNet reference codebase (stripped of .git)
├── shared/                                    # Reusable Utilities
│   └── utils/                                 # Common evaluation & image utilities
├── data/                                      # Data Isolation Layer (git-ignored)
│   ├── laparoscopic_liver/                    # Surgical video frames & landmark annotations
│   └── depth_maps/                            # Depth Anything v2 precomputed depth maps
├── checkpoints/                               # Pretrained weights & model checkpoints (git-ignored)
└── experiments/
    ├── CONTEXT.md                             # Stage Contract for Experiments
    ├── EXP_01_mask2former_pixelwise/          # Step 1: Pixel-wise Segmentation Baseline & Attention Maps
    │   └── sub_experiments/
    │       └── SUB_01_toponet_paper_reconstruction/ # TopoNet paper metric verification
    └── EXP_02_surgical_gemap/                 # Step 2: Main Novel Contribution — Surgical GeMap
```

---

## 🔬 Research Progression

1. **`EXP_01_mask2former_pixelwise`:**
   Pixel-wise landmark segmentation using Mask2Former and 9-layer attention visualization (`visualize_attention.py`). Contains `SUB_01` for verifying original TopoNet paper metrics (`train.py`, `test.py`).
2. **`EXP_02_surgical_gemap`:**
   Our main novel model integrating 3D geometric prompt learning from GeMap with depth prompt encoders (`depth_anything_v2`) and topological vector loss constraints (`vector_losses.py`, `cldice`).

---

## 💻 Quick Execution

```bash
# Surgical GeMap Training (Local macOS)
python experiments/EXP_02_surgical_gemap/scripts/train_surgical_gemap.py

# Cross-Model Benchmark Evaluation
python experiments/EXP_02_surgical_gemap/scripts/evaluate_all_models.py
```
