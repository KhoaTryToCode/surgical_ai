# Experimental Suite & Research Progression

This directory organizes all experimental models, training pipelines, ablation studies, and evaluation logs developed throughout this project.

---

## 📁 Directory Structure

```text
experiments/
├── static/                     # 16 Systematic Single-Frame Experiments
│   ├── EXPERIMENT_1/           # TopoNet replication & systematic ablations
│   ├── EXPERIMENT_2/           # Mask2Former Swin-Tiny baseline & query ablations
│   ├── EXPERIMENT_3/           # Patch-level Bézier curve formulation
│   ├── EXPERIMENT_4/           # Landmark Bézier curve formulation
│   ├── EXPERIMENT_5/           # Diagnostic analysis & Patient 40 failure audit
│   ├── ...
│   ├── EXPERIMENT_14/          # Pose-Stratified Junction-Steered Mask2Former
│   ├── EXPERIMENT_15_.../      # Temporal Prototype Transformer (T=3 causal keyframes)
│   └── EXPERIMENT_16_.../      # Pose-Stratified Temporal Transformer (SOTA)
│
├── temporal/                   # Video Landmark Segmentation Suite (MinVIS)
│   ├── train.py                # Universal 5-fold training entrypoint
│   ├── evaluate.py             # Validation and frozen test evaluation CLI
│   ├── aggregate.py            # Automated leaderboard aggregation
│   ├── core/                   # Dataset loader, trainer engine, metrics
│   ├── models/                 # MinVIS Mask2Former + Hungarian video tracker
│   ├── configs/                # Experiment configuration YAMLs
│   ├── notebooks/              # Lightweight Kaggle Git Runner notebook
│   └── scripts/                # Notebook generator and helper tools
│
└── archive/                    # Archived Early Prototypes (EXP_01 to EXP_13)
    ├── EXP_01_mask2former_pixelwise/
    ├── EXP_02_surgical_gemap/
    └── ...
```

---

## 🔬 Progression & Master Results

### 1. Official L3D Frozen Test Benchmark (109 Frames)

All models are evaluated on the exact 109 frozen test frames under native $1024 \times 1024$ resolution with class mutual exclusivity:

| Model | Architecture / Method | Macro Dice (%) $\uparrow$ | Micro Dice (%) $\uparrow$ | ASSD (px) $\downarrow$ | Ridge (%) | Silhouette (%) | Ligament (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TopoNet** (MICCAI 2025) | ResNet-50 + BTF + clDice + Betti | 59.79 | 65.19 | 28.07 | 64.30 | 72.10 | 42.97 |
| **BCRNet** (MICCAI 2025) | Bézier Curve Representation | 69.57 | --- | 43.55 | 68.12 | 74.30 | 66.29 |
| **A2ONet** (MICCAI 2026) | Adaptive Illumination + Curve Opt | 65.31 | --- | 49.89 | 63.80 | 71.20 | 60.93 |
| **Mask2Former Baseline** | Swin-Tiny + Masked Query Attn | 65.73 | --- | 28.43 | 66.81 | 71.45 | 58.93 |
| **Mask2Former Bézier** | Swin-Tiny + Bézier Control Points | 55.74 | --- | 38.46 | 57.10 | 62.40 | 47.72 |
| **Junction-Steered M2F** | 4-Junction Biomechanical Steering | 66.86 | --- | 22.64 | 67.55 | 73.12 | 59.91 |
| **Temporal M2F (T=3)** | Past-Prototype Cross-Attention | 65.36 | 68.18 | 26.86 | 66.22 | 70.04 | 59.83 |
| **Stratified Temporal M2F** | **Balanced Pose + Temporal Prototypes** | **71.92** | **74.31** | **15.37** | **72.53** | **76.66** | **66.58** |

### 2. Validation Benchmark (122 Frames — Patient 40 Dominated)

| Model | Macro Dice (%) | ASSD (px) | Key Finding |
|:---|:---:|:---:|:---|
| **TopoNet Baseline** | 54.64 | 49.64 | Vulnerable to topological discontinuity under surgical traction |
| **TopoNet Full** | 59.79 | 29.27 | Topological loss improves continuity but adds high latency |
| **Mask2Former Baseline** | 66.56 | 25.69 | Fast convergence; collapses on Patient 40 inverted frames (7.56% DSC) |
| **Mask2Former Single Scale**| 65.35 | 35.77 | Less multiscale spatial bias, but lower global accuracy |
| **Junction-Steered M2F** | **68.13** | **19.54** | Queries guided by biological junctions recover traction states |

---

## 💡 Key Architectural Insights

1. **Topological Loss vs. Query Attention**: TopoNet enforces continuity through persistent homology ($\mathcal{L}_{\text{per}}$) and skeleton loss ($\mathcal{L}_{\text{cl}}$), which requires non-parallel CPU computations. Mask2Former achieves superior spatial awareness and faster inference (10.6 FPS vs. 11.6 FPS) through masked cross-attention.
2. **The Limit of Static Queries**: Both CNN and transformer static models retain spatial prior inertia. When mechanical graspers lift the anterior ridge into the superior canvas, static models hallucinate boundaries at the bottom.
3. **Biological Junction Anchoring**: Injecting 4 anatomical junction coordinates ($J_{\text{top}}, J_{\text{bottom}}, J_{\text{lat\_r}}, J_{\text{lat\_l}}$) as query steering signals provides substantial boundary distance reductions (ASSD dropped from 28.43 px to 22.64 px).
4. **Temporal Context as Spatial Anchor**: Temporal prototypes condition the current prediction on observed preceding trajectories, smoothing landmark tracking and establishing a new state of the art when unified with pose stratification.
