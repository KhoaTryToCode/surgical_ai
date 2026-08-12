# EXP_04: Surgical-BeMapTR (Pure Vectorized Landmark Segmentation)

## Status: Active 🚀

### Overview
Surgical-BeMapTR is a pure vectorized landmark segmentation architecture that unifies:
1. **BeMapNet (CVPR 2023)**: Piecewise Bézier Curve parameterization $\langle k=3, n=3 \rangle$ (10 control handles), Bernstein matrix curve restoration ($P = B \cdot C$), and 3-level Point-Curve-Region (PCR) progressive loss.
2. **MapTRv2 (TPAMI 2024)**: Permutation-equivalent bipartite matching (orientation invariant), hierarchical queries ($q_{ij} = q_i^{\text{ins}} + q_j^{\text{pt}}$), geometry-decoupled attention (GDA), and 2D point positional encodings (`query_pos`).

---

### Core Architectural Features

1. **Piecewise Bézier Parameterization**: 10 control points per line generate 20 ultra-smooth 1D curve sample points via $P = B \cdot C$.
2. **Permutation-Equivalent Matching**: Evaluates both forward and reverse orientations to eliminate direction ambiguity.
3. **Point-Curve-Region (PCR) Loss**: Point L1 + Bernstein Curve L1 + Edge Direction Cosine + Focal Loss.
4. **Pure Vector Pipeline**: No secondary FPN pixel head required.

---

### Directory Layout

```
experiments/EXP_04_surgical_bemaptr_pure_vector/
├── EXP_MANIFEST.md
├── Run_Commands.md
├── models/
│   ├── surgical_bemaptr.py
│   └── vector_losses.py
└── scripts/
    └── train_bemaptr.py
```
