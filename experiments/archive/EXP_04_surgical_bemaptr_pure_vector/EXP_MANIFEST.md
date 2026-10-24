# EXP_04: Surgical-BeMapTR v2 (Pure Vectorized Landmark Segmentation)

## Status: Active & Validated 🚀

### Overview
Surgical-BeMapTR v2 is a pure vectorized landmark segmentation architecture that unifies:
1. **BeMapNet (CVPR 2023)**: Piecewise Bézier Curve parameterization <k=3, n=3> (10 control handles), Bernstein matrix curve restoration (P = B * C), 3-level Point-Curve-Region (PCR) progressive loss, and Spatial Centroid Coordinate Head (coords_encoder + einsum + GAP).
2. **MapTRv2 (TPAMI 2024)**: Permutation-equivalent bipartite matching (orientation invariant), hierarchical queries (q_ij = q_i^ins + q_j^pt), geometry-decoupled attention (GDA), 2D point positional encodings (`query_pos`), and Deformable Cross-Attention (`grid_sample` reference-point-guided sampling across multi-scale FPN feature maps).

---

### Core Architectural Features (v2)

1. **Spatial Centroid Coordinate Head**: Replaces MLP vector coordinate regression with 2D spatial voting over a coordinate feature map. Calculates centroids via `einsum` dot product and global average pooling (GAP).
2. **PyTorch-Native Deformable Cross-Attention**: Replaces global vanilla attention with reference-point-guided sampling (`F.grid_sample`) across 3 FPN scale levels (strides 4, 8, 16).
3. **Multi-Scale FPN Features**: Pixel decoder provides multi-scale feature maps [P2, P3, P4] for localized multi-resolution feature extraction.
4. **Piecewise Bézier Parameterization**: 10 control points per line generate 20 ultra-smooth 1D curve sample points via P = B * C.
5. **Permutation-Equivalent Matching**: Evaluates both forward and reverse orientations to eliminate direction ambiguity.
6. **Point-Curve-Region (PCR) Loss**: Point L1 + Bernstein Curve L1 + Edge Direction Cosine + Focal Loss.

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
    ├── train_bemaptr.py
    └── visualize_predictions.py
```

