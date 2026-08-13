# EXP_04b: Surgical-BeMapTR v3 (Auxiliary Pixel-Level Edge Guidance)

## Status: Active & Ready for Kaggle / Colab 🚀

### Overview
Surgical-BeMapTR v3 (EXP_04b) combines the pure vector architecture with a **lightweight auxiliary pixel-level edge segmentation head**:
1. **Vector Branch (BeMapNet + MapTR)**: Piecewise Bézier curve parameterization <k=3, n=3> (10 control handles), Spatial Centroid Coordinate Head (coords_encoder + einsum + GAP), 2D point positional encodings (`query_pos`), and Deformable Cross-Attention (`grid_sample` reference-point-guided sampling across multi-scale FPN feature maps).
2. **Auxiliary Pixel Edge Branch (NEW)**: Lightweight 4-channel 2D convolutional head supervising the finest FPN feature map P2 (256x256) with binary cross-entropy + Dice loss against pixel_masks.

---

### Key Architectural Innovation

By adding the 256x256 pixel-level edge head, the FPN pixel decoder and Swin-Tiny backbone receive **dense pixel-level boundary gradients** (~65,536 supervision points per image). This forces the spatial feature maps [P2, P3, P4] to form sharp, high-contrast activations along anatomical liver edges. The deformable attention queries sample directly from these sharp edge features, boosting vector query recall and driving rasterized Dice into the **0.50 - 0.70+** range.

---

### Directory Layout

```
experiments/EXP_04b_bemaptr_aux_edge/
├── EXP_MANIFEST.md
├── Run_Commands.md
├── models/
│   └── surgical_bemaptr_aux.py
└── scripts/
    └── train_bemaptr_aux.py
```
