# EXP_03: Pixel-Guided Vector Transformer (Surgical-GeMap v2)

## Status: Active 🚀

### Overview
Surgical-GeMap v2 unifies **Dense Pixel-Wise Mask Supervision** with **Vectorized Polyline Query Decoders**. It addresses the sparse point gradient bottleneck of pure vector models (`pts_loss` plateauing at 0.137) by co-supervising the FPN backbone with 2D Pixel BCE + Dice loss while refining ordered 1D polyline control points using Geometry-Decoupled Attention.

---

### Core Architectural Features

1. **Auxiliary FPN Pixel Segmentation Head**: Outputs 4-channel dense mask at $1024 \times 1024$, supervised by Pixel BCE + Dice loss.
2. **Heatmap-Guided Reference Point Proposals**: Peak spatial activations from pixel heatmaps initialize query reference points.
3. **Point-Sampled `grid_sample` & 2D Sinusoidal `query_pos`**: High-resolution FPN feature extraction at exact point locations $(x_k, y_k)$.
4. **Dual Evaluation Protocol**: Evaluates both Pixel Metrics (Dice, IoU, ASSD) and Vector Metrics (Chamfer, Fréchet).

---

### Directory Layout

```
experiments/EXP_03_pixel_guided_vector_transformer/
├── EXP_MANIFEST.md
├── Run_Commands.md
├── models/
│   ├── surgical_gemap_v2.py
│   └── vector_losses.py
└── scripts/
    └── train_surgical_gemap_v2.py
```
