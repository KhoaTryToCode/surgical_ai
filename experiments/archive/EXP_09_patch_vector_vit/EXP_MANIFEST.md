# EXP_09: Patch-Level Bézier Vector Vision Transformer (Patch-Bézier ViT)

## Identity & Executive Summary
EXP_09 is a feed-forward, spatially anchored **Patch-Level Bézier Vector Vision Transformer** for laparoscopic surgical liver landmark detection. It eliminates the bipartite Hungarian matching collapse of DETR queries (EXP_06) and the autoregressive drift of sequential drawing models (EXP_08) by partitioning the $512 \times 512$ image into a $32 \times 32$ grid of non-overlapping $16 \times 16$ px patches.

Each ViT patch token concurrently predicts:
1. **Landmark Presence & Classification:** Background ($0$), Ridge ($1$), Silhouette ($2$), Ligament ($3$), or Gallbladder ($4$).
2. **Parametric Cubic Bézier Curve:** 4 control points $(P_0, P_1, P_2, P_3) \in [0, 1]^2$ representing the exact local curve trajectory traversing that patch.

Finally, all active patch Bézier curves are merged into a high-resolution, sub-pixel accurate anatomical landmark image via **Global Coordinate Shift + Anti-Aliased Vector Rasterization**.

---

## Architectural Key Components

```
Input Image (3, 512, 512)
        ↓
Patch Partition & Linear Projection (P=16) → 1024 tokens of dim D=192
        ↓
Vision Transformer Backbone (timm vit_tiny_patch16_224)
        ↓
Patch Tokens: Z ∈ R^{1024 × 192}
        ↓
    ┌───────────────────────────┴───────────────────────────┐
    ▼                                                       ▼
Class / Presence Head                                   Bézier Control Head
Linear(D, num_classes + 1)                              MLP(D → 256 → 8) + Sigmoid
Logits: (B, 1024, 5)                                    Control Points: (B, 1024, 4, 2)
[Background, Ridge, Silhouette, Ligament]               [P0, P1, P2, P3] in [0, 1]^2
```

### 1. Ground Truth Extraction via Closed-Form Least-Squares
- Raw landmark polylines are resampled at $\sim 8\text{ px}$ intervals using arc-length parameterized cubic splines (`resample_polyline_by_arclength`).
- Points inside each patch $(r, c)$ are normalized to local patch space $[0, 1]^2$.
- Endpoints $P_0$ and $P_3$ are anchored to the entry and exit points. Intermediate control handles $(P_1, P_2)$ are computed deterministically via closed-form $2 \times 2$ linear least-squares.

### 2. Multi-Task Loss Suite
- **Focal Classification Loss ($L_{cls}$):** Addresses extreme class imbalance ($\sim 97\%$ background patches).
- **Control Point Smooth L1 ($L_{ctrl}$):** Penalizes error on $(P_0, P_1, P_2, P_3)$ exclusively on active landmark patches.
- **Sampled Curve Chamfer L1 ($L_{sample}$):** Differentiable Bernstein evaluation of $N_s=10$ points along the curve.
- **Tangent Alignment ($L_{tan}$):** Cosine alignment on entry tangent $(P_1 - P_0)$ and exit tangent $(P_3 - P_2)$.
- **Boundary Continuity ($L_{cont}$):** Penalizes endpoint gap $\|P_3^{(r, c)} - P_0^{(r', c')}\|_2$ between adjacent active patches.

Total Loss:
$L_{total} = \lambda_{cls} L_{cls} + \lambda_{ctrl} L_{ctrl} + \lambda_{sample} L_{sample} + \lambda_{tan} L_{tan} + \lambda_{cont} L_{cont}$

### 3. Merging Mechanism
- **Global Coordinate Shift:**
  $P_{global, j} = (c \cdot P, \ r \cdot P) + P_{local, j} \cdot P$
- **Anti-Aliased Rasterization:** Evaluates Bernstein polynomials along $t \in [0, 1]$ and renders continuous anti-aliased Bézier strokes with $2\text{ px}$ thickness onto an image canvas.

---

## Directory Structure
- `configs/exp09_config.py`: Hyperparameters, patch grid specs, and Kaggle/local path resolvers.
- `models/bezier_utils.py`: Arc-length cubic spline resampler, least-squares Bézier fitter, and differentiable Bernstein sampling.
- `models/patch_vector_vit.py`: Vision Transformer architecture with dual prediction heads.
- `models/patch_losses.py`: Multi-task loss suite (Focal + Control L1 + Sample L1 + Tangent + Continuity).
- `models/patch_merger.py`: Merging engine (Global Coordinate Shift + Anti-Aliased Vector Rasterizer).
- `utils/dataset_patch_vit.py`: Surgical dataset reader with automated Bézier target generation.
- `scripts/visualize_smoke_test_patch_vit.py`: 4-panel diagnostic visual verification script.
- `scripts/train_patch_vit.py`: Full training loop with WandB logging and checkpointing.
- `scripts/evaluate_patch_vit.py`: Quantitative metrics (Precision, Recall, Bézier Error in pixels, Merged Image Dice).
- `EXP_MANIFEST.md`: This manifest.
- `Run_Commands.md`: Step-by-step execution instructions for macOS and Kaggle.
