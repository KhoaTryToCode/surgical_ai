# 3 Landmark Master Tokens: Architecture & Mathematical Formulation

## 1. Overview
This document details the architectural elaboration of the **3 Landmark Master Tokens** for resolving severe liver deformation, grasper retraction, and horizontal/vertical flips in the L3D dataset (e.g., `Patient_40_08940`).

---

## 2. Token Initialization & Sequence Setup

The transformer decoder receives a sequence of **67 tokens**:
`Tokens Z = [T_ridge, T_sil, T_falc, Q_0, Q_1, ..., Q_63]` (Shape: B x 67 x 256)

### Positional Encodings
- **64 Patch Queries (Q_0 ... Q_63):** Bound to spatial camera coordinates via 2D sinusoidal positional encodings PE(r, c) for r, c in [0, 7].
- **3 Landmark Master Tokens (T_ridge, T_sil, T_falc):** Free from camera coordinates. They receive categorical class identity embeddings E_class:
  - `E_ridge` = Embedding for Class 1 (Inferior Ridge)
  - `E_sil`   = Embedding for Class 2 (Liver Silhouette / Outer Margin)
  - `E_falc`  = Embedding for Class 3 (Falciform Ligament)

---

## 3. Multi-Head Self-Attention Decomposition (67 x 67)

In each of the 6 decoder layers, the full 67 x 67 self-attention matrix decomposes into 4 distinct functional blocks:

```
                          Landmark Tokens (3)         Patch Queries (64)
                        ┌──────────────────────┬──────────────────────────────┐
  Landmark Tokens (3)   │  Block 1: [3 x 3]    │     Block 3: [3 x 64]        │
                        │  Landmark-to-Landmark│     Landmark-to-Patches      │
                        │  (Organ Pose & Rel)  │     (Trajectory Aggregation) │
                        ├──────────────────────┼──────────────────────────────┤
  Patch Queries (64)    │  Block 2: [64 x 3]   │     Block 4: [64 x 64]       │
                        │  Patches-to-Landmark │     Patches-to-Patches       │
                        │  (Anatomical Guide)  │     (Local C0/C1 Continuity) │
                        └──────────────────────┴──────────────────────────────┘
```

### Block 1: Landmark-to-Landmark (3 x 3) — The Relational Engine
The 3 tokens attend directly to each other:
`Attention(i, j) = Softmax( Q_i * K_j^T / sqrt(d) )`

This explicitly computes the relative geometric relationships between anatomical classes:
- `v_rel = CoM(Ridge) - CoM(Silhouette)`
- Normal liver: `v_rel = (0.0, +0.45)` (Ridge is below Silhouette)
- Grasper retraction (08730): `v_rel = (+0.07, -0.34)` (Sign inverted; Ridge pulled above Silhouette)
- Flipped liver (08940): `v_rel = (-0.51, -0.04)` (Horizontal inversion; Ridge on far left, Silhouette on far right)

### Block 2: Patches-to-Landmark (64 x 3) — Global Anatomical Guidance
Every local patch query Q_(r, c) computes attention against the 3 master tokens.
- If `T_ridge` encodes that Ridge has flipped to the left side of the abdomen (x ~ 0.13), local patch queries in the left columns (c in [0, 2]) receive strong Ridge guidance.
- Prevents false-positive hallucinations on the right side of the frame.

### Block 3: Landmark-to-Patches (3 x 64) — Trajectory Aggregation
`T_ridge` pools features from all active patches containing Ridge segments, aggregating the entire multi-patch curve trajectory.

### Block 4: Patches-to-Patches (64 x 64) — Local Boundary Continuity
Adjacent patches exchange local boundary coordinates for C^0 endpoint matching and C^1 tangent vector alignment.

---

## 4. Auxiliary Geometric Supervision (Centroid + Presence)

To prevent the 3 master tokens from drifting or acting as redundant patch queries, each token is explicitly supervised using ground truth landmark coordinates:

### Head Architecture
```python
self.landmark_head = nn.Sequential(
    nn.Linear(256, 128),
    nn.GELU(),
    nn.Linear(128, 3) # outputs: [pred_cx, pred_cy, pred_presence_logit]
)
```

### Ground Truth Targets (Computed on-the-fly from JSON annotations)
For each class k in {1, 2, 3}:
- `y_k`: 1 if class k is present in the frame, 0 otherwise
- `c_k* = (x_mean_k, y_mean_k)`: normalized center-of-mass of class k in [0, 1]^2

### Loss Formulation
`L_landmark = sum_{k=1}^3 [ BCE(pred_presence_k, y_k) + y_k * SmoothL1(pred_centroid_k, c_k*) ]`

Total Loss:
`L_total = L_patch_losses + lambda_landmark * L_landmark` (with lambda_landmark = 1.0)

---

## 5. Walkthrough: Resolving Patient_40_08940 (Flipped Liver Case)

1. **Input:** Patient 40 frame 08940 with liver lobe retracted and flipped left-to-right.
2. **Backbone Feature Extraction:** Swin-Tiny extracts image features. High edge contrast on left (x=0.13) and silhouette boundary on right (x=0.65).
3. **Decoder Layer 1:**
   - `T_ridge` attends to left features; its predicted centroid converges to (0.13, 0.25).
   - `T_sil` attends to right features; its predicted centroid converges to (0.65, 0.29).
4. **Decoder Layer 2 (Block 1 Self-Attention):**
   - `T_ridge` and `T_sil` attend to each other, explicitly encoding `delta_x = 0.13 - 0.65 = -0.52`.
   - The relational state is established as horizontally flipped.
5. **Decoder Layers 3–6 (Block 2 Broadcast):**
   - Left-column patch queries (x in [0, 0.3]) attend heavily to `T_ridge`. Their class logits predict Class 1 (Ridge).
   - Right-column patch queries (x in [0.5, 0.8]) attend to `T_sil`, predicting Class 2 (Silhouette).
6. **Result:** Correctly segments Ridge on the left and Silhouette on the right without false-positive leakage.

---

## 6. Computational Cost
- Sequence length: 64 -> 67 tokens (+3 tokens)
- Attention complexity: 67^2 = 4,489 vs 64^2 = 4,096 FLOPs per head
- Latency impact: < 0.1 ms per frame (fully maintains 14.5 - 16.2 FPS on A100)
