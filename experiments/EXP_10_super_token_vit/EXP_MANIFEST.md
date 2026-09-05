# EXP_10 Manifest — Super-Token Geometric Vision Transformer

## 1. Abstract & Executive Hypothesis
**EXP_10 (Super-Token Geometric ViT)** addresses the two fundamental failure modes observed in EXP_09:
1. **The Discreteness & Dash Problem**: Treating the image as 1,024 isolated $16\times 16$ patches with independent 1x1 heads caused sub-pixel boundary seams, broken dashes (`-- -- --`), and multi-row parallel bursts.
2. **The Organ Retraction / Flip Problem (Patient 40)**: A patch head evaluated in isolation cannot discern whether a liver lobe has been rotated, flipped, or retracted by surgical graspers, mistaking metallic grasper edges for anatomical boundaries.

**Hypothesis**: By aggregating active patch tokens into dedicated **Landmark Super-Tokens** via multi-head cross-attention *before* curve decoding, and conditioning those queries on the global ViT `[CLS]` organ pose descriptor:
- The model learns global organ orientation and topological invariants.
- Anatomical junctions (e.g. Falciform Ligament meeting Inferior Margin at the umbilical notch) naturally share patch features through soft multi-membership.
- The global curve decoder outputs a single continuous $K=6$ composite spline in $[0, 1]^2$ image space, guaranteeing **0 dashes** and **0 parallel bursts**.
- Training with **Dual-Domain Supervision** (Vector Smooth L1 from JSON + Differentiable Soft Dice from `masks_gt/`) directly optimizes the official benchmark evaluation metric.

---

## 2. Mathematical Formulations (Plain Math)

### A. ViT Feature Extraction & Organ Pose Representation
Given RGB-D input $X$ of shape $(B, 4, 512, 512)$:
- Tokens = ViT_Backbone(X)
- cls_token = Tokens[:, 0, :] -> Shape: (B, D), D = 768
- patch_tokens = Tokens[:, 1:, :] -> Shape: (B, 1024, D), Grid: 32x32

The `cls_token` interacts with all 1,024 patches across all 12 transformer layers, encoding the global surgical scene (camera angle, grasper activity, liver lobe deformation/retraction).

### B. Pose-Conditioned Semantic Landmark Queries
Instead of anonymous DETR queries that fight and collapse, we define $C=4$ semantically fixed queries (Ridge, Silhouette, Falciform, Gallbladder):
- Q_base in R^(C x D)
- pose_delta = Linear(cls_token) in R^(B x 1 x D)
- Q_conditioned = Q_base + pose_delta in R^(B x C x D)

When the liver is flipped or retracted (as in Patient 40), `pose_delta` shifts the base query vector, steering the attention mechanism to follow the transformed anatomical orientation.

### C. Super-Token Multi-Head Cross-Attention (Pre-Prediction Merge)
- Attention_Scores = (Q_conditioned * W_q) * (patch_tokens * W_k)^T / sqrt(d)
- Attention_Weights = softmax(Attention_Scores, dim=-1) -> Shape: (B, H, C, 1024)
- Super_Tokens = Attention_Weights @ (patch_tokens * W_v) -> Shape: (B, C, D)
- Attention_Heatmaps = mean_over_heads(Attention_Weights) -> Reshaped to (B, C, 32, 32)

Patches at anatomical intersections (e.g. umbilical notch) yield high attention for multiple queries simultaneously, acting as topological anchors.

### D. Global Parametric Spline Decoding (K = 6 Control Points)
Each Super-Token is decoded into:
- exist_probs = sigmoid(Linear(Super_Tokens)) -> Shape: (B, C)
- ctrl_points = sigmoid(MLP(Super_Tokens)) -> Shape: (B, C, 6, 2) in [0, 1]^2

Evaluating the degree-5 Bernstein polynomial for $t \in [0, 1]$:
- B(t) = sum_{i=0}^5 comb(5, i) * (1 - t)^(5 - i) * t^i * P_i
- Sampled Trajectory: S = Basis_Matrix @ ctrl_points -> Shape: (B, C, 64, 2)
The resulting trajectory is mathematically unbroken across the entire liver.

### E. Differentiable Soft Line Rasterizer
For each grid coordinate $(u, v)$ on a $(128 \times 128)$ canvas:
- dist_sq = min_{n in [0, 63]} [ (u - S_n^x)^2 + (v - S_n^y)^2 ]
- Soft_Mask = exp( - dist_sq / (2 * sigma_norm^2) ) * exist_probs
Enables end-to-end backpropagation of Dice loss directly into Bézier control point coordinates.

### F. Dual-Domain Loss Objective
Total_Loss = lambda_attn * L_attn + lambda_vector * L_vector + lambda_dice * L_dice + lambda_exist * L_exist
- L_exist: BinaryCrossEntropy(exist_logits, target_exists)
- L_attn: Focal_BCE(Attention_Heatmaps, Downsampled_GT_Patch_Masks_32x32)
- L_vector: Smooth_L1(ctrl_points[active], target_ctrl_points[active], beta=0.02)
- L_dice: 1 - [ 2 * sum(Soft_Mask * GT_Mask) / (sum(Soft_Mask^2) + sum(GT_Mask^2) + eps) ]

---

## 3. Directory Layout
```
experiments/EXP_10_super_token_vit/
├── configs/
│   └── exp10_config.py             # Hyperparameters, paths, loss weights, spline order
├── models/
│   ├── spline_utils.py             # Bernstein polynomial basis, GPU evaluation, least-squares fit
│   ├── curve_decoder.py            # Existence head and K=6 control point MLP
│   ├── soft_rasterizer.py          # GPU-differentiable soft line rasterizer
│   ├── super_token_vit.py          # ViT Backbone + Pose Conditioning + Super-Token Cross-Attention
│   └── dual_domain_loss.py         # Focal BCE + Vector Smooth L1 + Soft Dice + Existence BCE
├── utils/
│   └── dataset_super_token.py      # RGB-D loader + JSON global spline fit + GT mask alignment
├── scripts/
│   ├── smoke_test_exp10.py         # Offline CPU/MPS unit test verifying forward/backward passes
│   ├── train_super_token_vit.py    # Training loop with AMP, Dual-Domain loss, checkpointing
│   ├── evaluate_super_token.py     # Official Dice, IoU, and control point error validation
│   └── visualize_val_predictions.py# 4-panel diagnostic visualizer (RGB, GT vs Pred, Attn, Depth)
├── EXP_MANIFEST.md                 # Architecture documentation, math formulations, changelog
└── Run_Commands.md                 # Exact execution commands for macOS and Kaggle CUDA
```

---

## 4. Hyperparameter Specifications

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| `image_size` | $512\times 512$ | Standard surgical resolution matching L3D benchmark |
| `patch_size` | $16\times 16$ | Grid size: $32\times 32 = 1,024$ patches |
| `num_ctrl_points` | $K = 6$ | 1 start, 4 interior guides, 1 end point (degree-5 curve) |
| `backbone_name` | `vit_base_patch16_224` | 86M params, $D=768$, 12 layers, 12 attention heads |
| `cross_attn_heads` | 8 | Multi-head attention for Super-Token spatial pooling |
| `render_size` | $128\times 128$ | Lightweight canvas for differentiable Dice loss (<300MB VRAM) |
| `lambda_attn` | 2.0 | Supervised attention alignment to ground-truth patch mask |
| `lambda_vector` | 5.0 | Smooth L1 coordinate guide from JSON polylines |
| `lambda_dice` | 5.0 | Direct optimization of official benchmark Dice score |
| `lambda_exist` | 1.5 | Landmark visibility classification |
| `learning_rate` | 1e-4 | Head learning rate with 0.1x backbone multiplier (1e-5) |
| `batch_size` | 16 | Optimized for single 16GB GPU (T4 / P100 / RTX 3090) |
