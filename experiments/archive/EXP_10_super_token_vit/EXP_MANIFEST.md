# EXP_10 Manifest — Macro-Patch Geometric Vision Transformer (Way A)

## 1. Abstract & Executive Hypothesis
**EXP_10 (Macro-Patch Geometric ViT — Way A)** establishes an optimal intermediate spatial scale for laparoscopic liver landmark detection, resolving both:
1. **The Micro-Discreteness of EXP_09**: $16\times 16$ px tiles were too microscopic (3% of screen width), forcing a 400px landmark across 45 tile borders, causing broken sub-pixel dashes (`-- -- --`) and parallel row bursts.
2. **The Global Collapse of Early EXP_10**: Collapsing the entire image into 1 single vector destroyed spatial ordering, causing the curve decoder to fall back to a static template in the top-right corner.

**Hypothesis**: By extracting fine-grained $16\times 16$ features via a pretrained ViT backbone and hierarchically merging neighboring $4\times 4$ micro-tokens into an **$8\times 8$ grid of $64\times 64$ pixel Macro-Patches**:
- Each macro-patch covers 12.5% of the liver, providing ample canvas for smooth, continuous cubic Bézier curves without cutting corners.
- An anatomical line crosses only **5 to 7 macro-patches** (an 80%+ reduction in boundary seams).
- The 64 macro-tokens run through an **Inter-Macro Relational Transformer** (all-to-all self-attention) to perceive global organ pose, deformation, and flipping (Patient 40).
- Local coordinates are scaled and shifted by $(c\cdot 64, r\cdot 64)$, **mathematically guaranteeing 100% spatial grounding with zero chance of teleportation**.
- An adjacent **Endpoint Continuity Loss** forces curves to snap together seamlessly across macro-patch boundaries.

---

## 2. Mathematical Formulations (Plain Math)

### A. ViT Feature Extraction
Given RGB-D input $X$ of shape $(B, 4, 512, 512)$:
- Tokens = ViT_Backbone(X)
- micro_tokens = Tokens[:, 1:, :] -> Shape: (B, 1024, D), Grid: 32x32 of 16px patches, D = 768

### B. Spatial Macro-Patch Merge (4x4 Grouping)
Reshape micro-tokens to 2D spatial feature map:
- micro_spatial = reshape(micro_tokens) -> Shape: (B, D, 32, 32)
- macro_spatial = Conv2d(kernel_size=4, stride=4)(micro_spatial) -> Shape: (B, D, 8, 8)
- macro_tokens = reshape(macro_spatial) -> Shape: (B, 64, D)
Each macro-token aggregates visual details of 16 constituent micro-patches covering a 64x64 px territory.

### C. Dynamic Soft Positional Encoding Generator (PEG / CPVT)
Instead of rigid, static lookup tables (nn.Parameter pos_embed) that clash when the organ is inverted or retracted (Patient 40):
- pos_signal = Conv2d(groups=D, kernel_size=3, padding=1)(macro_spatial)
- macro_tokens = macro_tokens + pos_signal
Because the positional signal is derived via zero-padded depthwise convolutions over the 2D feature map, the coordinate frame dynamically deforms, translates, and rotates WITH the anatomical liver tissue.

### D. Inter-Macro Relational Attention (Multi-Stage PEG)
The 64 macro-tokens attend to each other across the entire organ:
- macro_tokens = PEG_1(macro_tokens)
- macro_tokens = TransformerEncoderLayer_1(macro_tokens)
- macro_tokens = PEG_2(macro_tokens)
- macro_tokens = TransformerEncoderLayer_2(macro_tokens)
- macro_context = LayerNorm(macro_tokens) -> Shape: (B, 64, D)
Macro-patch (1, 2) at the top-left directly exchanges geometric signals with macro-patch (7, 6) at the bottom-right. When surgical graspers rotate or retract the liver lobe (Patient 40), the dynamic PEG signals and relational attention weights adjust to the transformed organ frame.

### E. Per-Macro Dual Prediction Heads
For each macro-patch (r, c) where r in [0, 7], c in [0, 7]:
- macro_logits = Linear(D -> 256 -> 5) -> Shape: (B, 8, 8, 5)  [0: BG, 1: Ridge, 2: Silhouette, 3: Ligament, 4: Gallbladder]
- macro_beziers = Sigmoid(MLP(D -> 256 -> 8)) -> Shape: (B, 8, 8, 4, 2)  [P0, P1, P2, P3] in [0, 1]^2

### F. Spatial Coordinate Anchoring (Anti-Teleport Guarantee)
Global coordinates on the 512x512 image canvas:
- P_global_x = c * 64 + P_local_x * 64
- P_global_y = r * 64 + P_local_y * 64
Because P_local in [0, 1], the curve predicted by macro-patch (r, c) is strictly locked within [c*64, (c+1)*64] x [r*64, (r+1)*64].

### G. Multi-Task Macro Loss Objective
Total_Loss = lambda_cls * L_cls + lambda_ctrl * L_ctrl + lambda_sample * L_sample + lambda_tan * L_tan + lambda_cont * L_cont + lambda_tan_cont * L_tan_cont
- L_cls: MacroFocalLoss(macro_logits, target_classes)
- L_ctrl: Smooth_L1(macro_beziers[active], target_beziers[active], beta=0.02)
- L_sample: L1(sampled_bezier_pts, target_sampled_pts)
- L_tan: Endpoint tangent alignment inside patch: (1 - cos(P1 - P0, P1_gt - P0_gt)) + (1 - cos(P3 - P2, P3_gt - P2_gt))
- L_cont: C0 Endpoint continuity: Mean L1 distance between exit point of patch (r, c) and entry point of adjacent active patch (r', c')
- L_tan_cont: C1 Tangent angle continuity: (1 - cos(P3(r,c) - P2(r,c), P1(r',c') - P0(r',c'))) across adjacent active patches, preventing matchstick kinks

---

## 3. Directory Layout
```
experiments/EXP_10_super_token_vit/
├── configs/
│   └── exp10_config.py             # Macro-patch specs (patch_size=64, micro_patch=16, grid=8x8)
├── models/
│   ├── bezier_utils.py             # Cubic Bernstein evaluation & least-squares fitting
│   ├── macro_patch_vit.py          # ViT Backbone + 4x4 Token Merge + Inter-Macro Relational Transformer
│   ├── macro_losses.py             # Focal classification + Smooth L1 + Endpoint continuity loss
│   └── macro_merger.py             # Coordinate shifting (c*64, r*64) & anti-aliased rendering
├── utils/
│   └── dataset_macro_vit.py        # 64x64 macro-patch target extraction & polyline fitting
├── scripts/
│   ├── smoke_test_macro_vit.py     # Local offline CPU/MPS sanity check
│   ├── train_macro_vit.py          # Training loop with AMP, CosineAnnealingLR, validation Dice
│   ├── evaluate_macro_vit.py       # Benchmark evaluation computing Dice, IoU, and control point error
│   └── visualize_macro_vit.py      # 4-panel diagnostic visualizer (RGB, GT vs Pred, Macro-Grid, Depth)
├── EXP_MANIFEST.md                 # Architecture documentation, math formulations, changelog
└── Run_Commands.md                 # Exact execution commands for macOS and Kaggle CUDA
```

---

## 4. Hyperparameter Specifications

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| `image_size` | $512\times 512$ | Standard surgical resolution matching L3D benchmark |
| `micro_patch_size` | $16\times 16$ px | Pretrained ViT feature patch size (32x32 = 1,024 tokens) |
| `macro_patch_size` | $64\times 64$ px | Macro prediction patch size (8x8 = 64 tokens) |
| `merge_factor` | $4\times 4$ | 16 micro-tokens merged into 1 macro-token |
| `num_ctrl_points` | $K = 4$ | Cubic Bézier (P0, P1, P2, P3) inside each 64x64 macro-patch |
| `backbone_name` | `vit_base_patch16_224` | 86M params, $D=768$, 12 layers, 12 attention heads |
| `macro_depth` | 2 | 2 relational self-attention layers on the 64 macro-tokens |
| `macro_heads` | 8 | 8 heads for inter-macro spatial reasoning |
| `lambda_cls` | 2.0 | Macro classification Focal loss |
| `lambda_ctrl` | 5.0 | Control point coordinate Smooth L1 |
| `lambda_sample` | 5.0 | Sampled curve L1 loss |
| `lambda_tan` | 1.0 | Tangent cosine alignment inside patch |
| `lambda_cont` | 1.5 | Adjacent macro-patch endpoint (C0) continuity loss |
| `lambda_tan_cont` | 1.0 | Adjacent macro-patch tangent angle (C1) continuity loss |
| `learning_rate` | 1e-4 | Head learning rate with 0.1x backbone multiplier (1e-5) |
| `batch_size` | 16 | Optimized for single 16GB GPU (T4 / P100 / RTX 3090) |
