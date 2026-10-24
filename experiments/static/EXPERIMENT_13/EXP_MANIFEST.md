# EXPERIMENT_13: Controlled Single-Variable Replication on L3D-2K (Junction-Steered Mask2Former)

## 1. Executive Summary & Scientific Purpose
EXPERIMENT_13 is a controlled, single-variable ablation designed to resolve the fundamental bottleneck of 2D laparoscopic liver landmark detection. 

In EXPERIMENT_5, the `JunctionSteeredMask2Former` architecture established our current state-of-the-art benchmark:
- **Val Macro Dice:** 68.13%
- **Test Macro Dice:** 66.86%
- **Patient 40 (Severe Inversion/Deformation):** 70.91% Dice
- **Boundary Precision (ASSD):** 19.54 px

Subsequent multi-task attempts with non-liver surgical data (EXPERIMENT_12 with CholecSeg8k) degraded performance (66.47% Val Dice) due to procedure mismatch and domain dilution. 

To determine whether the 68% Dice plateau is caused by **clinical cohort scarcity** (39 patients in L3D) or an **architectural 2D ceiling**, EXPERIMENT_13 replicates the exact architecture, loss formulation, and optimization pipeline of EXPERIMENT_5, modifying *strictly one variable*: replacing the 921-frame L3D dataset with the expanded **L3D-2K** dataset (1,532 train frames across 47 patients).

---

## 2. Dataset Cohort Comparison

| Parameter | L3D (Experiments 1–5, 10–12) | L3D-2K (EXPERIMENT_13) | Net Expansion |
| :--- | :--- | :--- | :--- |
| **Total Frames** | 1,152 | 2,000 | +848 (+73.6%) |
| **Train Frames** | 921 | 1,532 | +611 (+66.3%) |
| **Val Frames** | 122 | 230 | +108 (+88.5%) |
| **Test Frames** | 109 | 238 | +129 (+118.3%) |
| **Patient Cohort** | 39 unique patients | 47 unique patients | +8 clinical patients (+20.5%) |
| **Anatomical Schema** | 3 landmark classes (Ridge, Sil, Falc) | 3 landmark classes (Ridge, Sil, Falc) | 100% Identical |
| **Junction Keypoints** | 4 biological junctions (J_top, J_bot, J_lat_r, J_lat_l) | 4 biological junctions (J_top, J_bot, J_lat_r, J_lat_l) | 100% Identical |
| **Stroke Rendering** | 35 px radius on raw canvas $\to$ 1024x1024 | 35 px radius on raw canvas $\to$ 1024x1024 | 100% Identical |

---

## 3. Architectural Design

The architecture is identical to EXPERIMENT_5:
1. **Backbone:** Swin-Tiny pretrained on ADE20K.
2. **Pixel Decoder:** Multi-Scale Deformable Attention (MSDeformAttn) producing continuous multi-scale feature pyramids (strides 8, 16, 32) and high-resolution mask features (stride 4).
3. **Junction Anchor Head:** 
   - 4 learnable anatomical queries: J_top (Falciform-Silhouette root), J_bottom (Umbilical notch), J_lat_right (Right extremity), J_lat_left (Left extremity).
   - 2-layer Transformer Decoder attending to stride-16 feature maps.
   - Dual projection heads: normalized coordinate regression (x, y) in [0, 1]^2 with anatomical prior initialization + binary visibility classification.
4. **Dynamic Query Steering Block:**
   - 100 Mask2Former object queries attend to the 4 junction anchor tokens via cross-attention with a learnable gating highway (initialized at gamma=0.1).
5. **Transformer Decoder:** 9-layer Mask2Former decoder computing Hungarian bipartite matching loss.

---

## 4. Multi-Task Loss Formulation

Plain math formulation:
L_total = lambda_m2f * L_m2f + lambda_coord * L_coord + lambda_vis * L_vis

Where:
- L_m2f = 2.0 * L_cls_focal + 5.0 * L_mask_bce + 5.0 * L_mask_dice
- L_coord = Smooth_L1(pred_junction_coords, gt_junction_coords, beta=0.02) computed over visible junctions
- L_vis = BCEWithLogits(pred_junction_vis, gt_junction_vis)
- Loss weights: lambda_m2f = 1.0, lambda_coord = 5.0, lambda_vis = 1.0

---

## 5. Optimization & Training Protocol

- **Epochs:** 60
- **Batch Size:** 2 per step
- **Gradient Accumulation:** 2 steps (Effective Batch Size = 4)
- **Optimizer:** AdamW with differential learning rates:
  - Backbone: 1e-5
  - Decoders, Heads, Steering Block: 1e-4
  - Weight Decay: 1e-4
- **Learning Rate Scheduler:** CosineAnnealingLR (T_max=60, eta_min=1e-6)
- **Mixed Precision:** Automatic Mixed Precision (AMP) with torch.amp.GradScaler('cuda')
- **Gradient Clipping:** Max norm 5.0
- **Validation Tracking:** L3D-2K Val Split (230 frames) evaluated at every epoch; checkpoint saved upon higher Val Macro Dice.
- **Cross-Evaluation:** At the conclusion of training, the best checkpoint is evaluated on:
  1. L3D-2K Validation Set (230 frames)
  2. L3D-2K Test Set (238 frames)
  3. Original L3D Benchmark Test Set (109 frames) for direct head-to-head comparison against EXPERIMENT_5.

---

## 6. Scientific Decision Rule

- **If Val Macro Dice > 71.0% and Test Macro Dice > 70.0%:**
  - *Empirically proves:* The historical bottleneck was clinical sample size and patient variability. The 2D transformer architecture continues to scale with expanded anatomical diversity.
- **If Val Macro Dice plateaus at 67.5% – 68.5%:**
  - *Empirically proves:* The 2D pixel-wise segmentation representation has hit an information-theoretic asymptote on thin 35-pixel surgical curves. Further progress strictly requires 3D geometric depth lifting (Surgical GeMap) or continuous parametric spline/graph reasoning rather than additional 2D image annotations.
