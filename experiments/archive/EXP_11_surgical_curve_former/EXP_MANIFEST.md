# EXP_11 Manifest: SurgicalCurveFormer (BCRNet ACPI + HCR + EXP_10 Existence Gate + Soft Rasterizer)

## 1. Research Context & Core Hypothesis
- **Paper References:**
  - BCRNet (Li et al., arXiv:2506.15279v1, June 2025): Parametric 5th-order Bézier curves, ACPI, HCR, sigmoid annealing.
  - D2GPLand (Pei et al., MICCAI 2024 / MedIA 2025): L3D dataset, depth-driven geometric prompts, RGB-D fusion.
  - TopoNet (Cui et al., arXiv 2025): Topology constraints and centerline clDice.
- **Parent Experiments:**
  - `EXP_01_mask2former_pixelwise`: Baseline pixel segmentation.
  - `EXP_04_surgical_bemaptr_pure_vector`: Bézier curve queries with deformable attention.
  - `EXP_10_super_token_vit`: Macro-token spatial grouping and CLS-pose Existence Gate.
- **Primary Objective:** Build the ultimate surgical landmark detector by marrying BCRNet's parametric 5th-order Bézier curve formulation with EXP_10's whole-organ Existence Gating and a differentiable Soft Gaussian Rasterizer for end-to-end Dice optimization.
- **Target SOTA Baseline:** BCRNet (69.57% DSC, 54.16% IoU, 43.55px ASSD on L3D).

---

## 2. Architecture & Data Flow

```
Input: 4-Channel RGB-D (512x512)
   │
   ├──> Frozen SAM-ViT-B (dim=768) ──> Foundation anatomical semantics
   └──> ResNet-50 FPN (Trainable)   ──> Multi-scale spatial pyramid {f1, f2, f3, f4}
           │
           ├──> CNN Seg Decoder (Auxiliary Deep Supervision L_s)
           │
           ├──> EXP_10 Existence Gate (CLS Token) ──> Class Presence [p_m in (0, 1)]
           │
           ├──> ACPI (Adaptive Curve Proposal Initialization on f4)
           │       Δb_i in R^12 ──> b_i^j = (σ(Δb_x + logit(c_x)), σ(Δb_y + logit(c_y)))
           │       Selects Top-K (K=10) 5th-order Bézier proposals per category
           │
           └──> HCR (Hierarchical Curve Refinement across 3 stages)
                   Coarse-to-Fine: {f3, f4} ──> {f2, f3} ──> {f1, f2}
                   26 Reference Points per curve (25 curve points + 1 midpoint)
                   Deformable Cross-Attention + 3-Way Structured Self-Attention
                   (Intra-curve N=26 × Inter-curve K=10 × Inter-category M=3)
                   Refits updated 5th-order Bézier curves B_h
           │
           └──> Soft Gaussian Rasterizer (σ-annealed 30px ──> 2px)
                   Generates differentiable pseudo-masks S_rast
                   Computes end-to-end Differentiable Dice Loss L_dice
```

---

## 3. Loss Suite & Annealing Mechanics

The total loss balances dense pixel supervision and sparse parametric curve refinement via smooth sigmoid annealing:

```
lambda_d = max(lambda_d_min, 1.0 - sigmoid((epoch - anneal_center) / anneal_slope))

L_dense = lambda_s * L_s + lambda_ind * L_ind
L_curve = sum_{h=0}^3 [ lambda_cs * L_cs(s^h) + lambda_crv * L_crv(B^h) ]
L_exist = BCEWithLogits(exist_logits, exist_gt, pos_weight=3.0)
L_dice  = (1.0 - lambda_d) * SoftDice(S_rast * exist_gate, S_gt)

L_total = lambda_d * L_dense + (1.0 - lambda_d) * L_curve + lambda_exist * L_exist + lambda_dice * L_dice
```

---

## 4. Run History & Diagnostics Log

| Run ID | Epochs | Batch | LR | Key Result (Val Dice) | Root Cause Analysis & Interventions |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Run_01` | 60 | 4 | 1e-5 | **13.81%** (Rasterizer) | 1. λ_d reached 0 at ep 21 → CNN forgetting.<br>2. σ=2px gave zero gradients outside 3px.<br>3. Plain BCE caused existence collapse.<br>4. Reported metric was rasterizer, not CNN. |
| `Run_02` | 60 | 4 | 5e-5 | **15.89%** (Rasterizer)<br>*CNN L_s=0.716 (~50-60% real)* | 1. Validation loop evaluated rasterizer output rather than CNN decoder.<br>2. λ_dice=5.0 dominated 91% of loss budget.<br>3. Slower annealing and σ=30px fixed gradient flow. |
| `Run_03` | 80 | 4 | 5e-5 | **In Progress (Active)** | 1. Val metric switched to CNN decoder seg_logits_list[0].<br>2. λ_dice reduced to 0.5; λ_crv boosted to 2.0.<br>3. λ_d_min=0.05 floor active.<br>4. Target: exceed BCRNet 69.57% SOTA. |
