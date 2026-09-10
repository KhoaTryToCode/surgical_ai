# Experiment Manifest: EXP_13 — Mask2Former + TopoNet Loss + BCRNet (Master Synthesis)

---

## 1. Research Context & Core Hypothesis
- **Parent Lineage:**
  - `EXP_01_mask2former_pixelwise` (Mask2Former Swin-T/ResNet-50 pixel baseline, 68% Dice when combined with TopoNet loss)
  - `EXP_12_surgical_curve_former_v2` (BCRNet reimplementation with FPN CNN decoder, 66.59% Dice)
  - Literature: TopoNet (Cui et al., 2024), Mask2Former (Cheng et al., 2022), BCRNet (Li et al., 2025)
- **Primary Objective:** Formulate and execute the master synthesis model combining:
  1. Mask2Former's Multi-Scale Pixel Decoder & Masked Attention Transformer Decoder
  2. TopoNet's Multi-Class Centerline Constraint (`clDice`)
  3. BCRNet's 5th-Order Parametric Bézier Curve Refinement (`M-HCR`)
- **Core Hypothesis:**
  Replacing BCRNet's naive 4-layer CNN decoder with Mask2Former's masked attention architecture, co-supervised by TopoNet's soft centerline clDice loss, breaks the 66% CNN decoder ceiling to secure 68%+ pixel segmentation while projecting refined query tokens directly into smooth 5th-order continuous Bézier curves for surgical robotic AR overlay.

---

## 2. Architecture & Codebase Design
- **Configuration (`configs/`):** [`exp13_config.py`](configs/exp13_config.py)
- **Utilities (`utils/`):**
  - [`bezier_ops.py`](utils/bezier_ops.py): Bernstein polynomial evaluation, least-squares curve fitting, unit tangent calculation.
  - [`dataset.py`](utils/dataset.py): Multi-modal 4-channel RGB-D dataset loader for L3D.
- **Models (`models/`):**
  - [`mask2former_engine.py`](models/mask2former_engine.py): 4-channel RGB-D ResNet-50 FPN + Pixel Embedding + Masked Attention Transformer Decoder.
  - [`toponet_cldice.py`](models/toponet_cldice.py): Soft morphological skeletonization + multi-class clDice loss + Analytical Continuous clDice.
  - [`query_curve_bridge.py`](models/query_curve_bridge.py): Mask2Former query-to-Bézier control point bridge with bounded tanh offsets.
  - [`masked_hcr.py`](models/masked_hcr.py): Mask-gated hierarchical curve refinement on $F_{\text{gated}} = F_{\text{pixel}} \odot (1 + \sigma(M_{\text{pred}}))$ with factored 3-way structured attention.
  - [`joint_losses.py`](models/joint_losses.py): Unified loss combining Mask2Former CE, TopoNet clDice, Bidirectional Chamfer L1, Tangent alignment, and Containment loss.
  - [`mask2former_bcrnet.py`](models/mask2former_bcrnet.py): End-to-end master model.
- **Execution Scripts (`scripts/`):**
  - [`train_exp13.py`](scripts/train_exp13.py): Training with AMP, Cosine Annealing, WandB live dashboard.
  - [`evaluate_exp13.py`](scripts/evaluate_exp13.py): Dual pixel/curve evaluation and 4-panel visual reporting (including Patient 40).

---

## 3. Configuration & Parameters
- **Backbone:** ResNet-50 FPN (4 channels: RGB-D)
- **Decoder:** 6 layers of Masked Cross-Attention ($N=15$ queries, 5 per landmark)
- **Curve Order:** 5th-order Bézier ($K=6$ control points, $N=26$ sampled points)
- **Resolution:** $512 \times 512$
- **Batch Size:** 4
- **Learning Rate:** $6 \times 10^{-5}$ with Cosine Annealing to $1 \times 10^{-6}$
- **Epochs:** 60

---

## 4. Benchmark Metric Targets

| Landmark Class | Baseline CNN (EXP_12) | Mask2Former + TopoNet (EXP_01) | Target Synthesis (EXP_13) | Output Representation |
| :--- | :---: | :---: | :---: | :---: |
| **Anterior Ridge** | 68.85% | ~70% | **>71%** | 2D Pixel Mask + 5th-Order Bézier |
| **Liver Silhouette** | 78.26% | ~80% | **>81%** | 2D Pixel Mask + 5th-Order Bézier |
| **Falciform Ligament** | 51.91% | ~54% | **>58%** | 2D Pixel Mask + 5th-Order Bézier |
| **Macro Mean Dice** | 66.59% | ~68.0% | **>70.0%** | Dual Pixel & Continuous Curves |
