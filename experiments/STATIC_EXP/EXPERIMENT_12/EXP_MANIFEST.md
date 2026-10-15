# EXPERIMENT 12 MANIFEST: Dual-Decoder Multi-Task Mask2Former (L3D + CholecSeg8k)

## 1. Executive Summary & Core Hypothesis
- **Hypothesis:** The primary performance bottleneck in laparoscopic liver landmark detection (Experiments 1–11) is clinical data scarcity (only 921 frames across 15 patients with 24 inverted cases). By co-training a shared Swin-Tiny + MSDeformAttn pixel decoder with a 13-class surgical scene segmentation decoder on CholecSeg8k (8,080 frames across 17 procedures), the shared backbone learns universal surgical texture, lighting invariance, and tool-tissue occlusion separation across 9,001 frames and 32+ patients.
- **Architectural Innovation:** A modular **Dual-Decoder Mask2Former** architecture:
  - **Shared:** Swin-Tiny Backbone + Multi-Scale Deformable Attention Pixel Decoder (outputting stride 4 mask features + strides 8, 16, 32 multi-scale features).
  - **Decoder 1 (Scene Decoder - CholecSeg8k):** 100 queries -> 9-layer Transformer Decoder -> 13 surgical scene classes.
  - **Decoder 2 (Landmark Decoder - L3D):** 4 continuous Junction Queries (`JunctionAnchorHead`) + `JunctionQuerySteering` cross-attention -> 100 steered queries -> 9-layer Transformer Decoder -> 3 landmark line classes (Ridge, Silhouette, Falciform) + (x, y, v) coordinates.

---

## 2. Loss Formulations (Plain Math)
The total joint optimization objective is defined as:

L_total = L_L3D + lambda_cholec * L_cholec

Where:
- L_L3D = L_mask_m2f + lambda_coord * L_coord + lambda_vis * L_vis
  - L_mask_m2f: HuggingFace Mask2Former Hungarian matching loss (Focal + Dice + BCE) across Ridge, Silhouette, and Falciform.
  - L_coord: Visibility-masked Smooth L1 loss on the 4 biological keypoints:
    L_coord = SmoothL1(pred_coords[vis_mask], gt_coords[vis_mask], beta=0.02)
  - L_vis: Binary cross-entropy on junction visibility logits.
- L_cholec: Hungarian matching loss across the 13 surgical scene classes (with ignore_index = 255 on watershed segment boundaries).
- Hyperparameters:
  - lambda_cholec = 0.5 (initial scene balance factor)
  - lambda_coord = 5.0
  - lambda_vis = 1.0

---

## 3. Dataset Specifications

| Dataset | Total Frames | Procedures | Spatial Classes | Annotation Format | Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **L3D** | 921 frames | ~15 patients | 3 Landmark Lines + 4 Continuous Junctions | JSON Polylines rasterized at 35px stroke | Fine-grained landmark supervision |
| **CholecSeg8k** | 8,080 frames | 17 procedures | 13 Surgical Scene Classes (Liver, Wall, Fat, Tools) | Dense Watershed PNGs (854x480) remapped via 256-LUT | Coarse organ & surgical representation learning |

---

## 4. Training & Optimization Protocol
- **Optimizer:** AdamW with Cosine Annealing Learning Rate Schedule (T_max = 60 epochs, eta_min = 1e-6).
- **Learning Rates:**
  - Backbone: 1e-5
  - Decoders & Heads: 1e-4
  - Weight Decay: 1e-4
- **Batching & Execution:**
  - Batch size: 2 per domain per GPU step.
  - Gradient Accumulation: 2 steps (effective batch size 4 per domain).
  - Mixed Precision: AMP (torch.cuda.amp.autocast float16).
  - Epoch definition: Governed by L3D (690 training samples with WeightedRandomSampler Tier-1 15.0x), while CholecSeg8k provides an infinite non-repeating stream.
- **Hardware Target:** NVIDIA A100 40GB VRAM on `gpu-a240`.

---

## 5. Validation & Model Selection
- Model checkpointing is **strictly governed by L3D Validation Macro Dice** evaluated on the 122 validation frames.
- Post-training evaluation runs full inference on both L3D Validation (122 frames) and Test (109 frames) splits.
- Trap 14 server packaging automatically packages all outputs, metrics, and Patient 40 diagnostic plots into `$SAVE_DIR/results.zip`.
