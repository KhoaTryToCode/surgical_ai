# Sub-Experiment Manifest: SUB_02 — Mask2Former RGB-D (Depth Anything V2 Conditioning)

---

## 1. Parent Experiment Context
- **Parent Experiment:** [`EXP_01_mask2former_pixelwise`](../../EXP_MANIFEST.md)
- **Sub-Experiment ID:** `SUB_02_mask2former_rgbd`
- **Status:** Active / In Progress

---

## 2. Research Hypothesis & Micro-Iteration Objective
- **Target Objective:** Test the minimal architectural extension of Mask2Former by introducing monocular Depth Anything V2 maps as a 4th input channel (RGB-D: 4-channel Swin-Tiny patch projection).
- **Core Hypothesis:** Supplying explicit 3D surface depth ($Z$) eliminates 2D out-of-plane angle ambiguity, suppresses false landmark hallucinations on occluding metallic instruments, and resolves anatomical inversion between Silhouette and Ridge boundaries under extreme surgical retraction.
- **Minimal Code Change Philosophy:** The Swin-Tiny patch embedding layer (`Conv2d(3, 96, 4, 4)`) is widened to `Conv2d(4, 96, 4, 4)` with ImageNet pretrained weights preserved and channel 4 initialized from RGB channel means. All Transformer decoders and masked attention mechanisms remain identical to baseline.

---

## 3. Configuration & Parameters
- **Backbone:** Swin-Tiny (Adapted to 4 channels: RGB-D)
- **Depth Source:** Precomputed Depth Anything V2 (`depth_anything_v2/*.png`)
- **Loss:** Standard BCE + Dice Query Matching Loss (No topological penalty)
- **Resolution:** 1024x1024
- **Batch Size:** 1 (Gradient accumulation = 4 -> Effective Batch Size = 4)
- **Learning Rate:** 8e-5 (Cosine Annealing)
- **Execution Script:** `sub_experiments/SUB_02_mask2former_rgbd/scripts/train_rgbd.py`

---

## 4. Expected Comparative Benchmark

| Model / Run | Input Channels | Depth Source | Val Dice (All) | Patient 40 Retracted (Ranks 1–4) | Patient 32 (Ranks 5, 8–13) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EXP_01 Baseline** | 3 (RGB) | None | ~0.65–0.70 | Collapsed (0.00–0.14) | Offset (0.38–0.50) |
| **SUB_02 RGB-D** | 4 (RGB-D) | Depth Anything V2 | *Pending* | *Evaluating* | *Evaluating* |
