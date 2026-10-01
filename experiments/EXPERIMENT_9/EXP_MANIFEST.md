# Experiment 9 Manifest: RGB-D Human-Anchor-Steered Mask2Former

## 1. Executive Summary & Objective
- **Experiment ID**: `EXPERIMENT_9`
- **Model Name**: `RGBDJunctionSteeredMask2Former`
- **Core Hypothesis**: The severe segmentation collapse on inverted/retracted laparoscopic frames (e.g. `Patient_40_08730`, where Dice is only 37.33% in 2D RGB models) stems from a 2D spatial coordinate shortcut where upper pixels are assumed to be Silhouette and lower pixels are assumed to be Ridge. Integrating a 4th monocular relative depth channel (Depth Anything v2) into the Swin-Tiny patch embedding provides explicit 3D surface depth discontinuity and instrument elevation cues without altering the Stage 1 2D segmentation benchmark formulation.
- **Scientific Control**: This experiment is a **strict, 1-variable ablation on EXPERIMENT_8**. Every single loss weight, learning rate, scheduler, query steering mechanism, biological anchor supervision, batch size, and epoch count is kept identical to EXPERIMENT_8.

---

## 2. Model Architecture & Changes Relative to EXP_8

| Component | EXPERIMENT_8 (RGB Baseline) | EXPERIMENT_9 (RGB-D Ablation) | Rationale |
| :--- | :--- | :--- | :--- |
| **Input Channels** | 3 (RGB) | **4 (RGB + Monocular Depth)** | Injects continuous relative surface depth from Depth Anything v2. |
| **Swin Patch Embed** | `Conv2d(3, 96, kernel_size=4, stride=4)` | `Conv2d(4, 96, kernel_size=4, stride=4)` | Initialized with ImageNet RGB weights for channels 0..2; channel 3 initialized as the channel mean of RGB weights. |
| **Pixel Decoder** | MSDeformAttn (HF Mask2Former) | MSDeformAttn (HF Mask2Former) | Identical multi-scale feature pyramids (stride 4, 8, 16, 32). |
| **Anchor Head** | 4-Query Biological Anchor Head | 4-Query Biological Anchor Head | Identical ($J_{\text{top}}, J_{\text{bottom}}, J_{\text{lat\_R}}, J_{\text{lat\_L}}$) with Sigmoid visibility. |
| **Query Steering** | Cross-Attention Gate ($\alpha = 0.1$) | Cross-Attention Gate ($\alpha = 0.1$) | Identical residual gated steering of 100 queries at layer 0. |
| **Transformer Decoder** | 9-layer Mask2Former Decoder | 9-layer Mask2Former Decoder | Identical query-based instance/semantic segmentation. |
| **Loss Formulation** | Hungarian Loss + $5.0 \cdot \mathcal{L}_{\text{coord}} + 1.0 \cdot \mathcal{L}_{\text{vis}}$ | Hungarian Loss + $5.0 \cdot \mathcal{L}_{\text{coord}} + 1.0 \cdot \mathcal{L}_{\text{vis}}$ | Identical loss balance. |

---

## 3. Dataset & Preprocessing

- **RGB Normalization**: ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Depth Normalization**: Empirical Depth Anything v2 statistics across 921 training frames (`mean=0.3905`, `std=0.2276`).
- **Input Tensor**: Shape `(4, 1024, 1024)`.
- **Target Annotations**: Dense 3-class segmentation masks (0=BG, 1=Ridge, 2=Silhouette, 3=Falciform) with stroke width 35.
- **Anchor Supervision**: Direct supervision from `train_biological_anchors_human.json` (921 frames).

---

## 4. Hyperparameters & Optimization

- **Optimizer**: AdamW (`lr_backbone = 1e-5`, `lr_head = 1e-4`, `weight_decay = 1e-4`).
- **Scheduler**: CosineAnnealingLR ($T_{\max} = 60$, $\eta_{\min} = 1\times 10^{-6}$).
- **Batch Size**: 1 per step, gradient accumulation steps = 4 (Effective batch size = 4).
- **Epochs**: 60.
- **Precision**: Mixed Precision (PyTorch AMP with bfloat16 / float16).
- **Hardware Target**: NVIDIA A100 MIG 2g.10gb MPS (`gpu-a240`).

---

## 5. Primary Verification Targets
1. **Validation Macro Dice**: Target $> 68.0\%$ (Beating EXP_8 67.23% peak).
2. **Patient 40 Recovery**: Specifically tracking `Patient_40_08730` Dice (EXP_8: 37.33%).
3. **ASSD**: Target $< 25.0$ px overall.
