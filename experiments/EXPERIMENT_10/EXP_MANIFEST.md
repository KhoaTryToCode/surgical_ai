# Experiment 10 Manifest: Depth-Geometric Junction-Steered Mask2Former

## 1. Executive Summary & Objective
- **Experiment ID**: `EXPERIMENT_10`
- **Model Name**: `DepthJunctionSteeredMask2Former`
- **Core Hypothesis**: 
  In 2D laparoscopic image segmentation, severe landmark inversion occurs during surgical retraction (e.g. `Patient_40_08730`) because 2D RGB models rely on canonical vertical spatial priors (assuming Ridge is at the bottom and Silhouette is at the top). When the liver is retracted/flipped, the visceral underbelly appears purplish-dark and the inferior margin is lifted upward by up to 400 pixels, confusing purely photometric features.
  By retaining the continuous 4-query junction steering mechanism of **EXPERIMENT_5** (which achieved 68.13% Dice / 19.54 px ASSD) and fusing features from a dedicated, lightweight **`DepthGeometryEncoder`** (stride 16) directly into the junction anchor head, the continuous landmark queries are explicitly grounded with 3D physical surface elevation ($\Delta z$) and boundary step edges. The 3-channel RGB Swin-Tiny backbone retains 100% of its native pretrained ImageNet/ADE20K weights.
- **Scientific Control vs. EXPERIMENT_5:**
  Every single Mask2Former component, query steering cross-attention mechanism, loss formulation, Hungarian bipartite matching weight, learning rate schedule, and native canvas rasterization (stroke width 35) is kept identical to EXPERIMENT_5. The sole architectural variable introduced is the `DepthGeometryEncoder` and its gated feature fusion at stride 16 before the junction anchor head.

---

## 2. Architecture Specifications & Comparative Control

| Component | EXPERIMENT_5 (Continuous 2D Baseline) | EXPERIMENT_10 (Depth-Geometric Upgrade) | Rationale |
| :--- | :--- | :--- | :--- |
| **RGB Backbone** | Swin-Tiny (3-channel, ADE20K pretrained) | **Swin-Tiny (3-channel, ADE20K pretrained)** | 100% preservation of pretrained ImageNet/ADE20K weights (no channel alteration). |
| **Depth Input** | None | **Monocular Depth Map ($1 \times 1024 \times 1024$)** | Extracted via Depth Anything v2, providing 3D surface elevation and step discontinuities. |
| **Depth Encoder** | None | **`DepthGeometryEncoder` (Stride 16 ConvNet)** | 4-stage residual ConvNet downsampling depth from $1024\times 1024$ to $256 \times 64 \times 64$ ($\approx 1.2\text{M}$ params). |
| **Feature Fusion** | None (RGB $F_{16}$ only) | **Gated Residual Fusion Layer** | $F_{\text{fused}} = \text{LayerNorm}(F_{16}^{\text{rgb}} + \text{Conv}_{1\times 1}([F_{16}^{\text{rgb}}, F_{16}^{\text{depth}}])) \in \mathbb{R}^{B \times 256 \times 64 \times 64}$. |
| **Junction Anchor Head** | 4 Continuous Queries ($J_{\text{top}}, J_{\text{bottom}}, J_{\text{lat\_R}}, J_{\text{lat\_L}}$) | **4 Continuous Queries** over $F_{\text{fused}}$ | Queries learn anatomical locations in both 2D semantic space and 3D depth space without suppression. |
| **Anchor Filtering** | None (Continuous un-gated flow) | **None (Continuous un-gated flow)** | Avoids the token starvation trap of EXP_8; queries always provide unbroken 4-point reference frame. |
| **Auxiliary Visibility**| Linear logit head ($v_k \in \mathbb{R}$) | **Linear logit head ($v_k \in \mathbb{R}$)** | Non-blocking diagnostic output; never masks or zeroes features fed to Mask2Former queries. |
| **Query Steering Block**| Cross-Attention ($\alpha = 0.1$) | **Cross-Attention ($\alpha = 0.1$)** | 100 Mask2Former queries attend to the 4 depth-steered geometric junction tokens. |
| **Transformer Decoder** | 9-layer Mask2Former Decoder | **9-layer Mask2Former Decoder** | Identical query-based semantic landmark prediction. |
| **Output** | Dense 3-class map ($1024 \times 1024$) | **Dense 3-class map ($1024 \times 1024$)** | Standardized native-canvas stroke width 35. |

---

## 3. Loss Formulations & Hyperparameters

- **Multi-Task Objective:**
  $L_{\text{total}} = L_{\text{m2f}} + 5.0 \cdot L_{\text{coord}} + 1.0 \cdot L_{\text{vis}}$
  - $L_{\text{m2f}}$: Standard Hungarian bipartite matching loss (Classification Focal + Mask BCE + Mask Dice).
  - $L_{\text{coord}}$: Smooth-L1 on active junction ground truth ($\beta = 0.02$).
  - $L_{\text{vis}}$: Binary Cross-Entropy on auxiliary visibility logits (non-blocking).

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Canvas Size** | $1024 \times 1024$ | Native working resolution |
| **RGB Normalization** | ImageNet stats | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |
| **Depth Normalization**| Empirical L3D stats | `mean=0.3905`, `std=0.2276` |
| **Line Thickness** | 35 px on raw canvas | Resized via `cv2.INTER_NEAREST` to maintain TopoNet / EXP_1/2/5 benchmark standard |
| **Epochs** | 60 | Training schedule with Cosine Annealing |
| **Batch Size** | 2 | Per GPU step |
| **Accumulation Steps** | 2 | Effective batch size = 4 |
| **LR (Backbone)** | 1e-5 | Preserves pretrained ADE20K weights |
| **LR (Heads & Depth Encoder)** | 1e-4 | Standard fine-tuning rate |
| **Weight Decay** | 1e-4 | AdamW regularizer |
| **Lambda Coord** | 5.0 | Weight for coordinate loss |
| **Lambda Vis** | 1.0 | Auxiliary visibility weight |

---

## 4. Directory Layout
- `models/`:
  - `depth_geometry_encoder.py`: 4-stage residual depth downsampler to stride 16.
  - `junction_head.py`: 4-query continuous junction anchor module.
  - `depth_junction_steered_mask2former.py`: Full model architecture with Swin-Tiny + Depth Encoder + Stride-16 Fusion + Steering Block.
  - `losses.py`: Joint multi-task loss.
- `utils/`:
  - `dataset.py`: PyTorch dataset loader supplying synchronized RGB (3-ch), Depth (1-ch), masks, and 4 anchors with multi-environment fallback.
  - `metrics.py`: Standard metric suite with NumPy 2.0 compatibility.
  - `junction_extractor.py`: Algorithmic biological junction coordinate extractor.
- `scripts/`:
  - `train.py`: Full training script with cosine annealing, validation tracking, and results.zip packaging.
  - `evaluate.py`: Standalone evaluation script with Patient 40 diagnostics.
  - `run_server.sbatch`: SLURM job script for execution on `gpu-a240`.
- `results/`: Output logs, predictions CSVs, and artifacts.

---

## 5. Primary Verification Targets
1. **Validation Macro Dice:** Target $> 69.0\%$ (Beating EXP_5 68.13% and EXP_8 67.23%).
2. **ASSD:** Target $< 19.0\text{ px}$ (Beating EXP_5 19.54 px).
3. **Patient 40 Hard Cases:** Target $> 72.0\%$ Dice across the 101 difficult deformed validation frames.
4. **Flipped Flap Recovery:** Significant Dice improvement on severe inverted flap cases (`Patient_40_08730`, `09000`).
