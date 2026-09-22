# EXP_07: Manifold-Steered Mask2Former Architecture

## 1. Overview & Motivation
EXPERIMENT_7 introduces **Manifold-Steered Mask2Former**, advancing beyond EXPERIMENT_5's 4-junction anchor mechanism to solve the fundamental challenges of:
1. **Partial Views & Missing Anatomy:** In laparoscopic videos (e.g. `Patient_12_0265080`), surgical retraction or camera zoom frequently occludes the Falciform ligament or lateral liver lobes. In naive steering, missing queries hallucinate phantom landmarks and inject noise into the segmentation queries.
2. **Dense Non-Rigid Organ Deformation:** Sparse 4-point anchors cannot capture curved liver surface topography under tool palpation and pneumoperitoneum pressure changes.

EXPERIMENT_7 resolves these failure modes via a **11-Query Canonical Manifold Atlas** combined with **Visibility-Gated Cross-Attention (VGS)** and an auxiliary **Continuous Ruled Manifold $(u, v)$ Head**.

---

## 2. Architecture Specifications
- **Backbone:** Swin-Tiny (pretrained on ADE20K via `facebook/mask2former-swin-tiny-ade-semantic`).
- **Pixel Decoder:** Multi-Scale Deformable Attention ($F_8, F_{16}, F_{32}$ + mask features $F_{\text{mask}}$ at stride 4, channel dimension 256).
- **11-Query Canonical Atlas Head:**
  - 11 Learnable Anatomical Queries: $Q_A \in \mathbb{R}^{11 \times 256}$, initialized with 2D sinusoidal positional encodings derived from canonical $(\bar{u}_k, \bar{v}_k)$ positions:
    - 4 Ridge queries ($R_0, R_1, R_2, R_3$) spaced evenly across $u \in [0, 1]$ at $v = 0.0$.
    - 4 Silhouette queries ($S_0, S_1, S_2, S_3$) spaced evenly across $u \in [0, 1]$ at $v = 1.0$.
    - 3 Falciform queries ($F_0, F_1, F_2$) spaced across $v \in [0, 1]$ along the central umbilical divide ($u = 0.5$).
  - 2-Layer Transformer Cross-Attention decoder over stride-16 features.
  - Coordinate Regression Head: MLP $\rightarrow$ Sigmoid $\rightarrow \hat{\mu}_k = (\hat{x}_k, \hat{y}_k) \in [0, 1]^2$.
  - Visibility Head: Linear $\rightarrow \hat{v}_k \in \mathbb{R}$ (logits).
- **Visibility-Gated Cross-Attention Steering (VGS):**
  - Base Mask2Former queries ($Q \in \mathbb{R}^{100 \times 256}$) attend to Atlas tokens ($K_A, V_A \in \mathbb{R}^{11 \times 256}$).
  - Logarithmic Attention Masking: $\text{AttnBias}_k = \log(\sigma(\hat{v}_k) + 10^{-5})$. When a landmark is occluded or absent ($\hat{v}_k \to 0$), $\text{AttnBias}_k \to -\infty$, rendering it mathematically silent in the attention softmax.
  - Value Double-Gating: $V_{\text{gated}} = \sigma(\hat{v}) \cdot V_A$.
  - Gated Residual Update: $Q_{\text{steered}} = \text{LayerNorm}(Q + \tanh(\alpha) \cdot \Delta Q)$.
- **Dense Manifold Head:**
  - Lightweight 2-layer Conv2D head on stride-4 $F_{\text{mask}}$ features predicting continuous organ coordinates $(\hat{u}, \hat{v}) \in [0, 1]^2$ with Sigmoid activation.
- **Transformer Decoder:** Standard 9-layer Mask2Former Transformer Decoder initialized with $Q_{\text{steered}}$.
- **Output:** Dense multi-class map $(B, 1024, 1024) \in \{0, 1, 2, 3\}$, 11-query atlas coordinates and visibility scores, and continuous $(u, v)$ manifold map.

---

## 3. Loss Formulations & Hyperparameters
- **Multi-Task Objective:**
  $L_{\text{total}} = L_{\text{m2f}} + 5.0 \cdot L_{\text{coord}} + 1.0 \cdot L_{\text{vis}} + 1.0 \cdot L_{\text{uv}}$
  - $L_{\text{m2f}}$: Standard Hungarian bipartite matching loss (Focal classification + Mask BCE + Mask Dice).
  - $L_{\text{coord}}$: Visibility-masked Smooth-L1 loss ($\beta = 0.02$) between predicted and GT coordinates for visible landmarks ($v_k^* = 1$).
  - $L_{\text{vis}}$: BCEWithLogitsLoss between predicted visibility logits and GT visibility flags across all 11 atlas points.
  - $L_{\text{uv}}$: Liver-masked Smooth-L1 on continuous manifold coordinates:
    - Vertical loss $L_v$: Always active over liver parenchyma mask.
    - Horizontal loss $L_u$: Dynamically masked to $0$ when `has_falc` is False, preventing false coordinate penalization and lobe inversion when the central dividing ligament is absent.

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Canvas Size** | $1024 \times 1024$ | Native working resolution |
| **Line Thickness** | 35 px on raw canvas | Drawn on native camera resolution (1920x1080/4K) at thickness 35 and resized to $1024 \times 1024$ via `cv2.INTER_NEAREST` (effective width $\approx 18.7\text{ px}$), ensuring exact bit-for-bit parity with TopoNet and EXPERIMENT_1/2/5 |
| **Epochs** | 60 | Training schedule with Cosine Annealing |
| **Batch Size** | 2 | Per GPU step |
| **Accumulation Steps** | 2 | Effective batch size = 4 |
| **LR (Backbone)** | 1e-5 | Low LR to preserve pretrained ADE20K Swin-Tiny weights |
| **LR (Heads & Decoder)** | 1e-4 | Standard fine-tuning rate |
| **Weight Decay** | 1e-4 | AdamW regularizer |
| **Lambda Coord** | 5.0 | Weight for coordinate regression loss |
| **Lambda Vis** | 1.0 | Weight for visibility classification loss |
| **Lambda UV** | 1.0 | Weight for continuous manifold loss |
| **Atlas Queries** | 11 | 4 Ridge + 4 Silhouette + 3 Falciform |
| **M2F Queries** | 100 | Standard Mask2Former instance/segment queries |

---

## 4. Directory Layout
- `models/`:
  - `manifold_atlas_head.py`: 11-query Canonical Atlas Head with sinusoidal $(\bar{u}, \bar{v})$ query initialization.
  - `visibility_gated_steering.py`: Visibility-Gated Cross-Attention (VGS) steering module with logarithmic attention bias.
  - `dense_manifold_head.py`: Stride-4 2-conv pixel decoder branch predicting continuous $(u, v) \in [0, 1]^2$.
  - `manifold_steered_mask2former.py`: Full hybrid architecture integrating Swin-Tiny, Pixel Decoder, Atlas Head, VGS, and Transformer Decoder.
  - `losses.py`: Joint multi-task loss ($L_{\text{m2f}} + 1.0 \cdot L_{\text{vis}} + 5.0 \cdot L_{\text{coord}} + 1.0 \cdot L_{\text{uv}}$) with conditional horizontal masking.
- `utils/`:
  - `atlas_extractor.py`: Deterministic 11-landmark canonical atlas extractor handling missing structures and visibility flags.
  - `dataset.py`: L3D PyTorch dataset loader with native-canvas stroke width 35, continuous $(u, v)$ ruled manifold generation, and `has_falc` detection.
  - `metrics.py`: Complete evaluation suite (Macro Dice, Mean IoU, ASSD, class Dices, Patient 40 diagnostics, and 11-query atlas pixel errors).
- `scripts/`:
  - `train.py`: Full training script with cosine annealing, differential LRs, AMP, validation tracking, and automatic `results.zip` packaging.
  - `evaluate.py`: Standalone evaluation script generating metrics summary, per-frame CSVs, Patient 40 diagnostic montages, and `results.zip`.
  - `run_server.sbatch`: Slurm job script for execution on `gpu-a240` (NVIDIA A100 MIG 20GB).
- `notebooks/`:
  - `EXP7_ManifoldSteered_Mask2Former_Kaggle.ipynb`: Self-contained 1-click execution notebook for Kaggle CUDA environments with automatic `results_exp7.zip` export.

---

## 5. Status
🚀 **Implemented, Unit-Tested, and Ready for Training & Benchmark Evaluation**
