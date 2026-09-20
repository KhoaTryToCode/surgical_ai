# EXP_05: Junction-Steered Mask2Former Architecture

## 1. Overview & Motivation
EXPERIMENT_5 introduces **Junction-Steered Mask2Former**, a hybrid architecture explicitly designed to overcome the **Spatial Prior Inertia** failure mode diagnosed in EXPERIMENT_2 and EXPERIMENT_4 (e.g. `Patient_40_08730` grasper retraction and `Patient_40_03870` camera zoom).

By dynamically steering the 100 Mask2Former queries using 4 biological junction anchor tokens ($J_{\text{top}}, J_{\text{bottom}}, J_{\text{lat\_right}}, J_{\text{lat\_left}}$), the queries can shift their spatial receptive field dynamically, becoming indifferent to static canvas coordinates and aligning with real organ deformation.

---

## 2. Architecture Specifications
- **Backbone:** Swin-Tiny (pretrained on ADE20K via `facebook/mask2former-swin-tiny-ade-semantic`).
- **Pixel Decoder:** Multi-Scale Deformable Attention ($F_8, F_{16}, F_{32}$ + mask features $F_{\text{mask}}$ at stride 4).
- **Junction Anchor Head:** 
  - 4 Learnable Anatomical Queries: $Q_J \in \mathbb{R}^{4 \times 256}$.
  - 2-Layer Transformer Cross-Attention decoder over stride-16 features.
  - Coordinate Regression Head: MLP $\rightarrow$ Sigmoid $\rightarrow \mu_k = (x_k, y_k) \in [0, 1]^2$.
  - Visibility Head: Linear $\rightarrow v_k \in \mathbb{R}$ (logits).
- **Query Steering Block:**
  - Multi-Head Cross-Attention: Base Mask2Former queries ($Q \in \mathbb{R}^{100 \times 256}$) attend to $K_J = F_J$, $V_J = F_J$.
  - Gated Residual Update: $Q_{\text{steered}} = \text{LayerNorm}(Q + \alpha \cdot \Delta Q)$.
- **Transformer Decoder:** 9-layer Mask2Former Transformer Decoder.
- **Output:** Dense multi-class map $(B, 1024, 1024) \in \{0, 1, 2, 3\}$, plus auxiliary junction predictions.

---

## 3. Loss Formulations & Hyperparameters
- **Multi-Task Objective:**
  $L_{\text{total}} = L_{\text{m2f}} + 5.0 \cdot L_{\text{coord}} + 1.0 \cdot L_{\text{vis}}$
  - $L_{\text{m2f}}$: Standard Hungarian bipartite matching loss (Focal classification + Mask BCE + Mask Dice).
  - $L_{\text{coord}}$: Visibility-masked Smooth-L1 on active junctions ($\beta = 0.02$).
  - $L_{\text{vis}}$: Binary Cross-Entropy with Logits on junction visibility flags.

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Canvas Size** | $1024 \times 1024$ | Native working resolution |
| **Line Thickness** | 35 px on raw canvas | Drawn on native camera resolution (1920x1080/4K) at thickness 35 and resized to $1024 \times 1024$ via `cv2.INTER_NEAREST` (effective width $\approx 18.7\text{ px}$), ensuring exact bit-for-bit parity with TopoNet and EXPERIMENT_1/2 |
| **Epochs** | 60 | Training schedule with Cosine Annealing |
| **Batch Size** | 2 | Per GPU step |
| **Accumulation Steps** | 2 | Effective batch size = 4 |
| **LR (Backbone)** | 1e-5 | Low LR to preserve pretrained ADE20K weights |
| **LR (Heads & Decoder)** | 1e-4 | Standard fine-tuning rate |
| **Weight Decay** | 1e-4 | AdamW regularizer |
| **Lambda Coord** | 5.0 | Weight for coordinate loss |
| **Lambda Vis** | 1.0 | Weight for visibility loss |

---

## 4. Directory Layout
- `models/`:
  - `junction_head.py`: 4-query junction anchor module.
  - `junction_steered_mask2former.py`: Full model architecture.
  - `losses.py`: Joint multi-task loss.
- `utils/`:
  - `junction_extractor.py`: Deterministic 4-junction polyline extractor.
  - `dataset.py`: L3D PyTorch dataset loader.
  - `metrics.py`: Metric evaluation suite with NumPy 2.0 compatibility.
- `scripts/`:
  - `train.py`: Training script with multi-task loss and checkpointing.
  - `evaluate.py`: Standalone evaluation script with Patient 40 diagnostics.

---

## 5. Status
🚀 **Implemented & Ready for Training / Benchmarking**
