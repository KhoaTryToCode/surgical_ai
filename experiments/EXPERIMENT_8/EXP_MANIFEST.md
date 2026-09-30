# EXP_08: Human-Anchor-Steered Mask2Former Architecture

## 1. Overview & Motivation
EXPERIMENT_8 builds upon the successful **Junction-Steered Mask2Former** (EXPERIMENT_5), replacing heuristic algorithmic pseudo-anchors with high-precision **Human-Refined Biological Anchors** across all 921 training frames ([`train_biological_anchors_human.json`](../../data/llm_annotate/train_biological_anchors_human.json)).

### Core Enhancements:
1. **Human Ground-Truth Supervision:** 921 surgical training images audited and verified by human expert annotation, fixing coordinate noise and resolving true organ-specific landmark confluences.
2. **Selective Anchor Prediction:** Visibility head incorporates Sigmoid probabilities $P(\text{vis}) = \text{sigmoid}(v_k)$, allowing the model the explicit choice whether to predict an anchor or suppress it when absent / occluded.
3. **Exact Benchmark Parity:** Adheres bit-for-bit to the evaluation suite across EXPERIMENT_1–7, generating `metrics_summary.json`, `val_predictions.csv`, `test_predictions.csv`, 101 Patient 40 diagnostic montages, and `results.zip`.

---

## 2. Architecture Specifications
- **Backbone:** Swin-Tiny (pretrained on ADE20K via `facebook/mask2former-swin-tiny-ade-semantic`).
- **Pixel Decoder:** Multi-Scale Deformable Attention ($F_8, F_{16}, F_{32}$ + mask features $F_{\text{mask}}$ at stride 4).
- **Junction Anchor Head:**
  - 4 Learnable Anatomical Queries: $Q_J \in \mathbb{R}^{4 \times 256}$.
  - 2-Layer Transformer Cross-Attention decoder over stride-16 features.
  - Coordinate Regression Head: MLP $\to$ Sigmoid $\to \mu_k = (x_k, y_k) \in [0, 1]^2$.
  - Visibility Head: Linear $\to$ Logits $v_k \to$ Sigmoid $\to P(\text{vis}_k) \in [0, 1]$.
- **Query Steering Block:**
  - Base Mask2Former queries ($Q \in \mathbb{R}^{100 \times 256}$) attend to $K_J = F_J$, $V_J = F_J$.
  - Gated Residual Update: $Q_{\text{steered}} = \text{LayerNorm}(Q + \text{gate} \cdot \Delta Q)$.
- **Transformer Decoder:** 9-layer Mask2Former Transformer Decoder.
- **Output:** Dense multi-class map $(B, 1024, 1024) \in \{0, 1, 2, 3\}$, plus coordinates and visibility probabilities.

---

## 3. Loss Formulations & Hyperparameters
- **Multi-Task Objective:**
  $L_{\text{total}} = L_{\text{m2f}} + 5.0 \cdot L_{\text{coord}} + 1.0 \cdot L_{\text{vis}}$
  - $L_{\text{m2f}}$: Standard Hungarian bipartite matching loss (Classification Focal + Mask BCE + Mask Dice).
  - $L_{\text{coord}}$: Visibility-masked Smooth-L1 on active human ground-truth junctions ($\beta = 0.02$).
  - $L_{\text{vis}}$: Binary Cross-Entropy with Logits on junction visibility flags.

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Canvas Size** | $1024 \times 1024$ | Native working resolution |
| **Line Thickness** | 35 px on raw canvas | Resized via `cv2.INTER_NEAREST` to maintain TopoNet / EXP_1/2/5 benchmark standard |
| **Epochs** | 60 | Training schedule with Cosine Annealing |
| **Batch Size** | 2 | Per GPU step |
| **Accumulation Steps** | 2 | Effective batch size = 4 |
| **LR (Backbone)** | 1e-5 | Preserves pretrained ADE20K weights |
| **LR (Heads & Decoder)** | 1e-4 | Standard fine-tuning rate |
| **Weight Decay** | 1e-4 | AdamW regularizer |
| **Lambda Coord** | 5.0 | Weight for coordinate loss |
| **Lambda Vis** | 1.0 | Weight for visibility loss |
| **Anchor Ground Truth** | `train_biological_anchors_human.json` | 921 human-refined surgical frames |

---

## 4. Directory Layout
- `models/`:
  - `junction_head.py`: 4-query junction anchor module with Sigmoid visibility output.
  - `junction_steered_mask2former.py`: Full model architecture.
  - `losses.py`: Joint multi-task loss.
- `utils/`:
  - `dataset.py`: L3D PyTorch dataset loader connecting to human anchor annotations.
  - `metrics.py`: Metric evaluation suite with NumPy 2.0 compatibility.
- `scripts/`:
  - `train.py`: Full training script with cosine annealing, validation tracking, and results.zip packaging.
  - `evaluate.py`: Standalone evaluation script with Patient 40 diagnostics.
  - `run_server.sbatch`: SLURM job script for execution on `gpu-a240`.

---

## 5. Status
🚀 **Implemented, Verified with Smoke Test, and Ready for Server Deployment**
