# EXP_06: Heatmap-Guided Junction-Steered Mask2Former Architecture

## 1. Overview & Motivation
EXPERIMENT_6 advances our landmark-steered segmentation framework by addressing the key limitation diagnosed in EXPERIMENT_5. 

While EXPERIMENT_5 set a new benchmark high (**69.84% Test Macro Dice**, surpassing BCRNet SOTA of 69.57%), our mathematical geometric probing and visual analysis revealed that:
1. **Absent Point Hallucination:** In deformed or zoomed surgical scenes (e.g. Patient 40), 1 to 3 anatomical junctions are physically off-screen or occluded. The MLP coordinate head regressed scalar (x, y) coordinates regardless of point existence, incurring large positional errors (147–255 px) and phantom latent hot-spots.
2. **Diffuse Latent Activation:** The J vectors captured global translation and scale (R^2 = 0.857 for Y-translation, 0.741 for area) but failed to capture fine boundary orientation or pinpoint landmarks.

**EXPERIMENT_6** keeps the core Mask2Former backbone, pixel decoder, 100 queries, and the proven delta_q gated cross-attention steering mechanism 100% intact. It solely redesigns how the 4 biological J vectors are guided and supervised: replacing the decoupled MLP coordinate head with **2D continuous spatial sigmoid heatmaps (4 x 64 x 64)** supervised by Gaussian Focal Loss (CenterNet style).

---

## 2. Architecture Specifications
- **Backbone:** Swin-Tiny (pretrained on ADE20K via `facebook/mask2former-swin-tiny-ade-semantic`).
- **Pixel Decoder:** Multi-Scale Deformable Attention ($F_8, F_{16}, F_{32}$ + mask features $F_{\text{mask}}$ at stride 4).
- **Junction Refiner Module:**
  - 4 Learnable Anatomical Queries: Q_J in R^{4 x 256} representing Top, Bottom, Lat-Right, Lat-Left.
  - 2-Layer Transformer Decoder with self-attention (anatomical graph reasoning) and cross-attention into stride-16 features.
- **Dynamic Spatial Heatmap Head (`DynamicHeatmapHead`):**
  - Linear projection of 4 J vectors: J_proj in R^{B x 4 x 128}.
  - 1x1 Conv + GroupNorm projection of stride-16 image features: F_proj in R^{B x 128 x 64 x 64}.
  - Dynamic spatial dot product:
    dots = einsum("bkc, bchw -> bkhw", J_proj, F_proj) / sqrt(d) + bias
    pred_heatmaps = sigmoid(dots) in [0, 1]^{B x 4 x 64 x 64}
  - RetinaNet/CenterNet focal prior bias initialized to -2.19 (sigmoid ~ 0.1) for zero-step gradient stability.
  - Automatic peak coordinate decoding via spatial argmax (coords in [0, 1]^2) and existence thresholding (peak >= 0.3).
- **Query Steering Block (Untouched from EXP_5):**
  - Base Mask2Former queries (Q in R^{100 x 256}) attend to K_J = J_features, V_J = J_features.
  - Gated Residual Update: Q_steered = LayerNorm(Q + alpha * delta_Q).
- **Transformer Decoder & Mask Head:** Standard 9-layer Mask2Former Transformer Decoder with Hungarian matching loss on Ridge, Silhouette, and Falciform.

---

## 3. Loss Formulations & Hyperparameters
- **Multi-Task Objective:**
  L_total = lambda_m2f * L_m2f + lambda_heatmap * L_heatmap
  - L_m2f: Standard Hungarian bipartite matching loss (Classification Focal + Mask BCE + Mask Dice).
  - L_heatmap: Continuous Gaussian Focal Loss (CenterNet modified focal loss):
    - Visible landmark (target == 1.0): - (1 - pred)^alpha * log(pred)
    - Background / near-miss (target < 1.0): - (1 - target)^beta * (pred)^alpha * log(1 - pred)
    - Normalized by visible landmark count: (pos_loss + neg_loss) / max(num_pos, 1.0)
    - When a landmark is absent, target is 0.0 everywhere, cleanly driving predicted heatmaps to zero without hallucinated coordinates.

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Canvas Size** | 1024 x 1024 | Native surgical working resolution |
| **Heatmap Resolution** | 64 x 64 | Stride-16 feature resolution (1 px = 16 canvas px) |
| **Gaussian Sigma** | 2.0 px | Peak spread in feature grid (~32 px in canvas space) |
| **Line Thickness** | 35 px on raw canvas | Drawn on native camera resolution (1920x1080/4K) at thickness 35 and resized to 1024x1024 via `cv2.INTER_NEAREST` (effective width $\approx 18.7\text{ px}$), ensuring exact bit-for-bit parity with TopoNet and EXPERIMENT_1/2 |
| **Epochs** | 60 | Training schedule with Cosine Annealing |
| **Batch Size** | 2 | Per GPU step |
| **Accumulation Steps** | 2 | Effective batch size = 4 |
| **LR (Backbone)** | 1e-5 | Low LR to preserve pretrained ADE20K Swin weights |
| **LR (Heads & Decoder)** | 1e-4 | Standard fine-tuning rate |
| **Weight Decay** | 1e-4 | AdamW regularizer |
| **Lambda Heatmap** | 1.0 | Balanced scale matching L_m2f (~2.5 - 3.5) |

---

## 4. Directory Layout
- `models/`:
  - `junction_refiner.py`: 4-query junction transformer decoder.
  - `heatmap_head.py`: Dynamic spatial dot-product heatmap head with peak decoding.
  - `heatmap_steered_m2f.py`: Full model architecture integrating refiner, heatmap head, delta_q steering, and M2F decoder.
  - `losses.py`: Joint multi-task loss (Hungarian + Gaussian Focal).
- `utils/`:
  - `heatmap_utils.py`: Deterministic continuous Gaussian heatmap synthesis and sub-pixel peak extraction.
  - `junction_extractor.py`: Geometric landmark junction extractor.
  - `dataset.py`: L3D PyTorch dataset loader with precomputed gt_heatmaps.
  - `metrics.py`: Metric evaluation suite (Macro Dice, IoU, ASSD, junction coordinate error).
- `scripts/`:
  - `train.py`: Full 60-epoch training loop with AMP bfloat16 and automated results.zip packaging.
  - `evaluate.py`: Standalone evaluation script with Patient 40 diagnostics.
  - `test_local_forward.py`: Verification script for shapes, focal loss, and autograd.
  - `run_server.sbatch`: SLURM submission script for gpu-a240 A100 MIG 20GB.

---

## 5. Status
🚀 **Code Implemented & Verified Locally — Ready for Server Training**
