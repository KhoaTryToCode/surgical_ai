# EXP_05 Manifest — Monocular 3D Vector Space Transformer with Masked Attention

## Experiment Metadata
- **Experiment ID:** `EXP_05_3d_vector_transformer`
- **Primary Goal:** Direct 3D vectorized polyline regression of surgical landmarks (Falciform ligament, Liver edge, Vessels) using monocular depth unprojection and hierarchical masked attention query decoders.
- **Parent Baseline:** `EXP_01_mask2former_pixelwise` (Mask2Former 2D segmentation) & `EXP_04_surgical_bemaptr_pure_vector` (BeMapTR 2D vector queries).
- **Date Scaffolding:** August 2026

---

## Technical Specifications & Architecture

1. **Backbone & Geometry Encoding:**
   - 2D Visual Backbone: Swin Transformer / ResNet-50 FPN (`facebook/mask2former-swin-tiny-ade-semantic`).
   - Canonical 3D Pinhole Unprojection: Maps $(u, v, d) \to (X, Y, Z)$ in canonical camera frustum using FOV $60^\circ$.
   - 3D Continuous Sinusoidal Embedding: $PE_{3D}(X, Y, Z)$ concatenated with 2D FPN features and projected via $1 \times 1$ Convolution to dimension $C = 256$.

2. **Hierarchical Query Decoder:**
   - $N = 10$ instance slots, $K = 20$ ordered point vertices per instance ($N \times K = 200$ total query tokens).
   - Dynamic 3D Proposal Head predicts initial anchors $p_{\text{anchor}}^{(0)} \in [-1, 1]^3$ for Layer 1.
   - Dual-Index Queries: $Q_{i, j} = e_{\text{inst}}^{(i)} + e_{\text{order}}^{(j)} + PE_{3D}(p_{\text{anchor}}^{(i, j)})$.
   - Shared Masked Cross-Attention: Instance query $i$ generates a 2D mask $M_i$ that constrains attention regions for all $K$ point queries of instance $i$.

3. **Dual Prediction Heads & Loss Functions:**
   - **2D Mask Head:** Predicts intermediate 2D segmentation masks $M_l \in \mathbb{R}^{B \times N \times 1024 \times 1024}$.
   - **3D Point Head:** Predicts 3D vertex coordinate displacements $\Delta p_l \in \mathbb{R}^{B \times N \times K \times 3}$.
   - **Supervision:** Instance-Level Hungarian Bipartite Matching + Bidirectional Smooth $L_1$ Position Loss + Cosine Tangent Direction Loss + 1D Discrete Laplacian Curvature Regularization + Auxiliary BCE + Dice Mask Loss.

---

## File Registry

- `configs/exp05_config.py`: Hyperparameters, loss weights, depth bounds, and dataset path resolution logic.
- `utils/dataset_analyzer.py`: Dataset GT point distribution analysis and spline approximation error optimizer.
- `utils/spline_utils.py`: Uniform arc-length cubic spline resampling and GT polyline mask rasterizer (`cv2.line`, thickness=35).
- `utils/dataset_3d.py`: PyTorch dataset loader for RGB images, depth maps, resampled 3D polylines, and rasterized 2D masks.
- `models/backbone.py`: Swin/ResNet FPN + 3D Pinhole unprojector + $PE_{3D}$ module.
- `models/proposal_head.py`: 2D/3D proposal network predicting layer 1 initial 3D anchors.
- `models/transformer_decoder.py`: Hierarchical masked attention decoder with dual prediction heads.
- `models/vector_losses_3d.py`: Hungarian matcher and complete 3D vector loss suite.
- `models/surgical_3d_vector_transformer.py`: Unified PyTorch model wrapper.
- `scripts/train_3d_vector_transformer.py`: End-to-end training loop with deep supervision.
- `scripts/evaluate_3d.py`: Evaluation script computing 3D Chamfer Distance errors.
- `scripts/visualize_predictions_3d.py`: Rendering script generating 2D image overlays with 3D polyline predictions.
