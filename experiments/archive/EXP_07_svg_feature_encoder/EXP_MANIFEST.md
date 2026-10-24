# EXP_07: SVG / Vector-Aware Feature Encoder vs Standard Swin Backbone

## Executive Summary
EXP_07 explores the conceptual and mathematical transition from standard pixel-wise multi-scale feature maps (used in EXP_06) to continuous SVG / Vector-Aware Neural Feature Maps. This experiment directly compares the representations extracted by both encoders on the same surgical laparoscopic image.

---

## Comparative Architectural Breakdown

### 1. Standard Swin Backbone Encoder (EXP_06)
* **Representation:** Dense 2D scalar activation grids across 4 pyramid levels (Stride 4: 256x256, Stride 8: 128x128, Stride 16: 64x64, Stride 32: 32x32).
* **Information Content:** Regional organ textures, homogeneous tissue color, diffuse receptive fields.
* **Limitation for Vectors:** Cross-attention over flattened 32x32 or 64x64 token grids cannot provide precise directional or tangent gradients for thin 1D surgical landmark lines (< 2px width).

### 2. SVG / Vector-Aware Feature Encoder (EXP_07)
* **Representation:** Continuous multi-channel geometric field comprising:
  1. **Landmark Saliency & Skeleton Field $\mathcal{S}(x, y) \in [0, 1]$:** Sharp 1D probability density along anatomical ridges.
  2. **2D Tangent Flow Field $\vec{T}(x, y) = (\cos \theta, \sin \theta)$:** Directional unit vectors tracing contour flow.
  3. **2D Normal Gradient Field $\vec{N}(x, y) = (-\sin \theta, \cos \theta)$:** Orthogonal vectors pointing across boundaries.
  4. **Curvature Field $\kappa(x, y)$:** Second-derivative bending energy.
  5. **Parametric Bézier Spline Primitives $B(t)$:** Continuous analytical cubic curves defined by $(P_0, C_1, C_2, P_3)$.
* **Advantage:** Provides explicit geometric vector guidance, preventing curve self-intersection and eliminating the discrete point-knot collapse observed in EXP_06.

---

## Directory Structure
* `configs/config_svg_encoder.py`: Configuration parameters for EXP_07.
* `models/swin_standard_encoder.py`: Standard Swin backbone feature extractor.
* `models/svg_vector_encoder.py`: SVG / Vector-Aware Feature Map Encoder.
* `utils/svg_visualizer_utils.py`: PCA projection, Quiver vector plotting, and HSV direction wheel utilities.
* `scripts/visualize_encoder_comparison.py`: 8-panel high-definition comparative visualizer.
* `outputs/encoder_feature_comparison.png`: Generated side-by-side comparison figure.
