# CONVERSATION.md — Mathematical Specs for Monocular 3D Surgical AI (EXP5)

## 1. Unprojecting 2D Pixels + Relative Depth to Canonical 3D Space

Given:
- Pixel coordinates: u in [0, W-1], v in [0, H-1]
- Relative depth map: d(u, v) in (0, 1] from Depth Anything V2
- Assumed standard laparoscopic camera field of view: FOV = 60 degrees

### Normalized Image Coordinates:
u_norm = (u - W / 2) / (W / 2)
v_norm = (v - H / 2) / (H / 2)

### Canonical Focal Length:
f_canon = 1.0 / tan(FOV / 2) = 1.0 / tan(30 degrees) approx 1.732

### Canonical 3D Pinhole Unprojection:
X_canon = (u_norm * Z_canon) / f_canon
Y_canon = (v_norm * Z_canon) / f_canon
Z_canon = Z_min + d(u, v) * (Z_max - Z_min)

---

## 2. Hierarchical Query Formulation (MapTR / BeMapTR 3D Adaptation)

Total Queries Q = N * K

For instance i in {1, ..., N} and point index j in {1, ..., K}:
Q_{i, j} = q_{instance, i} + q_{point, j}

Where:
- q_{instance, i} in R^C identifies landmark structure i (e.g. Falciform ligament, Liver edge, Vessel)
- q_{point, j} in R^C identifies sequential vertex j along the polyline path (from start 1 to end K)

---

## 3. Training Loss Formulations

Total Loss:
L_total = lambda_cls * L_cls + lambda_pos * L_pos + lambda_dir * L_dir + lambda_len * L_len

### 3.1 Hungarian Bipartite Matching Cost:
Cost(i, sigma(i)) = lambda_cls_match * L_cls_cost + lambda_pos_match * L_pos_cost

### 3.2 Bidirectional Smooth L1 Position Loss:
L_pos = min(
    sum_{j=1}^K SmoothL1( pred_p_{i, j} - gt_p_{sigma(i), j} ),
    sum_{j=1}^K SmoothL1( pred_p_{i, j} - gt_p_{sigma(i), K - j + 1} )
)

### 3.3 Cosine Edge Direction Loss:
pred_edge_j = pred_p_{i, j+1} - pred_p_{i, j}
gt_edge_j = gt_p_{sigma(i), j+1} - gt_p_{sigma(i), j}

L_dir = (1 / (K - 1)) * sum_{j=1}^{K-1} ( 1 - ( pred_edge_j . gt_edge_j ) / ( ||pred_edge_j|| * ||gt_edge_j|| + eps ) )

---

## 4. CNN/Swin-LSTM-MDN Sequential Surgical Landmark Architecture

### 4.1 Visual Feature Conditioning Mechanisms

#### Mechanism A: Global State Initialization & Step Conditioning
1. Global Visual Vector:
   v_img = GAP( Backbone(I) ) in R^C
   v_proj = Linear(v_img) in R^{D_lstm}

2. LSTM State Seeding (Step 0):
   h_0 = tanh( W_{init_h} v_proj + b_{init_h} )
   c_0 = tanh( W_{init_c} v_proj + b_{init_c} )

3. Step Input Concatenation (Step t):
   x_t = [ S_{t-1} ; v_proj ] in R^{5 + D_lstm}

#### Mechanism B: Spatial Cross-Attention Conditioning
1. 2D Feature Map:
   F in R^{H'W' x C} from Swin/ResNet

2. Dynamic Soft Attention at Step t:
   alpha_{t, i} = softmax_i( v_a^T * tanh( W_h h_{t-1} + W_F F_i ) )
   context_t = sum_{i=1}^{H'W'} alpha_{t, i} * F_i

3. Step Input:
   x_t = [ S_{t-1} ; context_t ]

#### Mechanism C: Local Dynamic Patch Sampling (Grid Sample)
1. Current Absolute Landmark Position:
   p_{t-1} = (u_{t-1}, v_{t-1}) in [-1, 1]^2
2. Local Feature Extraction:
   local_feat_t = grid_sample( F, p_{t-1} ) in R^C
3. Step Input:
   x_t = [ S_{t-1} ; local_feat_t ]

---

### 4.2 Complete Multi-Task Loss Formulation for Sequential MDN

Total Optimization Objective:
L_total = lambda_mdn * L_mdn + lambda_state * L_state + lambda_point * L_point + lambda_dir * L_dir + lambda_curv * L_curv

#### 1. Bivariate Gaussian Mixture Negative Log-Likelihood (L_mdn):
L_mdn = - (1 / N_max) * sum_{t=1}^{N_s} log( sum_{j=1}^M pi_{j,t} * Normal( Delta x_t, Delta y_t | mu_{x,j,t}, mu_{y,j,t}, sigma_{x,j,t}, sigma_{y,j,t}, rho_{xy,j,t} ) )

Where:
sigma_x = exp(hat_sigma_x), sigma_y = exp(hat_sigma_y), rho_xy = tanh(hat_rho_xy)
pi_j = exp(hat_pi_j) / sum_{k=1}^M exp(hat_pi_k)

#### 2. Pen-State / Termination Cross-Entropy (L_state):
L_state = - (1 / N_max) * sum_{t=1}^{N_max} sum_{k=1}^3 p_{k,t} * log( q_{k,t} )
q_k = exp(hat_q_k) / sum_{j=1}^3 exp(hat_q_j)

#### 3. Expected Point Regression Loss (L_point):
hat_p_t = sum_{j=1}^M pi_{j,t} * [ mu_{x,j,t}, mu_{y,j,t} ]
L_point = (1 / N_s) * sum_{t=1}^{N_s} SmoothL1( hat_p_t - p_t^{gt} )

#### 4. Directional Cosine Alignment Loss (L_dir):
hat_e_t = hat_p_t - hat_p_{t-1}
gt_e_t = p_t^{gt} - p_{t-1}^{gt}
L_dir = (1 / (N_s - 1)) * sum_{t=2}^{N_s} ( 1 - (hat_e_t . gt_e_t) / ( ||hat_e_t||_2 * ||gt_e_t||_2 + eps ) )

#### 5. 2nd-Order Curvature Regularization (L_curv):
hat_curv_t = hat_p_{t+1} - 2 * hat_p_t + hat_p_{t-1}
gt_curv_t = p_{t+1}^{gt} - 2 * p_t^{gt} + p_{t-1}^{gt}
L_curv = (1 / (N_s - 2)) * sum_{t=2}^{N_s - 1} || hat_curv_t - gt_curv_t ||_2

---

### 4.3 Mathematically Sound Spatial Preservation: Why GAP Fails and How to Fix It

#### Problem: Global Average Pooling (GAP) Destroys Spatial Topology
Given 2D feature map F in R^{C x H x W}:
GAP(F)_c = (1 / (H * W)) * sum_{h=1}^H sum_{w=1}^W F_{c, h, w}

GAP is strictly permutation-invariant across spatial coordinates (h, w).
Therefore, GAP(F) completely discards:
1. Absolute physical coordinates of anatomical boundaries.
2. Spatial directional gradients (nabla_x F, nabla_y F).
3. Local tissue context at the current predicted landmark point p_{t-1}.

#### Mathematical Solution 1: Differentiable Bilinear Feature Sampling (Continuous Feature Field)
Define the continuous 2D visual feature field F_cont(u, v) for normalized coordinates (u, v) in [-1, 1]^2:
F_cont(u, v) = sum_{h=1}^H sum_{w=1}^W F[:, h, w] * max(0, 1 - |(u + 1) * (W - 1) / 2 - w|) * max(0, 1 - |(v + 1) * (H - 1) / 2 - h|)

Properties:
- F_cont(u, v) is continuous and piece-wise differentiable with respect to (u, v) and F.
- Spatial gradient flow: dL / dp_{t-1} = (dL / dF_cont) * (dF_cont / dp_{t-1})
- The LSTM receives exact local visual features (edge gradient, specular glissonian reflection) evaluated at the predicted coordinate p_{t-1}.

#### Mathematical Solution 2: 2D Spatial Positional Embeddings + Coordinate Query Cross-Attention
1. 2D Coordinate Encoding of Feature Map:
   F_pos(h, w) = F[:, h, w] + PE_{2D}(h, w)
   Where PE_{2D}(h, w) = [ sin(omega_k * h), cos(omega_k * h), sin(omega_k * w), cos(omega_k * w) ]

2. Predicted Point Continuous Coordinate Query:
   q_t = MLP_{coord}(p_{t-1}) + W_h * h_{t-1} in R^D

3. Cross-Attention Spatial Spotlight:
   alpha_{t, h, w} = softmax_{h, w}( (q_t^T * F_pos(h, w)) / sqrt(D) )
   context_t = sum_{h=1}^H sum_{w=1}^W alpha_{t, h, w} * F[:, h, w]

4. Input to LSTM at Step t:
   x_t = [ S_{t-1} ; context_t ; MLP_{coord}(p_{t-1}) ]

---

### 4.4 Bilinear Grid Sampling Mechanics & Multi-Scale Texture vs. Position Memorization

#### 1. Discrete Feature Grid to Continuous Sub-Pixel Coordinate Mapping
Given feature map F in R^{C x H x W} from backbone stage (e.g. H = 32, W = 32):
Continuous normalized coordinate p = (u, v) in [-1, 1]^2 maps to sub-pixel coordinates:
x_grid = (u + 1) * (W - 1) / 2
y_grid = (v + 1) * (H - 1) / 2

Let:
x_0 = floor(x_grid), x_1 = x_0 + 1
y_0 = floor(y_grid), y_1 = y_0 + 1
dx = x_grid - x_0, dy = y_grid - y_0

Bilinear Interpolated Vector:
f_sampled(p) = (1 - dx) * (1 - dy) * F[:, y_0, x_0] + dx * (1 - dy) * F[:, y_0, x_1] + (1 - dx) * dy * F[:, y_1, x_0] + dx * dy * F[:, y_1, x_1]

#### 2. Receptive Field of a Sampled Feature Vector
Each cell F[:, y, x] in deep layers (e.g. Swin / ResNet Stage 3/4) has an effective receptive field covering a 32x32 to 128x128 pixel patch in the original surgical image.
Thus, f_sampled(p) does NOT contain a single RGB pixel, but a 256-to-512 dimensional embedding of:
- Local tissue texture (fibrous vs parenchyma vs vascular smooth tissue).
- Directional gradient / edge orientation (normal to liver contour).
- Local specular reflection and lighting.

#### 3. Multi-Scale Contextual Patch Sampling (Avoiding Absolute Position Overfitting)
To prevent the model from memorizing absolute camera coordinates (which fail under camera rotation/zoom) and force it to rely on visual texture + anatomical context:

Sample across FPN Pyramid:
f_fine = BilinearSample( P3, p ) in R^{C_fine}   (Local High-Res Texture, Receptive Field ~16px)
f_med  = BilinearSample( P4, p ) in R^{C_med}    (Intermediate Contour Geometry, Receptive Field ~64px)
f_coarse = BilinearSample( P5, p ) in R^{C_coarse} (Surrounding Organ Context: Diaphragm, Gallbladder, Receptive Field ~256px)

Multi-Scale Texture Feature:
f_texture(p) = Linear( [ f_fine ; f_med ; f_coarse ] ) in R^{D_texture}

#### 4. Deformable Context Sampling (Surrounding Tissue Geometry)
Instead of 1 point, sample K_offsets points around p (orthogonal to trajectory direction e_{t-1}):
offset_k = R(theta_k) * delta_r
p_k = p + offset_k
f_context(p) = sum_{k=1}^{K_offsets} w_k * BilinearSample( F, p_k )

---

## 5. Patch-Level Bézier Curve Formulation & ViT Patch Merging (EXP9)

### 5.1 Local Patch Bézier Curve Geometry
Within patch (r, c) of size P x P, points are normalized to [0, 1]^2:
u_local = (x - c * P) / P
v_local = (y - r * P) / P

A cubic Bézier curve is defined by 4 control points P_0, P_1, P_2, P_3 in [0, 1]^2:
B(t) = (1 - t)^3 * P_0 + 3 * (1 - t)^2 * t * P_1 + 3 * (1 - t) * t^2 * P_2 + t^3 * P_3, for t in [0, 1]

Properties:
- P_0: Exact entry point into the patch along contour flow.
- P_3: Exact exit point out of the patch along contour flow.
- P_1, P_2: Intermediate shape control handles capturing curvature, curvature sign change (inflection points), and local bending.
- Tangent at entry: T_0 = 3 * (P_1 - P_0)
- Tangent at exit:  T_1 = 3 * (P_3 - P_2)

For quadratic Bézier (3 control points P_0, P_1, P_2):
B_quad(t) = (1 - t)^2 * P_0 + 2 * (1 - t) * t * P_1 + t^2 * P_2

### 5.2 Closed-Form Least-Squares Bézier Fitting from Resampled Dense Points
Given M dense points { q_k = (u_k, v_k) }_{k=1}^M inside patch (r, c) sorted by arc-length:
1. Fixed Endpoints:
   P_0 = q_1
   P_3 = q_M

2. Arc-length Parameter Assignment:
   s_k = cumsum( || q_k - q_{k-1} ||_2 ), s_1 = 0
   t_k = s_k / s_M in [0, 1]

3. Residual System for Unknown Control Points P_1, P_2:
   q_k - (1 - t_k)^3 * P_0 - t_k^3 * P_3 = 3 * (1 - t_k)^2 * t_k * P_1 + 3 * (1 - t_k) * t_k^2 * P_2
   Let a_{k,1} = 3 * (1 - t_k)^2 * t_k, and a_{k,2} = 3 * (1 - t_k) * t_k^2.
   A = [ a_{k,1}, a_{k,2} ] in R^{M x 2}
   b = [ q_k - (1 - t_k)^3 * P_0 - t_k^3 * P_3 ] in R^{M x 2}

4. Closed-Form Normal Equation Solution:
   [ P_1 ; P_2 ] = (A^T * A + lambda * I)^{-1} * A^T * b

If M == 2 (only 2 points): P_1 = (2 * P_0 + P_3) / 3, P_2 = (P_0 + 2 * P_3) / 3 (straight line degenerate).

### 5.3 Differentiable Batch Point Sampling
For N_s uniformly spaced samples t in { 0, 1/(N_s-1), ..., 1 }:
Matrix of Bernstein basis: M_basis in R^{N_s x 4}
Sampled Points:
P_{sampled} = M_basis * [ P_0 ; P_1 ; P_2 ; P_3 ] in R^{N_s x 2}

Point Loss:
L_sample = (1 / N_s) * sum_{j=1}^{N_s} SmoothL1( hat_P_{sampled, j} - gt_P_{sampled, j} )

### 5.4 Standard ViT Patch Merging Paradigms
1. MAE Tensor Unpatchify / Fold (Dense Mask Reconstruction):
   Each patch token z_{(r, c)} predicts a P x P raster stroke patch:
   stroke_patch = MLP(z) or AnalyticalSplat(hat_B) in R^{P x P}
   Fold: (B, G, G, P, P) -> Reshape -> (B, 1, G * P, G * P) = (B, 1, H, W)
   Zero interpolation blur, exact spatial tiling.

2. Global Coordinate Shift (Continuous Parametric SVG / Vector Output):
   For active patches, unproject local Bézier control points to global image coordinates:
   P_{global, j} = (c * P, r * P) + P_{local, j} * P
   Draw directly via vector rendering engine (anti-aliased line / SVG cubic path "M P0 C P1 P2 P3").

3. Graph Stitching (Topological Continuous Polyline Assembly):
   Build adjacency graph where patch (r, c) connects to neighbor (r', c') if:
   || P_{3}^{(r, c)} - P_{0}^{(r', c')} ||_2 < threshold_px.
   Yields continuous, arbitrarily long anatomical curves.
