# Mathematical Formulations: EXPERIMENT_6 (Heatmap-Guided Junction-Steered Mask2Former)

## 1. Dynamic Spatial Heatmap Generation

In EXPERIMENT_6, the 4 biological junction vectors $J_k \in \mathbb{R}^{256}$ act as dynamic spatial filters over the stride-16 image feature map $F_{16} \in \mathbb{R}^{256 \times H \times W}$ ($H=W=64$).

The dot-product spatial affinity logits are computed via:
$$S_k(x, y) = \frac{\langle W_J J_k, W_F F_{16}(x, y) \rangle}{\sqrt{d}} + b_{\text{prior}}$$

where:
- $W_J \in \mathbb{R}^{128 \times 256}$ projects the junction vectors.
- $W_F \in \mathbb{R}^{128 \times 256}$ is a 1x1 convolution projecting the spatial feature map.
- $d = 128$ is the projection dimension.
- $b_{\text{prior}} = -2.19$ is the focal loss prior bias, setting the initial foreground probability $\sigma(b_{\text{prior}}) \approx 0.10$ to prevent gradient explosion from the dominant negative background pixels at step 0.

The predicted continuous spatial heatmaps are:
$$\hat{H}_k(x, y) = \sigma(S_k(x, y)) \in [0, 1]$$

---

## 2. Ground-Truth Continuous Gaussian Heatmaps

For each landmark $k \in \{1, 2, 3, 4\}$ with visibility flag $v_k \in \{0, 1\}$ and normalized coordinates $(x_k^*, y_k^*) \in [0, 1]^2$:

If the landmark is visible ($v_k = 1$):
$$H_k^*(x, y) = \exp\left( - \frac{(x - \tilde{x}_k)^2 + (y - \tilde{y}_k)^2}{2 \sigma^2} \right)$$
where $(\tilde{x}_k, \tilde{y}_k) = (x_k^* \cdot W, y_k^* \cdot H)$ and $\sigma = 2.0\text{ px}$. The discrete peak is normalized such that $\max_{(x,y)} H_k^*(x, y) = 1.0$.

If the landmark is absent / occluded ($v_k = 0$):
$$H_k^*(x, y) = 0.0 \quad \forall (x, y)$$

---

## 3. Modified Gaussian Focal Loss (CenterNet Style)

The heatmap loss $\mathcal{L}_{\text{heatmap}}$ suppresses background noise while penalizing near-misses proportionally to their distance from the true peak:

$$\mathcal{L}_{\text{heatmap}} = \frac{1}{\max(N_{\text{pos}}, 1)} \sum_{k=1}^4 \sum_{x, y} \begin{cases} - (1 - \hat{H}_k(x,y))^\alpha \log(\hat{H}_k(x,y)) & \text{if } H_k^*(x,y) = 1.0 \\ - (1 - H_k^*(x,y))^\beta (\hat{H}_k(x,y))^\alpha \log(1 - \hat{H}_k(x,y)) & \text{if } H_k^*(x,y) < 1.0 \end{cases}$$

with standard hyperparameters $\alpha = 2.0$, $\beta = 4.0$, and $N_{\text{pos}} = \sum_{k,x,y} \mathbf{1}[H_k^*(x,y) = 1.0]$.

When a landmark is absent, $H_k^*(x,y) = 0$ everywhere, so the loss drives $\hat{H}_k(x,y) \to 0$ across the entire spatial channel without forcing the model to hallucinate false coordinates.

---

## 4. Query Cross-Attention Steering (Preserved from EXPERIMENT_5)

The 100 Mask2Former queries $Q \in \mathbb{R}^{100 \times 256}$ attend to the 4 junction features $J \in \mathbb{R}^{4 \times 256}$:
$$\Delta Q = \text{Softmax}\left(\frac{Q W_q (J W_k)^T}{\sqrt{d_k}}\right) J W_v$$
$$Q_{\text{steered}} = \text{LayerNorm}(Q + \alpha \cdot \Delta Q)$$

where $\alpha$ is a learnable gating parameter initialized to 0.1.

---

## 5. Total Multi-Task Objective

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{m2f}} + \lambda_{\text{heatmap}} \mathcal{L}_{\text{heatmap}}$$

where $\lambda_{\text{heatmap}} = 1.0$ provides balanced gradient updates between the Mask2Former segmentation task ($\mathcal{L}_{\text{m2f}} \approx 2.5$) and the landmark localization task ($\mathcal{L}_{\text{heatmap}} \approx 3.0$).

---

# Mathematical Formulations: EXPERIMENT_7 (Self-Consistent Intrinsic Tangents via Differentiable Mask Gradients)

## 6. Continuous Boundary Representation & Spatial Mask Gradients

Instead of relying on heuristic, discontinuous JSON polylines to supervise directional vectors, the geometry of anatomical boundaries is treated as the level set of continuous 2D probability masks.

Let $M_c \in [0, 1]^{H \times W}$ denote the continuous spatial probability mask for anatomical class $c \in \{\text{Ridge}, \text{Silhouette}, \text{Falciform}\}$.

### 6.1 Differentiable Spatial Gradient (Sobel Operators)
The normal vector field $\vec{n}_c(x, y)$ perpendicular to the anatomical boundary is defined by the spatial gradient of the continuous mask:
$$\nabla M_c(x, y) = \left( \frac{\partial M_c}{\partial x}, \; \frac{\partial M_c}{\partial y} \right) = \left( M_c * K_x, \; M_c * K_y \right)$$

where $K_x, K_y \in \mathbb{R}^{3 \times 3}$ are fixed, non-trainable 2D Sobel convolution kernels:
$$K_x = \frac{1}{8} \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = \frac{1}{8} \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

### 6.2 Orthogonal Tangent Field
By the fundamental theorem of plane curves, the unit tangent vector $\vec{t}_c(x, y)$ parallel to the anatomical boundary is orthogonal to the gradient:
$$\vec{t}_c(x, y) = \frac{\left(-\frac{\partial M_c}{\partial y}, \; \frac{\partial M_c}{\partial x}\right)}{\sqrt{\left(\frac{\partial M_c}{\partial x}\right)^2 + \left(\frac{\partial M_c}{\partial y}\right)^2} + \epsilon}$$

Because $\vec{t}_c(x, y)$ is computed directly from the 2D mask, it is:
1. **Continuous & Smooth:** Free from discrete polyline clicking gaps or vertex jitter.
2. **Anatomically Bounded:** Physically constrained to follow the actual segmented boundary of the liver.
3. **100% Differentiable:** Gradients flow directly through $\vec{t}_c$ back into the Mask2Former pixel decoder.

### 6.3 Anchor Query Sampling & Self-Consistency Loss
Let $P_k = (x_k, y_k) \in [0, 1]^2$ be the predicted spatial coordinate of anchor landmark $k$, and let $\hat{T}_k \in \mathbb{R}^{B_k \times 2}$ be the $B_k$ unit tangent vectors predicted by the anchor query token.

The ground-truth directional target is obtained by sampling the Sobel tangent field $\vec{t}_c$ at coordinate $P_k$ using bilinear interpolation:
$$\vec{t}_c^*(P_k) = \text{GridSample}(\vec{t}_c, P_k)$$

The directional alignment loss is governed by the Cosine Similarity:
$$\mathcal{L}_{\text{tangent}} = \sum_{k=1}^4 v_k \sum_{b=1}^{B_k} \left( 1 - \langle \hat{T}_{k, b}, \; \vec{t}_{c(k, b)}^*(P_k) \rangle \right)$$

### 6.4 Chirality Cross-Product Invariant
For the 2-way corner anchors ($J_{\text{lat\_right}}$ and $J_{\text{lat\_left}}$), the canonical chirality is defined by the 2D determinant of the ridge and silhouette tangent vectors:
$$\chi(P) = \det\left( [\vec{t}_{\text{ridge}}(P), \; \vec{t}_{\text{sil}}(P)] \right) = t_{r, x} t_{s, y} - t_{r, y} t_{s, x}$$

In standard laparoscopic perspective:
- Normal Right Tip: $\chi(J_{\text{lat\_right}}) > 0$
- Inverted / Folded Flap: $\chi(J_{\text{lat\_right}}) < 0$

The network predicts a scalar chirality logit $\hat{\chi}_k$ trained via binary cross-entropy:
$$\mathcal{L}_{\text{chirality}} = \text{BCE}(\sigma(\hat{\chi}_k), \; \mathbf{1}[\chi^*(P_k) > 0])$$

---

# Mathematical Formulations: Stroke Width Discrepancy & Evaluation Sensitivity

## 7. Mathematical Proof of Dice Inflation for Ribbon/Contour Segmentations

Consider an anatomical boundary contour modeled as a ribbon of arc length $L$ and stroke width $W$.

### 7.1 Area Formulation
- Ground truth area: $|T| \approx L \cdot W$
- Predicted segment area: $|P| \approx L \cdot W$

### 7.2 Positional Centerline Shift
Suppose the predicted contour is shifted perpendicularly to the true boundary by a displacement $\delta$, where $0 \le \delta \le W$.
The overlapping intersection region between the two ribbons is:
$$|P \cap T| \approx L \cdot \max(0, W - \delta)$$

The resulting Dice coefficient is:
$$\text{Dice}(W, \delta) = \frac{2 |P \cap T|}{|P| + |T|} = \frac{2 L (W - \delta)}{L W + L W} = \frac{2(W - \delta)}{2W} = 1 - \frac{\delta}{W}$$

### 7.3 Sensitivity & Error Tolerance
The sensitivity of the Dice score to spatial centerline misalignment $\delta$ is given by:
$$\frac{\partial \, \text{Dice}}{\partial \delta} = - \frac{1}{W}$$

Evaluating for the two experimental conditions:
1. **Standard TopoNet / EXP_1 / EXP_2 Downsampled Mask ($W_1 \approx 18.7\text{ px}$):**
   $$\left| \frac{\partial \, \text{Dice}}{\partial \delta} \right|_{W_1} = \frac{1}{18.7} \approx 0.0535\text{ px}^{-1} \quad (5.35\% \text{ loss per pixel of error})$$
   For a modest $5\text{ px}$ boundary shift:
   $$\text{Dice}(18.7, 5) = 1 - \frac{5}{18.7} \approx 73.3\%$$

2. **EXPERIMENT_5 Direct 1024x1024 Mask ($W_2 = 35.0\text{ px}$):**
   $$\left| \frac{\partial \, \text{Dice}}{\partial \delta} \right|_{W_2} = \frac{1}{35.0} \approx 0.0286\text{ px}^{-1} \quad (2.86\% \text{ loss per pixel of error})$$
   For the exact same $5\text{ px}$ boundary shift:
   $$\text{Dice}(35.0, 5) = 1 - \frac{5}{35.0} \approx 85.7\%$$

**Mathematical Conclusion:**
Doubling the stroke thickness from $18.7\text{ px}$ to $35.0\text{ px}$ halves the metric's penalty for spatial errors ($\approx 1.88\times$ error forgiveness). A model evaluated on $35\text{ px}$ masks will report an artificially inflated Dice score relative to the standard benchmark.

---

## 8. Surface Distance (ASSD) Discrepancy

Let $\partial T$ and $\partial P$ denote the boundary contours of the ground-truth and predicted masks.
For a ribbon of width $W$ centered at $\gamma(s)$, its outer boundary surfaces lie at a normal distance of:
$$d_{\text{surface}}(s) = \pm \frac{W}{2}$$

When the ground truth stroke width increases from $W_1 = 18.7\text{ px}$ to $W_2 = 35.0\text{ px}$, the outer surface edges expand outwards by:
$$\Delta d = \frac{W_2 - W_1}{2} = \frac{35.0 - 18.7}{2} \approx 8.15\text{ px}$$

This explains why in `RESULTS.md`:
- Mask2Former Baseline (EXP_2, $W_1 \approx 18.7\text{ px}$): ASSD = $28.43\text{ px}$
- EXPERIMENT_5 ($W_2 = 35.0\text{ px}$): ASSD = $34.61\text{ px}$ ($\Delta \text{ASSD} \approx +6.18\text{ px}$)

The wider mask forces the boundary edge outwards, artificially inflating the Average Symmetric Surface Distance even when the centerline prediction is well-aligned.

---

# Mathematical Formulations: The Liver as an Embedded 2D Riemannian Manifold

## 9. The 2-Chart Rhombus Model & Continuous Canonical Coordinate Mapping

### 9.1 Anatomical Bijection to the Parametric Domain $\Omega$
The reference liver surface is modeled as a 2D parametric domain $\Omega \subset \mathbb{R}^2$ with intrinsic coordinates $(u, v) \in [0, 1]^2$:
- **Top Vertex $V_T = (0.5, 1.0)$:** Superior Falciform Junction $J_{\text{top}}$ (Falciform meets Silhouette).
- **Bottom Vertex $V_B = (0.5, 0.0)$:** Inferior Falciform Junction $J_{\text{bot}}$ (Falciform meets Anterior Ridge).
- **Left Vertex $V_L = (0.0, 0.5)$:** Left Lateral Apex $J_{\text{lat\_left}}$ (Segment II/III apex).
- **Right Vertex $V_R = (1.0, 0.5)$:** Right Lateral Apex $J_{\text{lat\_right}}$ (Segment VI/VII apex).
- **Crease / Hinge Edge $e_{\text{mid}} = (V_T, V_B)$:** Falciform Ligament dividing the surface into two triangular charts:
  $$\mathcal{F}_{\text{left}} = \Delta(V_L, V_T, V_B) \quad (\text{Left Lobe: Segments II, III, IV})$$
  $$\mathcal{F}_{\text{right}} = \Delta(V_R, V_T, V_B) \quad (\text{Right Lobe: Segments V, VI, VII, VIII})$$
- **Boundary Perimeters:**
  - $(V_L, V_T) \cup (V_T, V_R) \equiv$ Liver Silhouette (Superior border)
  - $(V_L, V_B) \cup (V_B, V_R) \equiv$ Anterior Ridge (Inferior margin)

### 9.2 The Physical 3D Embedding & Perspective Projection
Let $\mathbf{x}(u, v) = (x(u, v), y(u, v), z(u, v)) \in \mathbb{R}^3$ denote the deformed 3D physical liver state, observed by a calibrated camera matrix $K \in \mathbb{R}^{3 \times 3}$:
$$\mathbf{p}(u, v) = \pi(\mathbf{x}(u, v)) = \left( \frac{f_x x + c_x z}{z}, \; \frac{f_y y + c_y z}{z} \right) \in \mathbb{R}^2$$

### 9.3 Inherent Limitations of Discrete 4-Vertex Wireframes
Under local zoom (camera frustum restricted to a sub-domain $\mathcal{U} \subset \Omega$):
$$\mathcal{U} \cap \{V_T, V_B, V_L, V_R\} = \emptyset$$
All 4 global junction visibility flags collapse: $v_k = 0 \;\; \forall k \in \{1, 2, 3, 4\}$.
A discrete keypoint detector has zero spatial conditioning, causing catastrophic failure under camera close-ups.

### 9.4 Continuous Dense Canonical Coordinate Regression (DensePose for Surgery)
To make every local patch $\mathcal{U}$ observable at any arbitrary zoom level without artificial fiducial markers, a deep neural network predicts a dense canonical coordinate field $\Phi: \mathbb{R}^2 \to [0, 1]^2$:
$$\hat{\mathbf{u}}(p_x, p_y) = (\hat{u}, \hat{v})$$
for every foreground liver pixel $(p_x, p_y)$.

### 9.5 Local Metric Tensor, SVD, and Chirality/Flip Detection
Given the local affine Jacobian $J = \nabla_{(u, v)} \mathbf{p} \in \mathbb{R}^{2 \times 2}$:
$$J = \begin{bmatrix} \frac{\partial p_x}{\partial u} & \frac{\partial p_x}{\partial v} \\ \frac{\partial p_y}{\partial u} & \frac{\partial p_y}{\partial v} \end{bmatrix} = U \Sigma V^T$$

1. **Orientation / 3D Flip Invariant:**
   $$\operatorname{sgn}(\det(J)) = \begin{cases} +1 & \text{Normal Anatomical Anterior View (Right-handed)} \\ -1 & \text{Flipped / Retracted / Posterior View (Left-handed Parity)} \end{cases}$$
2. **Local Scale & Tilt:**
   Singular values $\sigma_1 \ge \sigma_2 > 0$ yield the isotropic zoom factor $\sqrt{\sigma_1 \sigma_2}$ and perspective foreshortening tilt angle $\theta = \arccos(\sigma_2 / \sigma_1)$.
3. **Dihedral Angle Across the Falciform Hinge:**
   Across the crease $e_{\text{mid}}$, the surface normals $\mathbf{n}_{\text{left}}$ and $\mathbf{n}_{\text{right}}$ define the 3D folding angle:
   $$\cos(\theta_{\text{fold}}) = \langle \mathbf{n}_{\text{left}}, \; \mathbf{n}_{\text{right}} \rangle, \quad [\![ \nabla \mathbf{p} ]\!] = \nabla \mathbf{p}\big|_{\mathcal{F}_{\text{right}}} - \nabla \mathbf{p}\big|_{\mathcal{F}_{\text{left}}} \ne 0$$

---

## 10. Dataset-Wide Transfinite Ruled Surface Parameterization & Interactive Inspector (921 Frames)

### 10.1 Boundary Interpolation via Monotonic Ruled Surface Parameterization
For any surgical video frame where laparoscopic tools or tissue overlap introduce arbitrary gaps in the 1D landmark contours, the continuous liver parenchyma domain $\mathcal{D} \subset \mathbb{R}^2$ is closed and continuously parameterized using transfinite profile interpolation along the horizontal image coordinate $x \in [x_{\min}, x_{\max}]$:
1. Let $y_{\text{sil}}(x)$ denote the piecewise linear interpolation of the superior silhouette points along the $x$-axis.
2. Let $y_{\text{ridge}}(x)$ denote the piecewise linear interpolation of the inferior anterior ridge points along the $x$-axis.
3. For every column $x$, the liver parenchyma span is bounded by $[y_{\text{top}}(x), y_{\text{bot}}(x)] = [\min(y_{\text{sil}}(x), y_{\text{ridge}}(x)), \max(y_{\text{sil}}(x), y_{\text{ridge}}(x))]$.
4. The normalized vertical coordinate $v \in [0, 1]$ is continuously defined for all $(x, y) \in \mathcal{D}$ by:
   $$v(x, y) = \begin{cases} \frac{y_{\text{bot}}(x) - y}{y_{\text{bot}}(x) - y_{\text{top}}(x)} & \text{Normal Anatomy (Ridge inferior, Silhouette superior)} \\ \frac{y - y_{\text{top}}(x)}{y_{\text{bot}}(x) - y_{\text{top}}(x)} & \text{Flipped Retraction (Ridge superior, Silhouette inferior)} \end{cases}$$
5. The normalized horizontal coordinate $u \in [0, 1]$ across the Falciform hinge $x_{\text{hinge}}$ is continuously defined by:
   $$u(x, y) = \begin{cases} 0.5 \cdot \frac{x - x_{\min}}{x_{\text{hinge}} - x_{\min}} & x \le x_{\text{hinge}} \\ 0.5 + 0.5 \cdot \frac{x - x_{\text{hinge}}}{x_{\max} - x_{\text{hinge}}} & x > x_{\text{hinge}} \end{cases}$$

### 10.2 Dataset Audit Statistics across 921 Training Frames
- **Total Valid Frames:** 921 / 921 (100.0% coverage, 0 corrupted files)
- **Complete Triad (3 Landmarks):** 841 / 921 (91.3%)
- **Partial Silhouette + Ridge (2 Landmarks):** 72 / 921 (7.8%) — Falciform hinge smoothly inferred from anatomical horizontal centerline
- **Single Landmark Visible (1 Landmark):** 8 / 921 (0.9%) — Extrapolated via anatomical margin profile
- **Flipped / Inverted Retraction Views:** 25 / 921 (2.7%) — Accurately detected via tip tangent cross-product parity and mean vertical landmark rank
- **Interactive Visualizer:** Accessible at `data/inspections/uv_viewer.html`

---

## 11. SAM 2-Guided Watertight Parenchyma Segmentation & Noise-Free Biological Anchor Derivation

### 11.1 Problem Formulation: Human Annotator Truncation Bias & Absent Landmark Dilemma
Human surgical annotators in the Laparoscopic 3D (L3D) dataset draw 1D polylines for the Anterior Ridge, Silhouette, and Falciform Ligament. Because annotating pixel-accurate curves under specular reflections and surgical smoke is laborious, annotations exhibit two systemic failure modes:
1. **Truncation Gap:** Annotators stop drawing 50–150 px before the physical anatomical boundary or where landmark polylines meet, leaving open boundaries.
2. **Forced Hallucination:** Previous heuristic anchor extractors (e.g. EXP_5 4-junction detector) assumed all 4 anchor points always exist in every frame, forcing models to regress nonexistent coordinates under camera close-ups or inverted retraction views.

### 11.2 Segment Anything Model 2 (SAM 2) Offline Super-Annotator Formulation
To eliminate human truncation bias without modifying or contaminating the official 109-frame Test Set, SAM 2 (`sam2_hiera_large`) is deployed strictly offline on the training and validation frames.

#### 11.2.1 Prompt Sampling Function $\mathcal{P}(\mathcal{C})$
Given the set of human annotated curves $\mathcal{C} = \{\mathbf{C}_{\text{ridge}}, \mathbf{C}_{\text{sil}}, \mathbf{C}_{\text{falc}}\}$:
1. **Organ Centroid Anchor:**
   $$\mathbf{c}_0 = \frac{1}{\sum_k |\mathbf{C}_k|} \sum_{k} \sum_{\mathbf{p} \in \mathbf{C}_k} \mathbf{p}$$
2. **Positive Prompts ($\mathcal{S}^+$):**
   Sample $N_{\text{pos}}$ points along $\mathbf{C}_{\text{falc}}$ (100% interior liver parenchyma) and along $\mathbf{C}_{\text{ridge}}, \mathbf{C}_{\text{sil}}$, combined with $\mathbf{c}_0$:
   $$\mathcal{S}^+ = \{\mathbf{c}_0\} \cup \left\{ \mathbf{p} \in \mathbf{C}_k \right\}$$
3. **Negative Exterior Prompts ($\mathcal{S}^-$):**
   To prevent leaking across low-contrast boundaries into the reddish diaphragm (superiorly) or stomach/bowel (inferiorly), negative prompts are projected $45\text{ px}$ outward along the radial centroid-to-boundary ray:
   $$\mathbf{p}_{\text{neg}} = \mathbf{p} + 45.0 \cdot \frac{\mathbf{p} - \mathbf{c}_0}{\|\mathbf{p} - \mathbf{c}_0\|_2}, \quad \forall \mathbf{p} \in \mathbf{C}_{\text{sil}} \cup \mathbf{C}_{\text{ridge}}$$
4. **Bounding Box Constraint:**
   $$B = [\min_x - 30, \; \min_y - 30, \; \max_x + 30, \; \max_y + 30]$$

#### 11.2.2 Watertight Boundary & Connected Component Filtering
SAM 2 outputs binary liver parenchyma mask $M \in \{0, 1\}^{H \times W}$. The continuous physical boundary contour $\Gamma$ is extracted from the maximal connected component:
$$\Gamma = \partial \left( \operatorname{CC}_{\max}(M) \right)$$

### 11.3 Geometric Derivation of the 4 Biological Anchors
From the closed contour $\Gamma = \{\mathbf{q}_i\}_{i=1}^{K}$ with $\mathbf{q}_i = (x_i, y_i)$, the 4 biological anchors are derived deterministically:
1. **Lateral Apexes ($J_{\text{lat\_left}}, J_{\text{lat\_right}}$):**
   $$\mathbf{J}_{\text{lat\_left}} = \arg\min_{\mathbf{q} \in \Gamma} x(\mathbf{q}), \quad v_3 = \mathbb{I}(x_{\min} > \delta_{\text{border}})$$
   $$\mathbf{J}_{\text{lat\_right}} = \arg\max_{\mathbf{q} \in \Gamma} x(\mathbf{q}), \quad v_2 = \mathbb{I}(x_{\max} < W - \delta_{\text{border}})$$
   where $\delta_{\text{border}} = 15\text{ px}$. If the apex touches the canvas edge, visibility is set to $v=0$.
2. **Falciform Superior and Inferior Roots ($J_{\text{top}}, J_{\text{bottom}}$):**
   If $\mathbf{C}_{\text{falc}} \ne \emptyset$, let $\mathbf{f}_{\text{top}}$ and $\mathbf{f}_{\text{bot}}$ be the superior ($y$-minimal) and inferior ($y$-maximal) endpoints of the falciform curve. The biological roots are projected onto $\Gamma$:
   $$\mathbf{J}_{\text{top}} = \arg\min_{\mathbf{q} \in \Gamma} \|\mathbf{q} - \mathbf{f}_{\text{top}}\|_2, \quad v_0 = \mathbb{I}(\min \|\mathbf{q} - \mathbf{f}_{\text{top}}\|_2 < 150 \land y(\mathbf{f}_{\text{top}}) > \delta_{\text{border}})$$
   $$\mathbf{J}_{\text{bottom}} = \arg\min_{\mathbf{q} \in \Gamma} \|\mathbf{q} - \mathbf{f}_{\text{bot}}\|_2, \quad v_1 = \mathbb{I}(\min \|\mathbf{q} - \mathbf{f}_{\text{bot}}\|_2 < 150 \land y(\mathbf{f}_{\text{bot}}) < H - \delta_{\text{border}})$$
   If no falciform ligament is visible (e.g. flipped retraction views), $v_0 = 0$ and $v_1 = 0$.

### 11.4 Visibility-Gated Cross-Attention for Mask2Former Integration
When integrating derived anchors into Junction-Steered Mask2Former, each anchor query $Q_j \in \mathbb{R}^{D}$ ($j \in \{0, 1, 2, 3\}$) predicts coordinates $\hat{\mathbf{J}}_j$ and visibility logit $\hat{s}_j$. During cross-attention with feature maps $K, V$:
$$\operatorname{Attn}(Q_j, K, V) = \operatorname{Softmax}\left( \frac{Q_j K^T}{\sqrt{d}} + \mathcal{M}_{\text{vis}}(j) \right) V$$
where the visibility mask bias is defined as:
$$\mathcal{M}_{\text{vis}}(j) = \begin{cases} 0 & \text{if } \sigma(\hat{s}_j) \ge 0.5 \\ -\infty & \text{if } \sigma(\hat{s}_j) < 0.5 \quad (\text{Absent / Occluded}) \end{cases}$$
The landmark regression loss is strictly conditioned on visibility:
$$\mathcal{L}_{\text{anchor}} = \sum_{j=0}^{3} \left[ \operatorname{BCE}(\hat{s}_j, v_j) + v_j \cdot \|\hat{\mathbf{J}}_j - \mathbf{J}_j\|_1 \right]$$
This guarantees that absent landmarks ($v_j = 0$) contribute zero spatial regression gradient and inject zero noise into the Mask2Former query representation.

### 11.5 Empirical Benchmark Across 5 Representative Test Frames
Tested using `checkpoints/sam2_hiera_large.pt` on Apple Silicon MPS (1024x1024 resolution):

| Case ID | Surgical Scenario | SAM Score | Mask Area (px) | Contour Vertices | Visible Anchors | Key Anatomical Observation |
|---|---|---|---|---|---|---|
| `case1_normal_p1` | Anterior View | 0.2113 | 784,337 | 8,834 | 2 ($J_{\text{top}}, J_{\text{bot}}$) | Lateral tips exit field of view; correctly marked $v=0$. Falciform roots locked. |
| `case2_truncated_p12` | Truncated Boundary | 0.0682 | 422,637 | 4,298 | 3 ($J_{\text{top}}, J_{\text{bot}}, J_{\text{lat\_R}}$) | Negative prompts at diaphragm prevent leakage; completes truncated boundary. |
| `case3_partial_falc_p22` | Partial Falciform Stalk | 0.8588 | 228,077 | 3,322 | 3 ($J_{\text{top}}, J_{\text{bot}}, J_{\text{lat\_L}}$) | Bridges 120px polyline gap; locks $J_{\text{top}}$ to superior contour seamlessly. |
| `case4_zoom_p40` | Patient 40 Close-Up | 0.6764 | 501,898 | 5,552 | 3 ($J_{\text{top}}, J_{\text{bot}}, J_{\text{lat\_R}}$) | Padded box prevents downward bowel bleeding; $J_{\text{lat\_L}}$ flagged absent. |
| `case5_flipped_p40` | Flipped Retraction | 0.7357 | 276,873 | 2,629 | 2 ($J_{\text{lat\_R}}, J_{\text{lat\_L}}$) | Metallic grasper segmented out cleanly; Falciform flagged ABSENT ($v=0$). Zero hallucination. |

---

## 12. Direct Zero-Shot LLM Vision Landmark Annotation vs Ground Truth

### 12.1 Evaluation Objective
To test whether Gemini Flash vision context can serve as an automated surgical annotator, we evaluated zero-shot localization of the 4 biological anchor junctions across 5 challenging surgical scenarios (normal anterior view, truncated vignette boundary, partial falciform stalk under grasper traction, high-magnification close-up, and active flipped retraction) on full-resolution (1920x1080) laparoscopic liver frames.

### 12.2 Coordinate Formulation & Error Metric
For each anchor keypoint $j \in \{J_{\text{top}}, J_{\text{bottom}}, J_{\text{lat\_right}}, J_{\text{lat\_left}}\}$, the prediction comprises a visibility flag $\hat{v}_j \in \{0, 1\}$ and coordinates $(\hat{x}_j, \hat{y}_j)$.
When the anchor is visible in ground truth ($v_j = 1$ and $\hat{v}_j = 1$), the Euclidean pixel localization error is:
$$E_j = \sqrt{(\hat{x}_j - x_j^*)^2 + (\hat{y}_j - y_j^*)^2}$$
Normalized error relative to image width $W = 1920$:
$$e_j = \frac{E_j}{W} \times 100\%$$

### 12.3 Quantitative Results
- Total evaluated anchor points: 20 (across 5 images)
- Visibility Classification Accuracy: 100.0% (20/20 correctly classified)
- Mean Absolute Pixel Error (MAPE): 10.50 px (0.55% of image width)
- Median Absolute Pixel Error: 3.95 px
- Minimum Error: 0.00 px (Patient 40 Close-up $J_{\text{bottom}}$)
- Maximum Error: 50.33 px (Patient 12 $J_{\text{bottom}}$, resulting from human polyline termination gaps)

### 12.4 Diagnostic Visualizations
Generated and verified under `data/llm_annotate/outputs/`:
- `eval_Patient_1_0174060.jpg`
- `eval_Patient_12_0025620.jpg`
- `eval_Patient_22_0029760.jpg`
- `eval_Patient_40_03870.jpg`
- `eval_Patient_40_08730.jpg`
- `benchmark_overview_dashboard.jpg`

---

## 13. Architectural Autopsy: Why Selective Anchor Suppression (EXP_8) Degraded Performance vs. Continuous Steering (EXP_5)

### 13.1 Empirical Discrepancy
- **EXPERIMENT_5 (Continuous Junction Steering):**
  - Validation Macro Dice: **68.13%**
  - Mean IoU: **55.11%**
  - Macro ASSD: **19.54 px**
  - Patient 40 Average Dice: **70.91%**
- **EXPERIMENT_8 (Selective Visibility-Gated Anchors):**
  - Validation Macro Dice: **67.23%** (-0.90% drop)
  - Patient 40 Hard Cases (`Patient_40_08730`): Stagnated at 37.33%

### 13.2 Mathematical and Structural Root Causes

#### 1. The Token Starvation Trap (Information Bottleneck)
In EXPERIMENT_5, all 4 junction queries Q_J in R^{4 x 256} continuously pass through cross-attention with the image feature map F_{16} and update the 100 Mask2Former segmentation queries:
Q_steered = LayerNorm(Q + alpha * CrossAttn(Q, Q_J, Q_J))

Even when an anatomical point (e.g. J_{lat_right}) is physically outside the camera field-of-view, its Transformer query does NOT output random static noise. Instead, it aggregates spatial global context from the visible liver margin and acts as a **continuous directional orientation vector** pointing toward the organ apex.

In EXPERIMENT_8, introducing selective visibility gating:
M_vis(j) = 0 if sigma(v_j) >= 0.5, else -inf (suppressed)
caused an immediate collapse on partial views:
- Empirical dataset distribution across 921 training images:
  - J_top visible: 86.8% (799 / 921)
  - J_bottom visible: 88.9% (819 / 921)
  - J_lat_right visible: **20.2%** (186 / 921) — absent in 79.8% of frames!
  - J_lat_left visible: **27.1%** (250 / 921) — absent in 72.9% of frames!

When lateral tips or retracted falciform roots are suppressed, the steering block receives null or zeroed key/value tokens (V_J -> 0). Exactly on the most difficult, cropped, or retracted frames (such as Patient 40), the Mask2Former decoder was **starved of geometric steering**, forcing it to fall back to static unsteered spatial queries.

#### 2. Gradient Freezing and BCE Objective Domination
The multi-task loss was formulated as:
L_total = L_m2f + 5.0 * L_coord + 1.0 * L_vis
where L_coord was visibility-masked:
L_coord = (1 / sum(v_k*)) * sum_k v_k* * SmoothL1(J_hat_k - J_k*)

Because lateral tips had v_k* = 0 for ~80% of images:
1. **Zero Spatial Gradient:** For 8 out of every 10 training batches, the lateral anchor query weights received **zero gradient** from the coordinate loss.
2. **Existence Classification Shortcut:** The BCE visibility loss L_vis heavily penalized any active coordinate feature, driving query weights to specialize strictly in binary presence detection (v_k -> -inf) rather than extracting rich spatial boundary features.

### 13.3 Prescriptive Decision for EXPERIMENT_10
1. **Build directly on EXPERIMENT_5:** EXPERIMENT_10 must retain the continuous, un-gated query steering formulation of EXPERIMENT_5, where all 4 junction tokens continuously pass spatial information into Mask2Former queries.
2. **Auxiliary Visibility (Non-Blocking):** If anchor visibility is needed for clinical interpretation or audit, it must remain a strictly feedforward diagnostic output branch that **never masks, thresholds, or zeroes** the continuous tokens Q_J entering the query steering cross-attention module.
3. **Synergy with Depth Fusion:** Depth maps provide physical surface elevation Delta z and boundary step discontinuities. Passing fused RGB-D features into continuous junction queries enables them to orient inverted flaps in 3D without suffering from missing-token starvation.


---

## 14. EXPERIMENT_11: Targeted Inversion Re-Weighting Formulation & Case Specification

### 14.1 Diagnostic Root Cause: The 99:1 Dataset Spatial Prior Collapse
In standard training across 921 laparoscopic frames:
- Canonical frames (897 frames, 97.4%): Silhouette is strictly at the top ($Y \approx 0.20 - 0.40$), Ridge is strictly at the bottom ($Y \approx 0.60 - 0.85$).
- Mean vertical margin delta: Delta_Y = mean(Y_ridge) - mean(Y_sil) = +0.365 (Ridge is on average 374 pixels lower than Silhouette).
- In standard uniform sampling, the 8 severe inverted flap frames appear in only 8 iterations per epoch (~1.7% of batches). Gradients from the remaining 913 frames constantly overwrite and erase any query weights that attempt to associate Ridge with the upper half of the image.
- Consequently, on retracted/inverted views (e.g. `Patient_40_08940` and `09000`), Mask2Former detects the anatomical curves with high precision, but assigns Silhouette to the top contour and Ridge to the bottom contour (Class-Swap Inversion Trap), plunging Dice to 0.0%.

### 14.2 Specification of Target Deformed Training Cases
From our exhaustive 921-frame audit, we isolate two distinct tiers of non-canonical frames:

#### Tier 1: Severe Flap Inversion (8 Frames Total, Weight Multiplier = 15.0x)
Frames where laparoscopic graspers pull the Inferior Ridge physically level with or ABOVE the Silhouette:
1. `Patient_12_0266520`: Delta_Y = -0.000, Ridge peak Y = 0.008 (Ridge pulled across screen ceiling; resection cavity exposed beneath)
2. `Patient_49_84900`: Delta_Y = -0.074, Ridge peak Y = 0.398 (Double metallic graspers & sutures lifting inferior edge above silhouette)
3. `Patient_38_0368040`: Delta_Y = +0.029, Ridge peak Y = 0.111 (Heavy instrument traction pulling liver diagonally; 94.3% vertical span overlap)
4. `Patient_20_0014880`: Delta_Y = +0.083, Ridge peak Y = 0.005 (Giant distended gallbladder elevates liver dome; Ridge loops across top border Y=0.005)
5. `Patient_53_0133800`: Delta_Y = +0.098, Ridge peak Y = 0.161 (Dual graspers clamping and lifting visceral flap; sharp depth elevation discontinuity)
6. `Patient_12_0267420`: Delta_Y = +0.112, Ridge peak Y = 0.125 (High-tension traction on segment boundary with tumor resection bed underneath)
7. `Patient_53_0089640`: Delta_Y = +0.061, Ridge peak Y = 0.367 (Grasper with gauze pad retracts left lobe; 31.5 deg Falciform tilt, severe lateral rotation)
8. `Patient_12_0265380`: Delta_Y = +0.158, Ridge peak Y = 0.110 (Retraction grasper on upper-left pulling margin upwards)

#### Tier 2: Moderate Traction & High Ridge (Top 16 Frames, Weight Multiplier = 5.0x)
Frames where instruments or gallbladder distension pull the inferior ridge significantly high up into the upper half of the image (Y < 0.25):
`Patient_49_84930`, `Patient_20_0016200`, `Patient_20_0016080`, `Patient_33_0210000`, `Patient_20_0015600`, `Patient_12_0265320`, `Patient_20_0015960`, `Patient_38_0038880`, `Patient_12_0266580`, `Patient_12_0265980`, `Patient_61_0049680`, `Patient_22_0140280`, `Patient_12_0265080`, `Patient_38_0039240`, `Patient_8_0018840`, `Patient_12_0265140`.

### 14.3 Strategy Comparison: Re-Weighting vs. Synthetic Geometric Augmentation
1. **Why Naive Synthetic Vertical Flipping Fails:**
   - Surgical laparoscopes illuminate from the trocar downward; blood and peritoneal fluid follow gravity. Flipping an image vertically creates impossible lighting physics.
   - When a surgeon lifts a liver flap, the *visceral undersurface* is exposed (rough, caudate process, gallbladder fossa). An upside-down flipped normal frame still shows the smooth diaphragmatic dome, but upside down. Flipping labels teaches the network that diaphragmatic tissue is called "Ridge", corrupting biological feature learning.
2. **Why Targeted Re-Weighting (WeightedRandomSampler) Succeeds:**
   - Uses 100% genuine surgical frames with true human-annotated landmarks, real grasper traction mechanics, and authentic visceral textures.
   - Sampling probability mass:
     P(Tier 1) = (8 * 15.0) / (897 * 1.0 + 8 * 15.0 + 16 * 5.0) = 120 / 1097 = 10.94%
     P(Tier 2) = (16 * 5.0) / 1097 = 80 / 1097 = 7.29%
     P(Base) = 897 / 1097 = 81.77%
     Combined deformed exposure = 18.23% (~1 every 5.5 samples).
   - In every training epoch, the model encounters retracted flaps ~100 times, persistently penalizing queries that default to the canonical top/bottom prior.
3. **Photometric Jitter as Overfitting Guardrail:**
   - To prevent the Swin backbone from memorizing the 8 specific RGB images over 60 epochs, stochastic ColorJitter (brightness +/- 15%, contrast +/- 15%, saturation +/- 15%) is applied during training.
   - This perturbs pixel values without altering the exact pixel coordinates of masks or biological junctions.

### 14.4 User Directive: Unifying All 24 Deformed/Traction Cases into Tier 1
Following the user's review of the visual atlas, all 24 identified non-standard cases (8 severe inversion + 16 high traction/margin elevation) have been unified into **Tier 1** with uniform 15.0x sampling weight multiplier:
- Total probability mass: 24 * 15.0 + 897 * 1.0 = 360 + 897 = 1257
- Unified Deformed sampling probability: 360 / 1257 = 28.64%
- In every batch of 2 frames, approximately 50% of batches will contain an inverted/traction frame.


## 15. EXPERIMENT_13: Controlled Single-Variable Replication on L3D-2K

### 15.1 Scientific Objective & Hypothesis
Following the conclusion of EXPERIMENT_12 (Dual-Decoder L3D + CholecSeg8k), which plateaued at 66.47% Val Macro Dice due to laparoscopic procedural mismatch (gallbladder Calot's triangle vs. liver resection) and visual scale conflict, EXPERIMENT_13 isolates and directly addresses the core question:
**Is the ~68% Val Macro Dice ceiling of 2D landmark segmentation caused by clinical patient cohort scarcity (39 patients) or an architectural representation asymptote?**

To answer this conclusively without confounding variables, EXPERIMENT_13 preserves 100% of the architecture, loss functions, and hyperparameters from our state-of-the-art baseline (EXPERIMENT_5: 68.13% Val Macro Dice, 70.91% Patient 40 Dice), while changing strictly one variable: replacing the 921-frame L3D training set with the 1,532-frame **L3D-2K** training set (47 unique patients).

### 15.2 Dataset Expansion (L3D vs. L3D-2K)
- **Cohort:** 39 patients -> 47 patients (+20.5% clinical expansion)
- **Training frames:** 921 frames -> 1,532 frames (+66.3% training data)
- **Validation frames:** 122 frames -> 230 frames (+88.5% validation evaluation)
- **Test frames:** 109 frames -> 238 frames (+118.3% test evaluation)
- **Schema & Annotations:** Identical 3 classes (Ridge, Silhouette, Falciform) and identical 4 biological junctions (J_top, J_bottom, J_lat_right, J_lat_left).

### 15.3 Mathematical Formulation
L_total = lambda_m2f * L_m2f + lambda_coord * L_coord + lambda_vis * L_vis

Where:
- L_m2f = 2.0 * L_cls_focal + 5.0 * L_mask_bce + 5.0 * L_mask_dice (Hungarian bipartite matching loss)
- L_coord = (1 / N_vis) * sum_{k in visible} Smooth_L1(pred_junction_coords_k, gt_junction_coords_k, beta=0.02)
- L_vis = (1 / 4) * sum_{k=1}^4 BCEWithLogits(pred_junction_vis_k, gt_junction_vis_k)
- Hyperparameters: lambda_m2f = 1.0, lambda_coord = 5.0, lambda_vis = 1.0, lr_backbone = 1e-5, lr_head = 1e-4, epochs = 60, effective batch size = 4 (batch_size=2, accum_steps=2).

### 15.4 Cross-Evaluation Strategy
Upon training completion, the best checkpoint (selected via L3D-2K Val Macro Dice) is evaluated across:
1. L3D-2K Validation Split (230 frames)
2. L3D-2K Test Split (238 frames)
3. Original L3D Benchmark Test Split (109 frames) — provides a direct head-to-head comparison against EXPERIMENT_5 (which scored 66.86% Macro Dice and 70.91% Patient 40 Dice on the exact same 109 frames).

### 15.5 Scientific Decision Matrix
- If Val Dice > 71.0% and Original L3D Test Dice > 69.5%: Proves the bottleneck was clinical cohort size. 2D transformer architectures benefit from expanded anatomical patient diversity.
- If Val Dice plateaus at ~67.5% - 68.5%: Proves the bottleneck is the 2D pixel-wise raster representation. High-frequency 35-pixel curves under severe deformation require continuous parametric splines or 3D depth-driven geometric lifting (Surgical GeMap).
