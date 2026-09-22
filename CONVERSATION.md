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

