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

