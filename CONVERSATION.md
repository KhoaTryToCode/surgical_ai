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
