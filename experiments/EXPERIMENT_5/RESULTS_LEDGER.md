# EXPERIMENT_5: Junction-Steered Mask2Former — Results Ledger

This ledger tracks the quantitative and qualitative benchmark performance of **EXPERIMENT_5 (Junction-Steered Mask2Former)** on the **L3D Laparoscopic Liver Landmark Dataset**.

---

## 1. Executive Summary & Core Hypotheses

- **The Problem Being Solved (Spatial Prior Inertia):**
  In standard Mask2Former, the 100 queries are static learnable embeddings that memorize canonical camera angles. When an extreme surgical deformation occurs (e.g., `Patient_40_08730` grasper retraction lifting the inferior border by 400 pixels), queries at the top of the canvas fail to recognize the ridge because they were trained to expect ridge only in the lower half of the image.
- **The Junction-Steering Solution:**
  By allowing queries to cross-attend to 4 biological landmark tokens ($J_{\text{top}}, J_{\text{bottom}}, J_{\text{lat\_right}}, J_{\text{lat\_left}}$), the queries dynamically receive coordinate translation vectors $\Delta Q = \text{Attn}(Q, K_J) \times V_J$, steering their attention to the real deformed organ location regardless of static canvas priors.
- **Topological & Metric Parity:**
  The output remains a dense $1024 \times 1024$ multi-class semantic map, evaluated using Macro Dice, IoU, ASSD, and Patient 40 diagnostic metrics.

---

## 2. Quantitative Benchmark Leaderboards

### 2.1. Validation Set Evaluation (122 frames)
*Benchmark Settings: Input RGB 1024x1024, Batch Size 4, 60 Epochs.*

| Model Configuration | Macro Dice | Mean IoU | Macro ASSD | Ridge DSC | Sil DSC | Falc DSC | Pat. 40 DSC | Pat. 40 ASSD | Inference Speed | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TopoNet Full (EXP_1) | 59.79% | 47.38% | 29.27 px | 61.2% | 66.8% | 51.4% | ~60.5% | ~28.5 px | 11.6 FPS (86.4 ms) | ✅ Done |
| Mask2Former Baseline (EXP_2) | **66.56%** | **53.63%** | **25.69 px** | **68.2%** | **74.1%** | **57.4%** | 68.5% | **24.2 px** | 10.5 FPS (94.9 ms) | ✅ Done |
| Mask2Former-Bezier (EXP_3) | 59.20% | 44.97% | 30.98 px | 58.56% | 65.99% | 53.06% | 62.74% | 26.81 px | 14.5 FPS (68.9 ms) | ✅ Done |
| Mask2Former-Bezier-SingleScale | 58.59% | 44.56% | 29.89 px | 60.40% | 65.29% | 50.07% | 61.73% | 27.08 px | **15.6 FPS (64.0 ms)** | ✅ Done |
| **Junction-Steered Mask2Former (EXP_5)** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Target: >70%* | *Target: <22px* | ~14.0 FPS | 🟡 **Ready to Run** |

### 2.2. Unseen Test Set Evaluation (109 frames)

| Model Configuration | Macro Dice | Mean IoU | Macro ASSD | Ridge DSC | Sil DSC | Falc DSC | Foreground DSC | Inference Speed | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TopoNet Full (EXP_1) | 65.19% | 50.56% | **28.07 px** | 61.2% | 66.8% | **57.6%** | 68.4% | 11.6 FPS (86.4 ms) | ✅ Done |
| Mask2Former Baseline (EXP_2) | **65.73%** | **53.05%** | 28.43 px | **66.8%** | **74.5%** | 55.9% | **71.2%** | 10.6 FPS (94.4 ms) | ✅ Done |
| Mask2Former-Bezier (EXP_3) | 55.74% | 41.64% | 38.46 px | 58.35% | 63.52% | 45.37% | 61.71% | 16.2 FPS (61.7 ms) | ✅ Done |
| Mask2Former-Bezier-SingleScale | 56.12% | 42.08% | 35.24 px | 60.75% | 62.98% | 44.62% | 62.10% | **16.7 FPS (59.9 ms)** | ✅ Done |
| **Junction-Steered Mask2Former (EXP_5)** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Target: >68%* | ~14.0 FPS | 🟡 **Ready to Run** |

---

## 3. Targeted Hard-Case Diagnostic Hypotheses (Patient 40)

| Frame ID | Challenge / Physics | Baseline Mask2Former Failure | Expected Junction-Steered Behavior |
| :--- | :--- | :--- | :--- |
| **`Patient_40_03870`** | Camera zoom-in on central liver | Over-predicts line thickness, boundary blur | $J_{\text{top}}$ and $J_{\text{bottom}}$ scale distance, correctly resizing query attention bounds |
| **`Patient_40_08730`** | Grasper pulls gallbladder upward ($\Delta y = -0.40$) | Ridge completely missed (0% DSC) | $J_{\text{bottom}}$ shifts upward to $y \approx 280$, pulling Ridge queries directly to the retracted liver border |
| **`Patient_40_08790`** | Severe vertical retraction & fold | Partial fragmented ridge | Queries maintain continuity along the vector connecting $J_{\text{bottom}}$ and $J_{\text{lat\_right}}$ |
| **`Patient_40_09450`** | Wide panoramic view with both lateral tips | Weak tip localization | Both $J_{\text{lat\_right}}$ and $J_{\text{lat\_left}}$ visible, locking the outer limits of the organ |
