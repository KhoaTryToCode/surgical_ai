# Experiment Manifest — EXPERIMENT_15: Temporal Mask2Former (Option B)

## 1. Experiment Metadata
* **Experiment ID:** `EXP_15`
* **Title:** Temporal-Augmented Mask2Former with Interleaved Landmark Prototype Conditioning
* **Domain:** Laparoscopic Liver Landmark Detection (Ridge, Silhouette, Falciform Ligament)
* **Dataset:** L3D Benchmark (1024x1024, stroke width 35px)
* **Target Literature Benchmarks:**
  * TopoNet (CVPR 2024): 65.19% Test Micro-Dice
  * BCRNet (MICCAI 2025): 69.57% Test Macro-Dice
  * A2ONet (2024): 65.31% Test Dice

---

## 2. Hypotheses Tested
1. **Teacher Forcing in Training:**
   During training, previous frames ($t-2, t-1$) use 100% Ground Truth masks to pool clean, noise-free visual prototypes ($P_{\text{Ridge}}, P_{\text{Silhouette}}, P_{\text{Ligament}}$), enabling stable, accelerated learning of landmark deformation dynamics without early training cascades.
2. **Distribution Shift / Exposure Bias Test:**
   During validation and testing, previous frames use the model's own detached predictions to directly measure whether test-distribution mismatch degrades performance or if prototype pooling is robust.
3. **Option B Interleaved Conditioning:**
   In each of the 9 decoder layers, target queries cross-attend to the 6 past landmark prototypes before performing standard masked cross-attention on current multi-scale features.
4. **100% Benchmark Coverage:**
   Using strictly causal context padding ($[I_{t-2}, I_{t-1}, I_t]$ when gap $\le 240$, $[I_{t-1}, I_{t-1}, I_t]$ for single-frame context, $[I_t, I_t, I_t]$ for isolated frames), exactly 100% of all frames across all splits are evaluated:
   * **Train:** 921 clips (100% of 921 frames, 32 patients)
   * **Val:** 122 clips (100% of 122 frames, 3 patients: P18=6, P32=15, P40=101)
   * **Test:** 109 clips (100% of 109 frames, 4 patients: P21=7, P31=14, P41=84, P51=4)
   * Zero dropped frames!

---

## 3. Benchmark Parity & Bug Fixes Integrated
1. **TopoNet Exact Two-Step Rasterization:** Fixed line thickness mismatch. Replaced direct 35px drawing at 1024x1024 with TopoNet's exact two-step method: lines are drawn with stroke thickness 35 on native 1920x1080 (or 3840x2160) canvas, then downsampled via `cv2.resize(canvas, (1024, 1024), interpolation=cv2.INTER_NEAREST)`. This produces an effective stroke width of ~18.7 pixels, matching published SOTA baselines pixel-for-pixel.
2. **Mutually Exclusive Argmax Competitive Assignment:** Eliminated independent multi-label thresholding. Every pixel is now uniquely assigned via `pred_map = np.where(max_prob >= threshold, argmax_cls, 0)` with class indices in {0: Background, 1: Ridge, 2: Silhouette, 3: Falciform Ligament}, identically matching TopoNet's `output.argmax(dim=1)` formulation.
3. **Official Typo Resolution (`'rigde'`):** Mapped `'rigde': 1` in `CLASS_MAP`. Audit revealed 100% of Ridge annotations in Val and Test are spelled `'rigde'`, restoring 121/122 Val frames and 108/109 Test frames to active ground truth supervision.
4. **Falciform Missing Label Fix:** Mapped `'ligament': 3` in `CLASS_MAP` (the official annotator string in L3D JSONs), guaranteeing Falciform Ligament is supervised and scored.
5. **Exact Dual SOTA Metric Reporting:**
   * **Micro-Dice (TopoNet CVPR 2024 standard):** Flattened one-hot foreground overlap: `2 * sum(pred[1:] * gt[1:]) / (sum(pred[1:]) + sum(gt[1:]))`.
   * **Macro-Dice (BCRNet MICCAI 2025 standard):** Arithmetic unweighted mean across landmark classes.
   * **Contour Distances:** ASSD and HD95 with Euclidean distance transforms.

---

## 4. Architectural Specifications
* **Base Architecture:** `facebook/mask2former-swin-tiny-ade-semantic`
* **Temporal Window:** $T=3$ frames ($t-2, t-1 \to t$)
* **Landmark Prototypes:** $K_{\text{past}} \in \mathbb{R}^{B \times 6 \times 256}$
* **Temporal Cross-Attention:** 8 attention heads with learnable residual gate $\gamma$ initialized to $0.1$.
* **Gradient Isolation:** Past frames are evaluated under `torch.no_grad()`; gradients only flow through current frame and temporal adapters. Peak VRAM $\le 5.8\text{ GB}$ on Tesla T4.

