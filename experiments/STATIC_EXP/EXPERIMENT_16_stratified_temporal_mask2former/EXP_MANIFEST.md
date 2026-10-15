# Experiment Manifest — EXPERIMENT_16: Stratified Temporal Mask2Former

## 1. Experiment Metadata
* **Experiment ID:** `EXP_16`
* **Title:** Stratified Temporal-Augmented Mask2Former with Interleaved Landmark Prototype Conditioning
* **Domain:** Laparoscopic Liver Landmark Detection (Ridge, Silhouette, Falciform Ligament)
* **Dataset:** L3D Benchmark (1024x1024, stroke width 35px native TopoNet canvas rasterization)
* **Target Literature Benchmarks:**
  * TopoNet (CVPR 2024): 65.19% Test Micro-Dice
  * BCRNet (MICCAI 2025): 69.57% Test Macro-Dice
  * A2ONet (2024): 65.31% Test Dice
  * EXP_15 (Temporal M2F baseline split): 68.18% Test Micro-Dice, 65.36% Test Macro-Dice

---

## 2. Rationale & Core Hypotheses
1. **Pipeline & Architectural Identity with Experiment 15:**
   * Retains 100% of the Temporal Mask2Former architecture: Option B interleaved prototype cross-attention in all 9 decoder layers.
   * Teacher forcing in training ($T=3$, past frames $t-2, t-1$ pool prototypes from 100% Ground Truth masks).
   * Autonomous inference during validation and testing (past prototypes pooled from model's own detached predictions).
   * TopoNet exact two-step 18.7px stroke rasterization, argmax mutual exclusivity with 0.5 threshold, and dual Micro/Macro Dice calculation.
   * Zero model, loss, or metric formula alterations.

2. **Stratification Hypothesis (Experiment 14 Option 1: Patient 40 <-> Patient 38 Swap):**
   * **Problem in Baseline L3D Partition:** In the standard L3D split used by EXP 15, `Patient_40` was 82.8% of Val (101 frames) and contained 15 frames of extreme mechanical elevation ($Y_{\text{ridge}} < 0.015$), whereas the Train set was 92% flat/resting views. Meanwhile, the frozen Test set is 77.1% `Patient_41`, who also exhibits severe mechanical elevation and flipped liver contours.
   * **Under-exposure During Training:** In EXP 15, the temporal attention layers never experienced dynamic liver hoisting during training, causing an out-of-distribution performance drop on Test `Patient_31` (45.32% Macro-Dice).
   * **Balanced Distribution:** Re-stratifying by swapping `Patient_40` (101 frames) into Train and `Patient_38` (99 frames) into Val gives balanced mechanical elevation rates (8.99% in Train, 8.33% in Val vs population 8.92%).
   * **Hypothesis:** Exposing the model to hoisted liver dynamics during training will prevent catastrophic pose shift on difficult test patients without compromising global benchmark stability.

---

## 3. Dataset Splits & Frame Counts (100% Benchmark Coverage)
* **Stratified Train:** 923 clips across 32 patients (includes `Patient_40`; excludes `Patient_38`).
* **Stratified Val:** 120 clips across 3 patients (`Patient_38`: 99, `Patient_32`: 15, `Patient_18`: 6).
* **Frozen Test Benchmark:** 109 clips across 4 patients (`Patient_21`: 7, `Patient_31`: 14, `Patient_41`: 84, `Patient_51`: 4) — 100% unmutated, strictly frozen for unassailable comparability with SOTA literature.
* **Strict Disjointness:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
* Zero disk mutation: reads from original Train and Val directories dynamically without copying or overwriting files on disk.

---

## 4. Key Architectural Specifications
* **Base Architecture:** `facebook/mask2former-swin-tiny-ade-semantic`
* **Temporal Window:** $T=3$ frames ($t-2, t-1 \to t$)
* **Landmark Prototypes:** $K_{\text{past}} \in \mathbb{R}^{B \times 6 \times 256}$
* **Temporal Cross-Attention:** 8 attention heads with learnable residual gate $\gamma$ initialized to $0.1$.
* **Gradient Isolation:** Past frames are evaluated under `torch.no_grad()`; gradients only flow through current frame and temporal adapters.
* **Peak VRAM:** $\le 5.8\text{ GB}$ on Tesla T4 (micro-batch size 1, gradient accumulation steps 4).
