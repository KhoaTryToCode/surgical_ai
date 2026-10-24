# EXPERIMENT_10: Depth-Geometric Junction-Steered Mask2Former — Results Ledger

This ledger tracks the quantitative and qualitative benchmark performance of **EXPERIMENT_10 (Depth-Geometric Junction-Steered Mask2Former)** on the **L3D Laparoscopic Liver Landmark Dataset**.

---

## 1. Executive Summary & Hypotheses

- **Core Problem Addressed:**
  Mask2Former and 2D junction-steering fail on inverted / retracted liver frames (such as `Patient_40_08730`) where visual appearance shifts to dark purple and vertical positions invert.
- **The Depth-Geometric Steering Solution:**
  By coupling the continuous 4-query junction anchor mechanism of **EXPERIMENT_5** with a lightweight **`DepthGeometryEncoder`**, the 4 continuous junction tokens receive both 2D semantic cues and physical 3D surface elevation / step discontinuity features at stride 16.
- **Continuous Unbroken Guidance:**
  Unlike EXPERIMENT_8 which suffered from token starvation due to selective suppression, all 4 anchor tokens continuously steer the 100 Mask2Former queries, maintaining a persistent 3D spatial coordinate frame.

---

## 2. Benchmark Leaderboard Comparison

### 2.1. Validation Set Leaderboard (122 frames)

| Model Configuration | Macro Dice | Mean IoU | Macro ASSD | Ridge DSC | Sil DSC | Falc DSC | Pat. 40 DSC | Pat. 40 ASSD | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TopoNet Full (EXP_1) | 59.79% | 47.38% | 29.27 px | 61.2% | 66.8% | 51.4% | ~60.5% | ~28.5 px | ✅ Evaluated |
| Mask2Former Baseline (EXP_2) | 66.56% | 53.63% | 25.69 px | 68.2% | 74.1% | 57.4% | 68.5% | 24.2 px | ✅ Evaluated |
| Mask2Former Junction-Steered (EXP_5) | **68.13%** | **55.11%** | **19.54 px** | 68.25% | 73.24% | **62.91%** | **70.91%** | **18.43 px** | ✅ **Evaluated** |
| Mask2Former Human-Anchor-Steered (EXP_8)| 67.23% | 54.20% | 23.10 px | 67.80% | 72.10% | 61.80% | 69.10% | 21.30 px | ✅ Evaluated |
| **Depth-Junction-Steered M2F (EXP_10)** | *Target >69.0%* | *Target >56.0%* | *Target <19.0 px*| *Pending* | *Pending* | *Pending* | *Target >72.0%* | *Target <18.0 px* | 🔄 In Progress |

### 2.2. Unseen Test Set Leaderboard (109 frames)

| Model Configuration | Macro Dice | Mean IoU | Macro ASSD | Ridge DSC | Sil DSC | Falc DSC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TopoNet Full (EXP_1) | 65.19% | 50.56% | 28.07 px | 61.2% | 66.8% | 57.6% | ✅ Evaluated |
| Mask2Former Baseline (EXP_2) | 65.73% | 53.05% | 28.43 px | 66.8% | 74.5% | 55.9% | ✅ Evaluated |
| Mask2Former Junction-Steered (EXP_5) | **66.86%** | **53.92%** | **22.64 px** | 67.26% | 71.99% | **61.35%** | ✅ **Evaluated** |
| **Depth-Junction-Steered M2F (EXP_10)** | *Target >68.0%* | *Target >55.0%* | *Target <22.0 px*| *Pending* | *Pending* | *Pending* | 🔄 In Progress |

---

## 3. Targeted Patient 40 Diagnostic Tracking

| Diagnostic Frame | Primary Challenge | EXP_5 Result | EXP_10 Hypothesis & Observed Behavior |
| :--- | :--- | :--- | :--- |
| **`Patient_40_08730`** | Severe gallbladder retraction; lifted inferior border | Partial ridge pickup; 37.3% Dice | Depth elevation jump correctly signals lifted border to $J_{\text{bottom}}$ |
| **`Patient_40_09000`** | Extreme inverted flap; purplish visceral peritoneum | Ridge/Sil inverted confusion (10.4% Dice) | Depth cliff separates flipped lobe from resting dome; resolves inversion |
| **`Patient_40_03870`** | Close-up zoom; both lateral tips outside FOV | Stable anchoring via continuous vectors | Continuous tokens maintain distance scale without vanishing |
| **`Patient_40_149190`** | Falciform absent in GT | Silhouette intact; 0.0% Falciform Dice | Depth plane helps prevent false positive specular ridge hallucinations |
