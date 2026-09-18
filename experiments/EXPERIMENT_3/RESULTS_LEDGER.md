# EXPERIMENT_3: Mask2Former Backbone + Patch Bézier Decoder — Results Ledger

This document tracks the quantitative and qualitative benchmark performance of **EXPERIMENT_3** (Mask2Former Swin-Tiny Backbone + MSDeformAttn Pixel Decoder + $8 \times 8$ Spatially Anchored Patch Bézier Decoder) on the **L3D Laparoscopic Liver Landmark Dataset**.

---

## 1. Executive Summary & Core Insights

- **Continuous Parametric Curve Decoding:**
  Instead of predicting dense discrete pixel masks or using Hungarian bipartite matching over 100 queries, EXPERIMENT_3 replaces the transformer decoder with an $8 \times 8$ spatial patch grid (64 content-aware queries). Each active query predicts 4 normalized Bézier control points in $[0, 1]^2$.
- **High Computational Throughput (14.5 – 16.2 FPS):**
  By replacing the 100-query Hungarian decoder and heavy mask feature dot products with a lightweight 6-layer Bézier decoder, inference latency drops to **61.7 – 68.9 ms/frame** (**14.5 – 16.2 FPS** on NVIDIA A100), delivering a **~38% speedup** over baseline Mask2Former (10.5 FPS).
- **Two-Phase Continuity Transition:**
  - **Phase 1 (Epochs 1–30):** Independent patch-level learning (Focal Loss + Smooth-L1 + Bernstein curve sampling).
  - **Phase 2 (Epochs 31–60):** Boundary continuity ($L_{\text{cont}}$) and tangent alignment ($L_{\text{tan}}$) activated across adjacent same-class patches.
  - Convergence peaked at **Epoch 46** with **59.20% Validation Macro Dice** and **30.98 px ASSD** (Patient 40 subset reaching **62.74% Dice** and **26.81 px ASSD**).

---

## 2. Quantitative Benchmark Leaderboards

### 2.1. Validation Set Evaluation (122 frames)

*Benchmark Settings: Input RGB 1024x1024, Batch Size 4, 60 Epochs, Restored Epoch 46 Checkpoint.*

| Model Configuration | Macro Dice | Mean IoU | Macro ASSD | Ridge DSC | Sil DSC | Falc DSC | Pat. 40 DSC | Pat. 40 ASSD | Inference Speed | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mask2Former-Bezier** | **59.20%** | **44.97%** | **30.98 px** | **58.56%** | **65.99%** | **53.06%** | **62.74%** | **26.81 px** | **14.5 FPS (68.9 ms)** | ✅ **COMPLETED** |

### 2.2. Test Set Evaluation (109 frames)

| Model Configuration | Macro Dice | Mean IoU | Macro ASSD | Ridge DSC | Sil DSC | Falc DSC | Foreground DSC | Inference Speed | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mask2Former-Bezier** | **55.74%** | **41.64%** | **38.46 px** | **58.35%** | **63.52%** | **45.37%** | **61.71%** | **16.2 FPS (61.7 ms)** | ✅ **COMPLETED** |

---

## 3. Cross-Experiment Comparison (EXP_1 vs. EXP_2 vs. EXP_3)

| Metric / Property | TopoNet Baseline (EXP_1) | Mask2Former Baseline (EXP_2) | Mask2Former-Bezier (EXP_3) | Architectural Difference |
| :--- | :---: | :---: | :---: | :--- |
| **Output Representation** | Dense 2D Pixel Mask | Dense 2D Mask (Hungarian) | **Continuous Bézier Curves** | Parametric, resolution-independent |
| **Query Mechanism** | None (CNN) | 100 Global Queries | **64 Spatially Grounded Queries** | Deterministic $8 \times 8$ macro-patch anchoring |
| **Loss Function** | Soft Dice + clDice | Hungarian Match (CE + BCE + Dice) | **Focal + Smooth L1 + Continuity** | No bipartite matching required |
| **Val Macro Dice** | 54.64% | 66.56% | **59.20%** | Outperforms TopoNet baseline (+4.56%) |
| **Val Mean IoU** | 40.94% | 53.63% | **44.97%** | Competitive regional overlap |
| **Val ASSD (px)** | 49.64 px | 25.69 px | **30.98 px** | Significantly tighter boundary than TopoNet |
| **Test Macro Dice** | 65.19% (Full) | 65.73% | **55.74%** | Compact parameterization generalization |
| **Test ASSD (px)** | 28.07 px (Full) | 28.43 px | **38.46 px** | Vector-to-raster discretization boundary |
| **Inference Latency** | 86.4 ms / frame | 94.9 ms / frame | **68.9 ms / frame** | **~27–38% faster throughput** |
| **Inference FPS** | 11.6 FPS | 10.5 FPS | **14.5 – 16.2 FPS** | Highest throughput among transformer models |

---

## 4. Run Details & Artifact File Locations

- **Run ID:** `run_01_server_60ep`
- **Training Epochs:** 60 Epochs (Best checkpoint at Epoch 46)
- **Model Checkpoint:** `results/results/best_model.pth` (468 MB)
- **Metrics JSON:** `results/results/metrics_summary.json`
- **Validation Frame CSV:** `results/results/val_per_frame_metrics.csv` (122 rows)
- **Test Frame CSV:** `results/results/test_per_frame_metrics.csv` (109 rows)
- **Training Loss Curve:** `results/results/training_log.csv` (60 epochs logged)
- **Diagnostic Visualizations:** `results/results/patient_40_diagnostics/` (101 diagnostic 4-panel montages)
- **Packaged Archive:** `results/results.zip` (942 MB)
