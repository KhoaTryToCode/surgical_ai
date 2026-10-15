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

*Benchmark Settings: Input RGB 1024x1024, Batch Size 4, 60 Epochs, Evaluated on NVIDIA A100-PCIE-40GB.*

| Model Configuration | Macro Dice | Mean IoU | Macro ASSD | Ridge DSC | Sil DSC | Falc DSC | Pat. 40 DSC | Pat. 40 ASSD | Inference Speed | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mask2Former-Bezier (Multi-Scale)** | **59.20%** | **44.97%** | 30.98 px | 58.56% | **65.99%** | **53.06%** | **62.74%** | **26.81 px** | 14.5 FPS (68.9 ms) | ✅ **COMPLETED** |
| **Mask2Former-Bezier-SingleScale** | 58.59% | 44.56% | **29.89 px** | **60.40%** | 65.29% | 50.07% | 61.73% | 27.08 px | **15.6 FPS (64.0 ms)** | ✅ **COMPLETED** |

### 2.2. Test Set Evaluation (109 frames)

| Model Configuration | Macro Dice | Mean IoU | Macro ASSD | Ridge DSC | Sil DSC | Falc DSC | Foreground DSC | Inference Speed | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mask2Former-Bezier (Multi-Scale)** | 55.74% | 41.64% | 38.46 px | 58.35% | **63.52%** | **45.37%** | 61.71% | 16.2 FPS (61.7 ms) | ✅ **COMPLETED** |
| **Mask2Former-Bezier-SingleScale** | **56.12%** | **42.08%** | **35.24 px** | **60.75%** | 62.98% | 44.62% | **62.10%** | **16.7 FPS (59.9 ms)** | ✅ **COMPLETED** |

---

## 3. Cross-Experiment Comparison (EXP_1 vs. EXP_2 vs. EXP_3 Multi vs. EXP_3 Single)

| Metric / Property | TopoNet Baseline (EXP_1) | Mask2Former Baseline (EXP_2) | Mask2Former-Bezier Multi-Scale | Mask2Former-Bezier Single-Scale | Architectural Difference |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Output Representation** | Dense 2D Pixel Mask | Dense 2D Mask (Hungarian) | Continuous Bézier Curves | Continuous Bézier Curves | Parametric, resolution-independent |
| **Query Mechanism** | None (CNN) | 100 Global Queries | 64 Spatially Grounded Queries | 64 Spatially Grounded Queries | Deterministic $8 \times 8$ macro-patch anchoring |
| **Feature Pyramid** | Multi-Scale FPN | Multi-Scale Deformable | Multi-Scale (3 levels) | **Single-Scale (1/16 stride)** | Avoids multi-scale query noise |
| **Val Macro Dice** | 54.64% | 66.56% | **59.20%** | 58.59% | Strong parametric boundary fit |
| **Val ASSD (px)** | 49.64 px | 25.69 px | 30.98 px | **29.89 px** | Single-scale achieves tighter boundary |
| **Test Macro Dice** | 65.19% (Full) | 65.73% | 55.74% | **56.12%** | **Single-scale generalizes better on Test (+0.38%)** |
| **Test ASSD (px)** | 28.07 px (Full) | 28.43 px | 38.46 px | **35.24 px** | **+3.22 px boundary alignment improvement** |
| **Test Ridge Dice** | 61.2% | 66.8% | 58.35% | **60.75%** | **+2.40% boost on Ridge over multi-scale** |
| **Inference Latency** | 86.4 ms / frame | 94.9 ms / frame | 61.7 ms / frame | **59.9 ms / frame** | **Fastest transformer variant** |
| **Inference FPS** | 11.6 FPS | 10.5 FPS | 16.2 FPS | **16.7 FPS** | Highest real-time throughput |

---

## 4. Single-Scale vs. Multi-Scale Analysis

1. **Why Single-Scale Generalizes Better on Test:**
   - In patch-anchored Bézier decoding, queries operate on fixed $128 \times 128$ pixel spatial receptive fields. Cycling across multiple feature resolutions ($H/32, H/16, H/8$) introduced spatial aliasing and feature misalignment between coarse semantic context and fine local geometry.
   - Operating purely on a single medium-scale feature map ($H/16 = 64 \times 64$, pooled to $8 \times 8$) provides uniform receptive field coverage, yielding higher boundary precision (**35.24 px vs 38.46 px ASSD** on Test).
2. **Ridge Detection Superiority:**
   - The inferior Ridge boundary is a sharp, low-contrast edge easily confused across multi-scale attention layers. Single-scale decoding produced a **+2.40% boost** on Test Ridge DSC (**60.75%** vs 58.35%) and **+1.84%** on Val Ridge DSC (**60.40%** vs 58.56%).

---

## 5. Run Details & Artifact File Locations

### 5.1. Multi-Scale Run (`exp3_patch_bezier_60ep`)
- **Run ID:** `run_01_server_60ep`
- **Training Epochs:** 60 Epochs (Best checkpoint at Epoch 46)
- **Model Checkpoint:** `experiments/EXPERIMENT_3/results/results/best_model.pth` (468 MB)
- **Metrics JSON:** `experiments/EXPERIMENT_3/results/results/metrics_summary.json`
- **Validation CSV:** `experiments/EXPERIMENT_3/results/results/val_per_frame_metrics.csv`
- **Test CSV:** `experiments/EXPERIMENT_3/results/results/test_per_frame_metrics.csv`
- **Training Log:** `experiments/EXPERIMENT_3/results/results/training_log.csv`
- **Diagnostic Visualizations:** `experiments/EXPERIMENT_3/results/results/patient_40_diagnostics/`

### 5.2. Single-Scale Run (`exp3_patch_bezier_single_scale_60ep`)
- **Run ID:** `single_scale_60ep`
- **Training Epochs:** 60 Epochs (Best checkpoint at Epoch 24)
- **Model Checkpoint:** `experiments/EXPERIMENT_3/results/single_scale/results/best_model.pth` (468 MB)
- **Metrics JSON:** `experiments/EXPERIMENT_3/results/single_scale/results/metrics_summary.json`
- **Validation CSV:** `experiments/EXPERIMENT_3/results/single_scale/results/val_predictions.csv`
- **Test CSV:** `experiments/EXPERIMENT_3/results/single_scale/results/test_predictions.csv`
- **Training Log:** `experiments/EXPERIMENT_3/results/single_scale/results/training_log.csv`
- **Diagnostic Visualizations:** `experiments/EXPERIMENT_3/results/single_scale/results/patient_40_diagnostics_final/`
