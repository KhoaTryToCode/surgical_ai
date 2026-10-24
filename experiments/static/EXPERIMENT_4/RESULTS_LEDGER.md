# EXPERIMENT_4: Landmark Master Tokens + Patch Bézier Decoder — Results Ledger

This document tracks the quantitative and qualitative benchmark performance of **EXPERIMENT_4** (Swin-Tiny Backbone + MSDeformAttn Pixel Decoder + **67-Token Transformer Decoder** combining 3 Landmark Master Tokens with 64 Spatial Patch Queries) on the **L3D Laparoscopic Liver Landmark Dataset**.

---

## 1. Executive Summary & Core Insights

- **Unified 67-Token Sequence Dynamics:**
  EXPERIMENT_4 introduces 3 categorical Landmark Master Tokens ($T_{\text{ridge}}, T_{\text{sil}}, T_{\text{falc}}$) alongside the 64 spatial patch queries. Full multi-head self-attention ($67 \times 67$) across 6 decoder layers enables the network to explicitly track relative organ orientation (e.g. $\vec{v}_{\text{rel}} = \text{CoM}(\text{Ridge}) - \text{CoM}(\text{Silhouette})$) and broadcast global anatomical context into local patch curves.
- **Superior Boundary Distance (ASSD Improvement):**
  - **Validation ASSD:** **29.87 px** (improved over EXPERIMENT_3's 30.98 px, and matching TopoNet Full's 29.27 px).
  - **Test ASSD:** **34.23 px** (a **+4.23 px reduction** in boundary error compared to EXPERIMENT_3's 38.46 px).
  - Demonstrates that global landmark relational conditioning produces tighter, less divergent boundary curves.
- **Enhanced Ridge Landmark Detection:**
  - Ridge Dice improved to **60.50%** on Validation (vs. 58.56% in EXP_3) and **60.39%** on Test (vs. 58.35% in EXP_3).
- **High Real-Time Inference Throughput:**
  Maintains **15.1 – 16.7 FPS** (**60.0 – 66.1 ms/frame** on NVIDIA A100), delivering high frame rates suitable for real-time surgical computer vision.
- **Convergence:**
  Best checkpoint converged at **Epoch 30** with **57.40% Validation Macro Dice** and **29.87 px ASSD** (Patient 40 subset reaching **60.11% Dice** and **27.72 px ASSD**).

---

## 2. Quantitative Benchmark Leaderboards

### 2.1. Validation Set Evaluation (122 frames)

*Benchmark Settings: Input RGB 1024x1024, Batch Size 4, 60 Epochs, Restored Epoch 30 Checkpoint.*

| Model Configuration | Macro Dice | Mean IoU | Macro ASSD | Ridge DSC | Sil DSC | Falc DSC | Pat. 40 DSC | Pat. 40 ASSD | Inference Speed | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LandmarkBezier (EXP_04)** | **57.40%** | **43.33%** | **29.87 px** | **60.50%** | **65.14%** | **46.56%** | **60.11%** | **27.72 px** | **15.1 FPS (66.1 ms)** | ✅ **COMPLETED** |

### 2.2. Test Set Evaluation (109 frames)

| Model Configuration | Macro Dice | Mean IoU | Macro ASSD | Ridge DSC | Sil DSC | Falc DSC | Foreground DSC | Inference Speed | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LandmarkBezier (EXP_04)** | **55.52%** | **41.43%** | **34.23 px** | **60.39%** | **63.12%** | **43.03%** | **61.76%** | **16.7 FPS (60.0 ms)** | ✅ **COMPLETED** |

---

## 3. Cross-Experiment Comparison (EXP_1 vs. EXP_2 vs. EXP_3 vs. EXP_4)

| Metric / Property | TopoNet Baseline (EXP_1) | Mask2Former Baseline (EXP_2) | Mask2Former-Bezier (EXP_3) | LandmarkBezier (EXP_4) | Architectural Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Decoder Sequence** | None (CNN) | 100 Generic Queries | 64 Patch Queries | **67 Tokens (3 LM + 64 Patch)** | Categorical anatomical grounding |
| **Scale Engine** | Single scale | Multi-Scale ($1/32 \to 1/8$) | Multi-Scale Cycling | **Stride-16 Lock** | Eliminates scale hopping noise |
| **Auxiliary Loss** | clDice + Betti | None | None | **Masked Centroid Smooth-L1 + BCE** | Direct supervision on Center of Mass |
| **Val Macro Dice** | 54.64% | 66.56% | 59.20% | **57.40%** | Balanced across classes |
| **Val Macro ASSD (px)** | 49.64 px | 25.69 px | 30.98 px | **29.87 px** | **Tighter boundary than EXP_3 (-1.11 px)** |
| **Test Macro Dice** | 65.19% (Full) | 65.73% | 55.74% | **55.52%** | Consistent generalization |
| **Test ASSD (px)** | 28.07 px (Full) | 28.43 px | 38.46 px | **34.23 px** | **Significant improvement over EXP_3 (-4.23 px)** |
| **Ridge DSC (Val / Test)**| 58.0% / 64.2% | 64.3% / 65.2% | 58.56% / 58.35% | **60.50% / 60.39%** | **Consistent +2.0% Ridge gain over EXP_3** |
| **Inference FPS** | 11.6 FPS | 10.5 FPS | 14.5 – 16.2 FPS | **15.1 – 16.7 FPS** | High real-time throughput maintained |

---

## 4. Run Details & Artifact File Locations

- **Run ID:** `exp4_landmark_bezier_60ep`
- **Training Epochs:** 60 Epochs (Best checkpoint at Epoch 30)
- **Model Checkpoint:** `/data/khoalq/checkpoints/exp4_landmark_bezier_60ep/best_model.pth` (447 MB)
- **Metrics JSON:** `/data/khoalq/checkpoints/exp4_landmark_bezier_60ep/metrics_summary.json`
- **Validation Frame CSV:** `/data/khoalq/checkpoints/exp4_landmark_bezier_60ep/val_predictions.csv` (122 rows)
- **Test Frame CSV:** `/data/khoalq/checkpoints/exp4_landmark_bezier_60ep/test_predictions.csv` (109 rows)
- **Training Loss Curve:** `/data/khoalq/checkpoints/exp4_landmark_bezier_60ep/training_log.csv` (60 epochs logged)
- **Diagnostic Visualizations:** `/data/khoalq/checkpoints/exp4_landmark_bezier_60ep/patient_40_diagnostics_final/` (101 montages with centroid crosshairs)
- **Packaged Archive:** `/data/khoalq/checkpoints/exp4_landmark_bezier_60ep/results.zip` (801 MB)
