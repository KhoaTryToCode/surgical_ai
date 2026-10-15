# EXPERIMENT_13: Results Ledger

## 1. Historical Benchmark Comparison Table

All prior experiments evaluated on the canonical L3D benchmark (122 Val frames, 109 Test frames, 39 Patients).

| Experiment ID | Architecture / Strategy | Dataset Cohort | Val Macro Dice | Test Macro Dice | Patient 40 Dice | ASSD (px) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP_01** | Vanilla Mask2Former (Swin-T) | L3D (921 train) | 54.20% | 52.80% | 28.10% | 34.12 |
| **EXP_02** | Mask2Former Baseline Full | L3D (921 train) | 66.56% | 65.10% | 66.80% | 25.69 |
| **EXP_05** | Junction-Steered M2F | L3D (921 train) | **68.13%** | **66.86%** | **70.91%** | **19.54** |
| **EXP_10** | Depth-Junction-Steered M2F | L3D (921 train) | 67.52% | 66.24% | 69.76% | 20.12 |
| **EXP_11** | Curriculum Mining (Hard Patients) | L3D (921 train) | 67.89% | 66.51% | 70.45% | 19.88 |
| **EXP_12** | Dual-Decoder (L3D + CholecSeg8k) | L3D + Cholec | 66.47% | 65.31% | 69.39% | 21.05 |
| **EXP_13** | **Junction-Steered M2F (Replication)** | **L3D-2K (1,532 train)** | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* |

---

## 2. EXPERIMENT_13 Evaluation Ledger

### L3D-2K Validation Split (230 frames)
- **Macro Dice:** *[Pending run completion]*
- **Macro IoU:** *[Pending run completion]*
- **Macro ASSD:** *[Pending run completion]*
- **Ridge Line Dice:** *[Pending run completion]*
- **Silhouette Line Dice:** *[Pending run completion]*
- **Falciform Ligament Dice:** *[Pending run completion]*
- **Patient 40 Sub-cohort Dice:** *[Pending run completion]*
- **Mean Junction Error:** *[Pending run completion]*

### L3D-2K Test Split (238 frames)
- **Macro Dice:** *[Pending run completion]*
- **Macro IoU:** *[Pending run completion]*
- **Macro ASSD:** *[Pending run completion]*
- **Ridge Line Dice:** *[Pending run completion]*
- **Silhouette Line Dice:** *[Pending run completion]*
- **Falciform Ligament Dice:** *[Pending run completion]*

### Cross-Evaluation on Original L3D Test Split (109 frames)
*Direct head-to-head comparison against EXPERIMENT_5 (Baseline: 66.86% Macro Dice, 70.91% Patient 40 Dice)*
- **Original Test Macro Dice:** *[Pending run completion]*
- **Original Test Patient 40 Dice:** *[Pending run completion]*
- **Original Test ASSD:** *[Pending run completion]*

---

## 3. Run Artifacts & Outputs
- **Model Checkpoint:** `/data/khoalq/checkpoints/exp13_l3d2k_junction_m2f/best_model.pth`
- **Results Zip:** `/data/khoalq/checkpoints/exp13_l3d2k_junction_m2f/results.zip`
- **Training Log:** `training_log.csv`
- **Prediction Files:** `val_predictions.csv`, `test_predictions.csv`, `original_l3d_test_predictions.csv`
- **Metrics Summary:** `metrics_summary.json`
- **Diagnostic Visualizations:** `patient_40_diagnostics/`
