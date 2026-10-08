# EXPERIMENT_14: Stratified Junction-Steered Mask2Former (Option 1 Re-partitioning)

## Executive Summary
* **Experiment ID:** `EXPERIMENT_14`
* **Parent Architecture:** `EXPERIMENT_5` (Junction-Steered Mask2Former)
* **Objective:** Test whether re-stratifying the training and validation splits to balance anatomical pose inversion (swapping `Patient_40` $\leftrightarrow$ `Patient_38`) enables the network to learn extreme grasper traction mechanics during training, and evaluate whether this improves out-of-distribution generalization on the frozen test benchmark (`Patient_41`).

---

## Dataset Re-Stratification (Option 1: 1-to-1 Patient Swap)

### 1. Motivation
In the original L3D benchmark:
- **Validation** was dominated by **Patient 40 (82.8% of Val)**, with 15 frames of extreme mechanical grasper elevation hoisting the anterior ridge to the upper boundary ($Y_{\text{ridge}} < 0.015$).
- **Training** was 92.0% canonical flat views, completely withholding Patient 40 from the model during training.
- Consequently, vanilla models collapsed on Patient 40 (28.1% DSC), and even junction-steered models exhibited their highest residual errors on Patient 40.
- Furthermore, the official **Test** benchmark is dominated by **Patient 41 (77.1% of Test)**, who also presents heavy surgical traction (34.5% inversion rate).

### 2. Swap Details
| Cohort | Direction | Total Frames | Inverted Frames | Inversion Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Patient 40** | **Val $\to$ Train** | 101 | 15 | 14.85% |
| **Patient 38** | **Train $\to$ Val** | 99 | 6 | 6.06% |
| **Patient 32** | Stays in Val | 15 | 4 | 26.67% |
| **Patient 18** | Stays in Val | 6 | 0 | 0.00% |

### 3. Population Balance Parity
| Split | Original Frames | Original Flipped (%) | **EXP_14 Frames** | **EXP_14 Flipped (%)** | Population Target |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 921 | 74 (8.03%) | **923** | **83 (8.99%)** | **8.92%** |
| **Val** | 122 | 19 (15.57%) | **120** | **10 (8.33%)** | **8.92%** |
| **Test (Frozen)** | 109 | — | **109** | **Frozen** | — |

---

## Technical Specifications
* **Backbone:** Swin-Tiny (`facebook/mask2former-swin-tiny-ade-semantic`), pretrained on ADE20K.
* **Junction Head:** 4 learned queries cross-attending to stride-16 features ($F_{16}$).
* **Steering Mechanism:** Gated query injection ($\gamma=0.1$) into Mask2Former object queries.
* **Loss Function:**
  `L_total = L_m2f + 5.0 * L_coord + 1.0 * L_vis`
* **Optimizer:** AdamW (`lr_backbone = 1e-5`, `lr_head = 1e-4`, `weight_decay = 1e-4`).
* **Scheduler:** CosineAnnealingLR (`epochs = 60`, `eta_min = 1e-6`).
* **Effective Batch Size:** 4
  - Standard/20GB Profile: `batch_size = 2`, `accum_steps = 2` (`run_server.sbatch`)
  - 10GB Profile: `batch_size = 1`, `accum_steps = 4` (`run_server_10gb.sbatch`, peak VRAM ~7.5 GB)

---

## Output Artifacts & Deliverables
* Checkpoint: `experiments/EXPERIMENT_14/results/best_model.pth`
* Training log: `experiments/EXPERIMENT_14/results/training_log.csv`
* Predictions: `val_predictions.csv` and `test_predictions.csv`
* Metrics: `metrics_summary.json` (with per-patient breakdowns)
* Archive: `results.zip`
