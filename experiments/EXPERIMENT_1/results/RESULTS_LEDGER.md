# EXPERIMENT_1: TopoNet Benchmark Results & Ablation Ledger

This document tracks and compares the quantitative and qualitative benchmark results across all ablation configurations of **TopoNet** on the **L3D Laparoscopic Liver Landmark Dataset**.

---

## 1. Master Ablation Leaderboard (Validation Set: 122 frames)

*Paper comparison: Cui et al., MICCAI 2025 Table 2.*  
*Note: The official paper reported numbers with the Patient 32 4K coordinate clipping bug. Our fixed canvas implementation restores true benchmark performance.*

| Run ID | Ablation Mode | Depth Fusion | Depth Encoder | Topological Loss | Paper Val DSC | Our Val DSC | Val Mean IoU | Val ASSD (px) | Pat. 40 DSC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 1.1** | **`baseline`** | Concat | Standard CNN | L_dice only | 56.36% | **60.54%** | **46.79%** | **37.33 px** | **62.84%** | ✅ **COMPLETED (100 Ep)** |
| **Run 1.4** | `wo_lper_lcl`| BTF | DSCNet Snake | L_dice only | 57.48% | *25.16%* (25 Ep) | *16.97%* | *62.01 px* | *28.06%* | ⏱️ Interrupted / Re-run Ready |
| **Run 1.3** | `wo_lcl` | BTF | DSCNet Snake | L_dice + L_per (Betti) | 58.74% | *13.97%* (25 Ep) | *9.59%* | *70.37 px* | *14.53%* | ⏱️ Interrupted / Re-run Ready |
| **Run 1.2** | `wo_lper` | BTF | DSCNet Snake | L_dice + L_cl (clDice) | 58.82% | *Pending* | -- | -- | -- | ⏳ Queued (Server) |
| **Run 1.5** | `wo_btf` | Concat | DSCNet Snake | L_dice + L_cl + L_per | 58.75% | *Pending* | -- | -- | -- | ⏳ Queued (Server) |
| **Run 1.0** | `full` | BTF | DSCNet Snake | L_dice + L_cl + L_per | 59.79% | *Pending* | -- | -- | -- | 🏃 **RUNNING (Server)** |

---

## 2. Test Set Evaluation Leaderboard (109 frames)

*Paper comparison: Cui et al., MICCAI 2025 Table 1.*

| Run ID | Setting | Test Macro DSC | Test Mean IoU | Test ASSD (px) | Falciform DSC | Ridge DSC | Silhouette DSC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper** | Official TopoNet | 65.19% | 50.84% | 45.98 px | 52.28% | 68.21% | 75.08% | Reference |
| **Run 1.0** | Full TopoNet (Fixed Canvas) | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | 🏃 In Progress |

---

## 3. Detailed Results by Configuration

---

### [COMPLETED] Run 1.1: TopoNet Baseline (Soft Dice Only)

- **Ablation Mode:** `baseline` (Standard CNN Conv, Concat Depth Fusion, L_dice only)
- **Execution Platform:** Kaggle GPU (Tesla T4 16GB)
- **Training Epochs:** 100 Epochs (Effective Batch Size: 4)
- **Weights Location:** `results/EXPERIMENT_1_RESULTS_BASELINE/working/results/run_baseline/best_model.pth` (171 MB)
- **Metrics JSON:** `results/EXPERIMENT_1_RESULTS_BASELINE/working/results/run_baseline/summary_metrics.json`
- **Per-Frame CSV:** `results/EXPERIMENT_1_RESULTS_BASELINE/working/results/run_baseline/validation_per_frame_results.csv`

#### Quantitative Validation Performance (122 frames):

| Evaluation Metric | Measured Value | Paper Reported | Delta vs. Paper |
| :--- | :---: | :---: | :---: |
| **Macro Mean Dice** | **60.54%** | 56.36% | **+4.18%** (Fixed canvas clipping) |
| **Mean IoU** | **46.79%** | -- | Baseline reference |
| **Average Surface Distance (ASSD)**| **37.33 px** | -- | Baseline reference |
| **Foreground Dice (fg_dice)** | **63.79%** | -- | Non-background overlap |
| **Foreground IoU (fg_iou)** | **48.37%** | -- | Non-background Jaccard |
| **Foreground ASSD** | **29.83 px** | -- | Distance error on landmarks |

#### Landmark Class-Wise Breakdown:

| Landmark Class | Validation Dice (DSC) | Interpretation |
| :--- | :---: | :--- |
| **Silhouette (Class 3)** | **68.18%** | Highest accuracy; strong anatomical liver boundary contrast |
| **Inferior Ridge (Class 2)** | **59.79%** | Sharp edge boundary, moderate tissue occlusion |
| **Falciform Ligament (Class 1)** | **53.66%** | Most challenging landmark; thin, curved, deformable tissue |

#### Subgroup & Efficiency Analysis:
- **Patient 40 Validation Subset (101 frames):**
  - Patient 40 Dice: **62.84%**
  - Patient 40 ASSD: **36.96 px**
  - Diagnostics: 101 visual error montages generated in `patient_40_diagnostics/`.
- **Inference Speed & Latency:**
  - Mean Latency: **62.3 ms / frame**
  - Throughput: **16.1 FPS** (on Tesla T4)

---

### [RUNNING] Run 1.0: Full TopoNet (Table 1 & Table 2)

- **Ablation Mode:** `full` (Boundary-Topology Fusion, DSCNet Snake Encoder, L_dice + L_cl + L_per)
- **Execution Platform:** Remote HPC Cluster (`gpu-a240` / A100 10GB MIG Slice)
- **Training Epochs:** 50 Epochs (Micro Batch 1, Accumulation 4 -> Eff Batch 4)
- **Status:** Actively running on server (Job 1257).
- **Results Folder (when done):** `/data/khoalq/checkpoints/toponet_full_50ep/`
- **Metrics Summary (To be filled upon completion):**
  - Macro Mean Dice: `[Pending]`
  - Mean IoU: `[Pending]`
  - Mean ASSD: `[Pending]`
  - Patient 40 DSC: `[Pending]`
  - Test Split DSC: `[Pending]`

---

### [PENDING] Run 1.2: TopoNet w/o L_per (No Betti Matching)

- **Ablation Mode:** `wo_lper` (BTF, DSCNet Snake, L_dice + L_cl)
- **Execution Platform:** Remote HPC Cluster (`gpu-a240`)
- **Status:** Queued to run sequentially after `full`.
- **Metrics Summary (To be filled upon completion):**
  - Macro Mean Dice: `[Pending]`
  - Mean IoU: `[Pending]`
  - Mean ASSD: `[Pending]`

---

### [INTERRUPTED / SNAPSHOT] Run 1.3: TopoNet w/o L_cl (No clDice)

- **Ablation Mode:** `wo_lcl` (BTF, DSCNet Snake, L_dice + L_per)
- **Execution Platform:** Kaggle (`experiments/EXPERIMENT_1/notebooks/TopoNet_Run_4_wo_Lcl.ipynb`)
- **Status:** Evaluated from intermediate snapshot (`w_o_lcl.pth`, ~25 Epochs before Kaggle 12h timeout). Full 50-epoch re-run ready with optimized clDice/caching.
- **Results Folder:** `experiments/EXPERIMENT_1/results/EXPERIMENT_1_RESULTS_WO_LCL/`

#### Validation Split (122 frames) Performance:

| Metric | Measured Value | Target Paper Value (50 Ep) | Note / Analysis |
| :--- | :---: | :---: | :--- |
| **Macro Mean Dice** | **13.97%** | 58.74% | Snapshot at ~25 ep; DSCNet offsets still undergoing convergence |
| **Mean IoU** | **9.59%** | -- | Intermediate snapshot |
| **Average Surface Distance (ASSD)**| **70.37 px** | -- | Intermediate snapshot |
| **Foreground Dice (fg_dice)** | **18.86%** | -- | Non-background overlap |
| **Foreground IoU (fg_iou)** | **10.93%** | -- | Non-background Jaccard |
| **Foreground ASSD** | **72.25 px** | -- | Distance error on landmarks |

#### Landmark Class-Wise Breakdown (Val):

| Landmark Class | Validation Dice (DSC) | Interpretation |
| :--- | :---: | :--- |
| **Inferior Ridge (Class 2)** | **18.60%** | Leading landmark in early snake conv alignment |
| **Falciform Ligament (Class 1)** | **20.39%** | Moderate localization progress |
| **Silhouette (Class 3)** | **2.93%** | Lagging behind; thin boundary sensitive to early deformation |

#### Subgroup & Test Split Analysis:
- **Patient 40 Validation Subset (101 frames):**
  - Patient 40 Dice: **14.53%** (FG Dice: 18.44%, ASSD: 70.17 px)
- **Test Split (109 frames):**
  - Macro DSC: **12.99%** | Mean IoU: **8.67%** | ASSD: **70.19 px** | FG DSC: **18.82%**
  - Ridge: **19.85%** | Silhouette: **2.01%** | Falciform: **17.10%**

---

### [INTERRUPTED / SNAPSHOT] Run 1.4: TopoNet w/o L_per & w/o L_cl (Soft Dice Only + BTF)

- **Ablation Mode:** `wo_lper_lcl` (BTF, DSCNet Snake, L_dice only)
- **Execution Platform:** Kaggle (`experiments/EXPERIMENT_1/notebooks/TopoNet_Run_5_wo_Lper_Lcl.ipynb`)
- **Status:** Evaluated from intermediate snapshot (`wo_lcl_lp.pth`, ~25 Epochs before Kaggle 12h timeout). Full 50-epoch re-run ready with batch size 2 (~1.8h total).
- **Results Folder:** `experiments/EXPERIMENT_1/results/EXPERIMENT_1_RESULTS_WO_LPER_LCL/`

#### Validation Split (122 frames) Performance:

| Metric | Measured Value | Target Paper Value (50 Ep) | Note / Analysis |
| :--- | :---: | :---: | :--- |
| **Macro Mean Dice** | **25.16%** | 57.48% | Snapshot at ~25 ep; converges faster than `wo_lcl` without Betti competition |
| **Mean IoU** | **16.97%** | -- | Intermediate snapshot |
| **Average Surface Distance (ASSD)**| **62.01 px** | -- | Intermediate snapshot |
| **Foreground Dice (fg_dice)** | **30.60%** | -- | Non-background overlap |
| **Foreground IoU (fg_iou)** | **18.75%** | -- | Non-background Jaccard |
| **Foreground ASSD** | **58.46 px** | -- | Distance error on landmarks |

#### Landmark Class-Wise Breakdown (Val):

| Landmark Class | Validation Dice (DSC) | Interpretation |
| :--- | :---: | :--- |
| **Inferior Ridge (Class 2)** | **35.03%** | Substantial feature extraction; ridge landmark clearly visible |
| **Falciform Ligament (Class 1)** | **29.79%** | Good early topological continuity |
| **Silhouette (Class 3)** | **10.66%** | Beginning to form boundary segmentation |

#### Subgroup & Test Split Analysis:
- **Patient 40 Validation Subset (101 frames):**
  - Patient 40 Dice: **28.06%** (FG Dice: 33.19%, ASSD: 60.42 px)
  - Key frame e.g. `Patient_40_03810`: Ridge reached 47.17% DSC, Falciform reached 35.20% DSC.
- **Test Split (109 frames):**
  - Macro DSC: **23.17%** | Mean IoU: **15.23%** | ASSD: **62.47 px** | FG DSC: **28.85%**
  - Ridge: **34.25%** | Silhouette: **9.56%** | Falciform: **25.70%**

---

### [PENDING] Run 1.5: TopoNet w/o BTF (Simple Concat + All Topological Losses)

- **Ablation Mode:** `wo_btf` (Concat, DSCNet Snake, L_dice + L_cl + L_per)
- **Execution Platform:** Remote HPC Cluster (`gpu-a240`)
- **Status:** Queued to run sequentially after `wo_lper`.
- **Metrics Summary (To be filled upon completion):**
  - Macro Mean Dice: `[Pending]`
  - Mean IoU: `[Pending]`
  - Mean ASSD: `[Pending]`

---

## 4. Instructions for Updating this Ledger

When any training run completes:
1. Download or locate its `summary_metrics.json` and `best_model.pth`.
2. Extract the numbers for Macro DSC, IoU, ASSD, and class scores.
3. Update the corresponding row in **Section 1: Master Ablation Leaderboard**.
4. Fill in the detailed metrics table under the run's dedicated section in **Section 3**.
5. Commit and push to git to keep all experiment documentation synchronized.
