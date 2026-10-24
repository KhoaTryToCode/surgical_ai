# Experiment Manifest: EXPERIMENT_2 — Pure Mask2Former Component Ablation Study (Set A)

---

## 1. Research Context & Objective
- **Paper References:**
  - *Masked-attention Mask Transformer for Universal Image Segmentation (Mask2Former)* (Cheng et al., NeurIPS 2022)
  - *Topology-Constrained Learning for Efficient Laparoscopic Liver Landmark Detection (TopoNet)* (Cui et al., MICCAI 2025)
- **Primary Objective:**
  1. Faithfully replicate and benchmark pure standard Mask2Former on the L3D laparoscopic liver landmark dataset, establishing the authentic **~0.68 Macro Dice** control benchmark using the official pretrained Swin-Tiny model (`facebook/mask2former-swin-tiny-ade-semantic`).
  2. Dissect and ablate Mask2Former's three core internal architectural mechanisms (Set A Component Suite):
     - **Run 0: Control Baseline (`baseline`):** Full Pretrained Swin-Tiny + MSDeformAttn Pixel Decoder + Masked Cross-Attention + Query Self-Attention + Deep Supervision. Target: **~0.68 DSC**.
     - **Run 1: Spatial Gating Ablation (`wo_masked_attn`):** Pretrained Swin-Tiny with Full Global Cross-Attention (`attn_mask = None` across all 9 decoder layers). Isolates whether localized spatial gating filters out surgical glare and cautery smoke.
     - **Run 2: Scale Engine Ablation (`wo_multiscale`):** Pretrained Swin-Tiny with Single-Scale Feature Decoding (locked to stride-16 level_index=1). Isolates whether multi-scale feature pyramids are necessary to resolve thin 35-pixel boundaries.
     - **Run 3: Query Interaction Ablation (`wo_self_attn`):** Pretrained Swin-Tiny with Independent Parallel Queries (bypassing the $Q \cdot Q^T$ self-attention sublayer). Isolates whether inter-query communication is essential for coordinating continuous anatomical curve coverage.
  3. Deliver 4 self-contained, 1-click Kaggle runner notebooks with automated results archiving (`.zip`).
  4. Deliver an HPC Slurm cluster execution suite (`run_mask2former_suite.sbatch` and `run_mask2former_10gb.sbatch`) strictly adhering to the `/data/khoalq/` cluster layout on `gpu-a240`.
- **Status:** **4 Kaggle Notebooks Generated & AST-Verified** | **Slurm HPC Scripts Created & Verified**

---

## 2. Directory Structure
```
experiments/EXPERIMENT_2/
├── EXP_MANIFEST.md                       # Technical specification, benchmarks, and run ledger
├── Run_Commands.md                       # Slurm cluster execution, Kaggle guide, & monitoring commands
├── Mask2Former_Run_0_Baseline.ipynb      # Dedicated Kaggle runner: Run 0 (Baseline Control -> 0.68 DSC)
├── Mask2Former_Run_1_wo_MaskedAttn.ipynb # Dedicated Kaggle runner: Run 1 (Spatial Gating: w/o Masked Attn)
├── Mask2Former_Run_2_wo_MultiScale.ipynb # Dedicated Kaggle runner: Run 2 (Scale Engine: w/o Multi-Scale)
├── Mask2Former_Run_3_wo_SelfAttn.ipynb   # Dedicated Kaggle runner: Run 3 (Query Comm: w/o Query Self-Attn)
├── configs/
│   └── base.yaml                         # Global training parameters, split paths, and mode specs
├── models/
│   ├── __init__.py
│   └── mask2former_ablation.py           # Unified Mask2Former module
├── utils/
│   ├── __init__.py
│   ├── dataset.py                        # RGB-Only L3D Dataset with Patient 32 4K canvas fix
│   └── metrics.py                        # Macro Dice, IoU, ASSD (matching EXPERIMENT_1)
├── scripts/
│   ├── __init__.py
│   ├── run_mask2former_suite.sbatch      # Dedicated Slurm runner for 40GB A100 GPU
│   ├── run_mask2former_10gb.sbatch       # Dedicated Slurm runner for 10GB MIG GPU slice
│   ├── train_server.py                   # Cluster & server training runner (Batch size 4, AMP, multi-worker)
│   ├── run_server_all_ablations.sh       # Fallback shell runner for interactive GPU execution
│   ├── train_mask2former.py              # Micro-batch 1, Accum 4, AMP FP16, Patient 40 panels, Zip bundle
│   ├── smoke_test_local.py               # Local verification script (macOS MPS/CPU)
│   └── generate_notebooks.py             # Notebook compilation generator with AST verification
├── notebooks/                            # Mirrored notebook storage
└── results/                              # Metric summaries, logs, weights, and Patient 40 diagnostics
```

---

## 3. The 4 Set A Ablation Configurations

| Run ID | Mode Name | Spatial Attention | Scale Engine | Query Self-Attn | Dedicated Kaggle Notebook | Output Archive | Target Dice |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **Run 0** | `baseline` | Masked ($M_{l-1}$) | Multi-Scale ($1/32 \to 1/8$) | Active ($Q \cdot Q^T$) | `Mask2Former_Run_0_Baseline.ipynb` | `EXPERIMENT_2_RESULTS_RUN_0_BASELINE.zip` | **~0.68** |
| **Run 1** | `wo_masked_attn` | Global ($H \times W$) | Multi-Scale ($1/32 \to 1/8$) | Active ($Q \cdot Q^T$) | `Mask2Former_Run_1_wo_MaskedAttn.ipynb` | `EXPERIMENT_2_RESULTS_RUN_1_WO_MASKED_ATTN.zip` | Delta |
| **Run 2** | `wo_multiscale` | Masked ($M_{l-1}$) | Single-Scale (stride 16) | Active ($Q \cdot Q^T$) | `Mask2Former_Run_2_wo_MultiScale.ipynb` | `EXPERIMENT_2_RESULTS_RUN_2_WO_MULTISCALE.zip` | Delta |
| **Run 3** | `wo_self_attn` | Masked ($M_{l-1}$) | Multi-Scale ($1/32 \to 1/8$) | Bypassed (Identity) | `Mask2Former_Run_3_wo_SelfAttn.ipynb` | `EXPERIMENT_2_RESULTS_RUN_3_WO_SELF_ATTN.zip` | Delta |

---

## 4. Benchmark Parity Standards with EXPERIMENT_1

To ensure strict 1:1 scientific comparison with EXPERIMENT_1 (TopoNet replication):
1. **Dataset Splits:** Exact L3D splits: Train (921 frames), Val (122 frames), Test (109 frames).
2. **Patient 32 4K Canvas Preservation:** Dynamic reading of `imageHeight` and `imageWidth` from JSON metadata prevents coordinate clipping on 4K frames.
3. **RGB-Only Standard Pipeline:** Pure 3-channel input `(3, 1024, 1024)`. No depth images, no depth encoders, zero multimodal confounding.
4. **Standardized Thickness 35:** All landmark ground truth maps drawn with thickness 35 on the native canvas before resizing to 1024x1024.
5. **Loss Formulation:** Deep-supervised Hungarian matching loss across all 9 decoder layers:
   `Cost(i, j) = 2.0 * CE + 5.0 * BCE + 5.0 * Dice`
6. **Inference Formulation:** Multi-query ensemble projection over all 100 queries:
   `S(c, h, w) = sum_{q=1}^Q p_q(c) * sigmoid(m_q(h, w))`
7. **Failure Diagnostics:** Automated 4-panel visual reporting on Patient 40 frames (`RGB | GT | Pred | Error Map`).
8. **One-Click Kaggle Download:** Automatic creation of a consolidated `.zip` archive at the end of execution.

---

## 5. Completed Benchmark Results (Set A Ablation Suite)

*Centralized Ledger:* [`results/RESULTS_LEDGER.md`](results/RESULTS_LEDGER.md)

### Validation Leaderboard (122 frames):
| Run ID | Mode Name | Val Macro DSC | Val Mean IoU | Val ASSD | Pat. 40 DSC | Pat. 40 ASSD | FG Union DSC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 2.0** | **`baseline`** | **66.56%** | **53.63%** | **25.69 px** | **68.88%** | **23.96 px** | **69.76%** | ✅ **COMPLETED** |
| **Run 2.1** | **`wo_masked_attn`** | **66.17%** | **53.18%** | **23.45 px** | **68.49%** | **19.95 px** | **69.69%** | ✅ **COMPLETED** |
| **Run 2.3** | **`wo_self_attn`** | **66.34%** | **53.12%** | **22.54 px** | **68.63%** | **18.73 px** | **69.48%** | ✅ **COMPLETED** |
| **Run 2.2** | **`wo_multiscale`** | **65.35%** | **52.22%** | **35.77 px** | **67.84%** | **31.99 px** | **68.93%** | ✅ **COMPLETED** |

### Test Benchmark vs Published SOTA (109 frames):
| Method / Model | Modality | Test Macro DSC | Test Mean IoU | Test ASSD | Ridge DSC | Silhouette DSC | Falciform DSC | FG Union DSC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TopoNet (Published Table 1) | RGB-D | 65.19% | 50.56% | 28.07 px | 68.21% | 75.08% | 52.28% | -- |
| **M2F Run 2.0 (`baseline`)** | **RGB-Only** | **65.73%** | **53.05%** | **28.43 px** | **67.69%** | **71.24%** | **58.24%** | **69.59%** |
| **M2F Run 2.1 (`wo_masked_attn`)**| **RGB-Only** | **65.68%** | **52.94%** | **26.86 px** | **67.11%** | **72.25%** | **57.68%** | **69.22%** |
| **M2F Run 2.2 (`wo_multiscale`)** | **RGB-Only** | **64.58%** | **51.54%** | **36.65 px** | **65.01%** | **71.05%** | **57.67%** | **68.28%** |
| **M2F Run 2.3 (`wo_self_attn`)** | **RGB-Only** | **64.12%** | **51.05%** | **24.02 px** | **66.29%** | **70.44%** | **55.63%** | **68.04%** |
