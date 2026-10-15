# Experiment Manifest — EXPERIMENT 17: MinVIS (Minimal Video Instance Segmentation)

## 1. Metadata
* **Experiment ID**: `EXPERIMENT_17_minvis`
* **Date**: October 15, 2026
* **Objective**: Evaluate NVIDIA MinVIS (NeurIPS 2022) as the first baseline in our video landmark segmentation series across the canonical 5-fold patient-disjoint cross-validation split of L3D.
* **Pretrained Weights**: `facebook/mask2former-swin-tiny-ade-semantic` (identical to EXP 01 and EXP 15).
* **Evaluation Standard**: Exact TopoNet argmax mutual exclusivity at threshold 0.5 (Macro/Micro Dice, ASSD, HD95).

---

## 2. Hypothesis & Architectural Principles
1. **Hypothesis**: Query-based image Mask2Former produces semantically consistent instance query embeddings across video frames. Tracking these queries temporally via bipartite cosine Hungarian matching will stabilize anatomical landmark boundaries across severe view shifts without requiring video-level 3D layers.
2. **Zero Forward-Pass Leakage**: MinVIS processes every video frame completely independently through the 2D network. Temporal tracking occurs strictly in post-processing via bipartite matching of query vectors ($E_t \in \mathbb{R}^{100 \times 256}$).
3. **Template Standardization**: Serves as the static code template (dataset rasterization, evaluation metrics, result aggregation) for subsequent video architectures (Cutie/XMem, SAM 2).

---

## 3. Dataset Configuration & Kaggle Portability
* **Split Configuration**: `_config/5fold_patient_splits.json` (Locked SHA-256: `18d76bcd...`).
* **Train + Val Pool**: 35 patients (1,043 frames) partitioned into 5 folds (7 val patients, 28 train patients per fold).
* **Frozen Test Set**: Patients 21, 31, 41, 51 (109 frames) — evaluated after training to benchmark against TopoNet.
* **Kaggle Multi-Dataset Mount Parity (from EXP 15)**: On Kaggle, L3D is mounted as three distinct sibling datasets (`l3d-train`, `l3d-val`, `l3d-test`). The resolver dynamically resolves all three candidate roots (`resolve_all_split_dirs`) and scans across splits case-insensitively (`labels`, `images`, `.png`, `.jpg`). Because cross-validation partitions pool patients across the original `Train` and `Val` directories, searching across all resolved split directories guarantees zero missing frames and 100% coverage.
* **Rasterization Parity**: 35px native canvas stroke thickness, `cv2.INTER_NEAREST` downsampling to 1024 x 1024.

---

## 4. Hyperparameters
| Parameter | Value |
| :--- | :--- |
| **Backbone** | Swin-Tiny (ImageNet-1K + ADE20K pretraining) |
| **Decoder** | 9-layer Multi-Scale Masked Attention Decoder ($Q=100$ queries) |
| **Epochs** | 40 epochs |
| **Batch Size** | 4 |
| **Optimizer** | AdamW |
| **Learning Rates** | Decoder: $1.0 \times 10^{-4}$ \| Backbone: $1.0 \times 10^{-5}$ |
| **Weight Decay** | 0.05 |
| **LR Scheduler** | CosineAnnealingLR ($\eta_{\min} = 1.0 \times 10^{-6}$) |
| **Mixed Precision** | `torch.cuda.amp.autocast(dtype=torch.float16)` |

---

## 5. Result Deliverables
* Checkpoints: `checkpoints/best_minvis_fold{0..4}.pth`
* Per-fold Results: `results/results_fold_{0..4}.json`
* Frozen Test Results: `results/test_results_fold_{0..4}.json`
* Master Summary: `results/EXP17_5Fold_Summary.csv` and `results/EXP17_5Fold_Summary.md`
