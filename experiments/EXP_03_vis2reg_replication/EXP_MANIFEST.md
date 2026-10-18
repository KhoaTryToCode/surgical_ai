# Experiment Manifest: EXP_03 Vis2Reg Replication

## 1. Metadata
- **Experiment ID:** `EXP_03_vis2reg_replication`
- **Paper Target:** MICCAI 2026 — *"Vis2Reg: Visibility-Aware Landmark-Free Geometric 3D–2D Registration for Liver Laparoscopy"* (arXiv:2607.17810)
- **Primary Goal:** Train and replicate landmark-free 3D-2D liver registration on the official P2I-LReg benchmark dataset.
- **Target Performance (Paper Table 1):**
  - Synthetic Rigid RRE <= 0.08 deg, RTE <= 0.26 mm
  - Real Deformable Silhouette Dice: ~91 - 93% (92.6% +- 7.2%)
  - Real Deformable One-Way Chamfer Distance: ~1.4 - 1.8 mm (1.43 +- 1.26 mm)

---

## 2. Architecture & Components
1. **Geometric Encoder:** Multi-scale DGCNN with EdgeConv graph feature propagation and dynamic kNN clustering (256-dim feature space).
2. **Correspondence Matcher:** GeoTransformer geometric self/cross-attention with dual-softmax / Sinkhorn optimal transport soft correspondence assignment.
3. **Rigid Solver:** Differentiable Weighted Singular Value Decomposition (SVD) yielding SE(3) rotation R in SO(3) and translation t in R^3.
4. **Deformation Field:** 5-layer SIREN implicit neural representation with sinusoidal activations and positional encoding skips (frozen in Stage 1, active in Stage 2).
5. **Differentiable Silhouette Renderer:** PyTorch3D points silhouette renderer with soft radius r = 0.01 for 2D overlap supervision.

---

## 3. Dataset Adapter (`data/dataset.py`)
- **P2ILRegFolds:** Deterministic 5-fold cross-validation partition across 21 patients (exactly 12 train, 4 val, 5 test per fold, 0 patient leakage).
- **P2ILRegSyntheticDataset:** 40,000 synthetic Blender simulation frames (up to 2,000 per patient), preoperative CT mesh (.obj scaled x1000 to mm), camera intrinsics (CAM_K.yml), ground-truth camera poses (camPose.yml), and adaptive world-to-cam orientation resolution.
- **P2ILRegRealDataset:** Real laparoscopic video keyframes, intraoperative surface point clouds (filtered Z > 0, removing background artifact points), scale factors (scale/*.txt), and 2D ground-truth liver masks (labels/*/label.png).
- **collate_p2i_lreg:** Universal batch collation supporting single-item dicts, batch sizes B >= 1, and PyTorch 2.6+ compliance.

---

## 4. Multi-Task Loss Objectives (`losses/objectives.py`)
- **Stage 1 (Synthetic Rigid):**
  - L_trans = SmoothL1(t_pred, t_gt)  (mm)
  - L_rot = (3 - Trace(R_pred @ R_gt^T)) / 2  (weight 10.0)
  - L_pts = Chamfer_oneway(warped_pts, target_pts)  (weight 0.5)
- **Stage 2 (Real Self-Supervised):**
  - L_3D = Chamfer_oneway(target_pts, warped_pts)  (weight 0.5)
  - L_vis = Chamfer_symmetric(visible_subset, target_pts)  (weight 1.2)
  - L_sil = BCE(pred_sil, gt_mask) + Dice(pred_sil, gt_mask)  (weight 1.0)
  - L_def = (1/N) * sum ||Delta_i||^2  (weight 0.1)
  - L_smooth = (1/N) * sum ||Delta_i - Delta_j||^2  (weight 0.1)
  - L_topo = (1/N) * sum (||w_i - w_j|| - ||p_i - p_j||)^2  (weight 0.3)

---

## 5. Execution & Verification Log
| Phase | Script / Milestone | Status | Key Metrics / Verification Result |
| :--- | :--- | :--- | :--- |
| **Phase 1** | `test_smoke.py` | **PASSED** | 2,119,438 parameters initialized; clean forward execution in rigid and deformable modes on Tesla T4 GPU. |
| **Phase 2** | `scripts/01_test_data.py` | **PASSED** | 5-fold stratification verified (12/4/5); unit conversion verified (CT diagonal 303.6 mm, intra-op Z in [10.4, 117.6] mm). |
| **Phase 3** | `scripts/02_test_losses.py` | **PASSED** | All 6 multi-task loss terms verified; 100% finite gradients across all 42 model parameter tensors. |
| **Phase 4** | `scripts/03_pretrain_syn.py` | **READY FOR KAGGLE** | Multi-batch EdgeConv + GeoTransformer + Weighted SVD pretraining; deformation field frozen; verified on local MPS. |
| **Phase 5** | `scripts/04_train_real.py` | *Pending* | Stage 2 real fine-tuning (20-epoch rigid warm-up -> 40-epoch joint deformable). |
| **Phase 6** | `scripts/05_evaluate.py` | *Pending* | 5-fold cross-validation evaluation and comparison against paper Table 1. |

---

## 6. Checkpoint Registry
- `checkpoints/stage1_rigid_seed.pth`: Stage 1 rigid matching weights (EdgeConv + GeoTransformer + SVD solver).
- `checkpoints/best_real_model.pth`: Stage 2 fine-tuned model checkpoint (upcoming).
