# Experiment Manifest: EXP_12 — BCRNet Paper Replication (L3D Dataset)

## 1. Experiment Overview
- **Experiment ID:** `EXP_12_bcrnet_replication`
- **Objective:** Replicate the official MICCAI 2025 paper (*BCRNet: Enhancing Landmark Detection in Laparoscopic Liver Surgery via Bezier Curve Refinement*) on Kaggle for the L3D dataset across both **Val** and **Test** splits.
- **Reference Codebase:** `repos/BCRNet/` (maintained 100% clean and immutable).
- **Core Architecture:**
  - Multi-modal Feature Extraction: ResNet-50 (RGB + AdelaiDepth, 4 channels) + frozen SAM ViT-B image encoder (256x64x64 embeddings).
  - Adaptive Curve Proposal Initialization (ACPI): Dense 5th-order Bézier proposal generation on top feature pyramid layer.
  - Hierarchical Curve Refinement (HCR): 3-stage coarse-to-fine deformable cross-attention across feature pairs ({f3, f4} -> {f2, f3} -> {f1, f2}).
  - Bipartite Hungarian matching with 5th-order Bézier coordinates and 25 uniform curve interpolation points.

---

## 2. Target Benchmark Numbers (Paper Ground Truth)
L3D Dataset Test Set (from paper Table 1 & Table 3):
- **DSC (Dice Similarity Coefficient):** 69.57%
- **IoU (Intersection over Union):** 54.16%
- **ASSD (Average Symmetric Surface Distance):** 43.55 px
- **Margin vs previous SOTA (D2GPLand):** +5.53% DSC, +4.62% IoU, -17.25 px ASSD

---

## 3. Dataset Specifications & Modalities
The experiment processes 4 modalities per laparoscopic surgical frame:
1. **RGB Image:** 1024x1024 normalized image (.jpg).
2. **Monocular Depth:** Authentic AdelaiDepth depth maps from Kaggle `/kaggle/input/datasets/khoatrytopublish/l3d-*/.../depth_AdelaiDepth/` (.png).
3. **SAM Semantic Prior:** Meta SAM ViT-B (`sam_vit_b_01ec64.pth`) frozen image embeddings of shape (256, 64, 64) (.npy).
4. **Ground Truth Annotations:** 5th-order Bézier control points (6 points x 2 coords) and 30px rasterized line masks (.npz) generated via `repos/BCRNet/utils/preprocess.py`.

---

## 4. Training Schedule & Loss Formulation
- **Optimizer:** Adam, learning rate = 1e-5, weight decay = 1e-4
- **Batch Size:** 2 per GPU
- **Epochs:** 80
- **Dynamic Loss Weighting:**
  lambda(epoch) = 1 - sigmoid((epoch - 10) / 2)
  - `loss_ce_enc`: 1 - lambda(epoch)
  - `loss_pos_enc`: lambda(epoch)
  - `segmentation_loss`: lambda(epoch)
- **Checkpointing:** Checkpoint saved every 10 epochs (`epoch-10.pt`, ..., `epoch-80.pt`) and `best_model.pt` tracked on validation DSC.

---

## 5. Directory & File Structure
```
experiments/EXP_12_bcrnet_replication/
├── EXP_MANIFEST.md                      # Experiment design, formulas, and target metrics
├── Run_Commands.md                      # Copy-pasteable Kaggle execution commands
├── configs/
│   └── bcrnet_l3d.yaml                  # Model, dataset, and solver YAML configuration
├── scripts/
│   ├── setup_kaggle.sh                  # One-click dependency & CUDA extension installer
│   ├── build_adet_ext.py                # Standalone CUDA/C++ extension builder for MSDeformAttn
│   ├── prepare_data.py                  # Symlinks AdelaiDepth, labels, images, generates s_bezier & sam
│   ├── train_bcrnet.py                  # Training runner with BCRNet schedule and W&B logging
│   └── evaluate_bcrnet.py               # Comprehensive multi-split evaluation (DSC, IoU, ASSD)
└── results/                             # Evaluation metrics JSONs and visual prediction overlays
```
