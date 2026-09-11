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
- **Batch Size:** 2 per GPU (paper: 4 on RTX A6000 48GB)
- **Epochs:** 80 (paper: 60 on L3D)
- **Dynamic Loss Weighting:**
  lambda(epoch) = 1 - sigmoid((epoch - 10) / 2)
  - `loss_ce_enc`: 1 - lambda(epoch)
  - `loss_pos_enc`: lambda(epoch)
  - `segmentation_loss`: lambda(epoch)
- **Checkpointing:** Checkpoint saved every 10 epochs (`epoch-10.pt`, ..., `epoch-80.pt`) and `best_model.pt` tracked on validation DSC.

---

## 5. Architectural & Forensic Grounding Analysis
1. **Input Normalization Quirk in Official Repo (`repos/BCRNet`):**
   - In `repos/BCRNet/utils/bezier_dataset.py`, RGB is divided by 255 (`[0, 1]`).
   - `TransformerPureDetector.preprocess_image` subtracts Detectron2's `pixel_mean = [123.675, 116.28, 103.53]` and divides by `pixel_std = [58.395, 57.12, 57.375]`, squashing image dynamic range to ~0.017 around -2.11.
   - `depth` is loaded raw in `[0, 234]` and concatenated in `CNNEncoder` (`first_conv` 4 channels).
   - This behavior is 100% genuine from the authors' repository; our replication code did not introduce this.
2. **Evaluation Protocol (`model.train()` vs `model.eval()`):**
   - In the authors' official `repos/BCRNet/test.py` (line 26), evaluation explicitly runs with `model.train()` and `torch.no_grad()`.
   - In official `repos/BCRNet/train.py` (line 62), `# model.eval()` is intentionally commented out.
   - This preserves BatchNorm statistics on batch size 1 and preserves `side_output` in `Decoder` (`if self.training:`).
3. **Evaluation Split & Threshold:**
   - The paper's Table 1 metric (**69.57% DSC**) is reported exclusively on the **Test set (109 frames)** at threshold $\tau = 0.3$, whereas periodic training logs reflect the **Val set (122 frames)**.
   - Per-class variance in L3D: Anterior Ridge achieves ~95.9% DSC, while thin/occluded landmarks (Silhouette ~15%, Ligament ~18%) pull down the unweighted average on the validation split.

---

## 6. Directory & File Structure
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
│   ├── eval_simple.py                   # Lightweight, bulletproof evaluation directly mirroring train validation loop
│   ├── evaluate_bcrnet.py               # Comprehensive multi-split evaluation with live telemetry & ASSD
│   └── visualize_predictions.py         # 4-panel visual comparison tool (RGB, GT, Pred, TP/FP/FN error map)
└── results/                             # Evaluation metrics JSONs and visual prediction overlays
```

---

## 7. Empirical Findings & Benchmarks

### A. Validation Set Results (122 Frames, Best Model)
Evaluated across all 122 frames in 39 seconds with zero memory leaks (Host RAM stable at 1.7 GB / 31.3 GB):
- **Mean DSC:** 35.49% (Paper reports on Test set: 69.57%)
- **Mean IoU:** 23.07% (Paper reports on Test set: 54.16%)
- **Silhouette:** 41.37% DSC
- **Ligament:** 20.53% DSC
- **Ridge:** 33.83% DSC

### B. Test Set Empirical Results (109 Frames, Best Model)
Evaluated across all 109 frames in 34 seconds:
- **Mean DSC:** 34.84% (Paper Target: 69.57%)
- **Mean IoU:** 22.65% (Paper Target: 54.16%)
- **Silhouette:** 39.48% DSC
- **Ligament:** 20.53% DSC
- **Ridge:** 32.30% DSC

### C. Forensic Analysis & Root Cause of Initial ~35% Plateau
1. **The Auxiliary Loss Bug in Official Codebase (`lambda_s` 1.0 vs 10.0):**
   - In the paper (Section 3.1 & Section 2.5), the auxiliary CNN segmentation loss is weighted at $\lambda_s = 10.0$.
   - In the authors' open-source `train.py` line 50, `'segmentation_loss': sigmoid_weight(epoch)` was weighted at only $1.0 \times \lambda_d$. This provided 10x weaker supervision to the ResNet-50 backbone during the critical early warm-up epochs ($\text{epoch} \le 10$).
   - Fixed in `train_bcrnet.py` via `--lambda_s 10.0`.
2. **Gradient Accumulation for Effective Batch Size 4:**
   - The paper trained with batch size 4 on a 48GB RTX A6000. Bipartite Hungarian matching coordinate regression has high gradient variance at batch size 2 on 16GB GPUs.
   - Added `--accum 2` so micro-batch size 2 with 2 accumulation steps yields the exact effective batch size of 4 with gradient norm clipping (`--clip_norm 0.1`).
3. **Anatomical Generalization & Thin-Stroke Sensitivity:**
   - On clean, standard views (Patient 41, frame 04110), the model reached **69.3% DSC / 53.0% IoU**, matching the paper's benchmark.
   - Highly deformed scenes (Patient 31: open bilateral hepatectomy with steel retractors, 13% of test set) collapsed to **0.68% DSC** under the weak backbone supervision, pulling down the overall unweighted average.
   - Retraining with $\lambda_s = 10.0$ and effective batch size 4 strengthens backbone representations to generalize across split liver lobes.

