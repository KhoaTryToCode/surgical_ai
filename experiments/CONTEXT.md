# Surgical AI Experiment Context & Handover Guide

> **Target Audience:** Future AI agents and researchers picking up this codebase.  
> **Purpose:** Document the entire workflow, decisions, key findings, and—most importantly—**every mistake made, bug diagnosed, and trap to avoid** to ensure seamless continuation with zero repeated errors.

---

## 1. Project Overview & Current State

We are working on **Laparoscopic Liver Landmark Detection** on the **L3D Benchmark Dataset** (1,152 keyframes: 921 Train / 122 Val / 109 Test).  
Target landmarks:
- **Class 1: Anterior Liver Ridge** (`r` / `ridge` / `rigde`)
- **Class 2: Liver Silhouette** (`s` / `silhouette`)
- **Class 3: Falciform Ligament** (`f` / `falciform`)
- **Class 0: Background**

### Experiment Generations:
1. **EXPERIMENT_1 (`experiments/EXPERIMENT_1/`):**  
   TopoNet Replication & Topological Loss Ablations (RGB-D dual-stream, Snake-CNN, BTF module, Centerline $L_{cl}$, Betti persistence $L_{per}$).
2. **EXPERIMENT_2 (`experiments/EXPERIMENT_2/`):**  
   Pure Mask2Former Swin-Tiny Component Ablations (Pure RGB-only, no depth estimator). Dissects the 3 core internal mechanisms:
   - **Run 0 (`baseline`):** Full Pretrained Swin-Tiny + MSDeformAttn Pixel Decoder + Masked Cross-Attention + Query Self-Attention.
   - **Run 1 (`wo_masked_attn`):** Spatial Gating ablation (Global Cross-Attention, `attn_mask = None`).
   - **Run 2 (`wo_multiscale`):** Scale Engine ablation (Single-Scale stride-16 decoding).
   - **Run 3 (`wo_self_attn`):** Query Interaction ablation (Independent parallel queries, bypassing $Q \cdot Q^T$).

**Current Status:** All 4 EXPERIMENT_2 ablations are **100% completed, evaluated, archived, and cross-compared**. Full TopoNet (`Run 1.0`) in EXPERIMENT_1 is actively running on the remote cluster (`gpu-a240`).

---

## 2. Summary Benchmark Numbers

Master reference tables are stored at [`experiments/RESULTS.md`](RESULTS.md):

### Validation Set (122 frames):
| Model Architecture | Macro Dice | Mean IoU | ASSD | Inference Speed |
| :--- | :---: | :---: | :---: | :---: |
| TopoNet Baseline (Paper) | 54.64% | 40.94% | 49.64 px | ~86.4 ms |
| TopoNet w/o L_per (Paper) | 58.87% | 46.69% | 38.66 px | ~86.4 ms |
| TopoNet w/o L_cl (Paper) | 58.63% | 46.37% | 32.01 px | ~86.4 ms |
| TopoNet w/o L_per & L_cl (Paper) | 57.44% | 45.86% | 43.32 px | ~86.4 ms |
| TopoNet w/o BTF (Paper) | 57.92% | 46.03% | 37.47 px | ~86.4 ms |
| TopoNet Full (Paper) | 59.79% | 47.38% | 29.27 px | 11.6 FPS (86.4 ms) |
| **Mask2Former Baseline** | **66.56%** | **53.63%** | **25.69 px** | **10.5 FPS (94.9 ms)** |
| **Mask2Former Full Attention** | **66.17%** | **53.18%** | **23.45 px** | **10.6 FPS (94.1 ms)** |
| **Mask2Former Single Scale** | **65.35%** | **52.22%** | **35.77 px** | **10.6 FPS (94.1 ms)** |
| **Mask2Former w/o Self Query** | **66.34%** | **53.12%** | **22.54 px** | **10.6 FPS (94.5 ms)** |

### Test Set (109 frames):
| Model Architecture | Macro Dice | Mean IoU | ASSD | Inference Speed |
| :--- | :---: | :---: | :---: | :---: |
| TopoNet Full (Paper SOTA) | 65.19% | 50.56% | 28.07 px | 11.6 FPS (86.4 ms) |
| **Mask2Former Baseline** | **65.73%** | **53.05%** | **28.43 px** | **10.6 FPS (94.4 ms)** |
| **Mask2Former Full Attention** | **65.68%** | **52.94%** | **26.86 px** | **10.7 FPS (93.3 ms)** |
| **Mask2Former Single Scale** | **64.58%** | **51.54%** | **36.65 px** | **10.7 FPS (93.8 ms)** |
| **Mask2Former w/o Self Query** | **64.12%** | **51.05%** | **24.02 px** | **10.6 FPS (93.9 ms)** |

---

## 3. Critical Bugs & Mistakes Made in This Session (AND HOW TO AVOID THEM)

Pay close attention to these 7 traps. **Do not repeat them:**

### Trap 1: Annotation Typo in the L3D Dataset (`"rigde"` vs `"ridge"`)
- **What happened:** In the standalone post-hoc inference script, Ridge Dice suddenly crashed to `4.85e-10` (~0.0), pulling Patient 40 Dice from 68% down to 45%.
- **Root Cause:** The raw L3D JSON annotation files have human typos—annotators misspelled Ridge as `"rigde"`.
- **How to fix / Golden Rule:** NEVER use strict string matching like `if label == 'ridge'`. ALWAYS use the typo-tolerant matcher:
  ```python
  if label.startswith('r') or 'ridge' in label.lower() or 'rigde' in label.lower():
      color = 1  # Ridge
  elif label.startswith('s') or 'sil' in label.lower():
      color = 2  # Silhouette
  elif label.startswith('f') or 'falc' in label.lower():
      color = 3  # Falciform
  ```

### Trap 2: String Formatting Bug in Patient 40 Logging
- **What happened:** Training logs displayed `P40 Dice: 0.0000` across all 60 epochs, causing panic that Patient 40 was completely failing.
- **Root Cause:** The script checked `if 'patient40' in filename.lower():`. But L3D files are named `Patient_40_XXXXX.jpg` (with an underscore), so `patient40` never matched!
- **Fact:** Patient 40 makes up **101 of the 122 validation frames (82.8%)**. The model was actually getting 68.5%+ on Patient 40.
- **How to fix:** Always check:
  ```python
  if 'patient_40' in filename.lower() or '_40_' in filename.lower():
  ```

### Trap 3: Mask2Former Attention Mask Shape Mismatch on Single-Scale (`wo_multiscale`)
- **What happened:** Execution crashed with:  
  `RuntimeError: The shape of the 3D attn_mask is torch.Size([8, 100, 144]), but should be (8, 100, 576)`
- **Root Cause:** In Mask2Former, the Transformer decoder expects multi-scale features `[stride_32, stride_16, stride_8]`. If you pass a list of length 1 `[stride_16]`, the internal layer-level cycling (`level_idx = layer_idx % len(features)`) mismatches the spatial dimensions expected by the pixel decoder's attention mask generator.
- **How to fix:** When ablating multi-scale, repeat the chosen stride-16 tensor across the 3 feature slots:
  ```python
  features = [stride_16_feat, stride_16_feat, stride_16_feat]
  ```

### Trap 4: NumPy 2.0 / Kaggle Python 3.12 Incompatibility (`np.Inf` Removed)
- **What happened:** The evaluation script crashed during ASSD calculation:  
  `AttributeError: 'np.Inf' was removed in NumPy 2.0. Use 'np.inf' instead.`
- **Root Cause:** DeepMind's `surface_distance` library was written for NumPy 1.x and imports `np.Inf`. Kaggle now runs Python 3.12 with NumPy 2.x.
- **How to fix:** Inject a monkeypatch at the very top of your evaluation scripts before importing `surface_distance`:
  ```python
  import numpy as np
  if not hasattr(np, 'Inf'):
      np.Inf = np.inf
      np.PINF = np.inf
      np.NINF = -np.inf
  ```
  Also wrap ASSD calls in `try ... except` with an 80.0 px fallback if masks are completely empty.

### Trap 5: Discrepancy Between Output Deliverables (Missing EXP_1 Parity)
- **What happened:** The user downloaded the Kaggle output zip for Run 2, but it was missing `patient_40_diagnostics/` image montages and per-frame CSVs, breaking parity with Experiment 1.
- **Root Cause:** To save GPU memory during 60-epoch training, visualization generation was commented out, leaving no automated way for the user to get the visual assets.
- **How to fix / Standard Operating Procedure:**
  Every experiment run MUST produce the complete deliverable quartet inside its result directory:
  1. `best_model.pth` (Model checkpoint)
  2. `summary_metrics.json` (Structured JSON metrics for Val & Test)
  3. `val_per_frame_results.csv` and `test_per_frame_results.csv` (Per-frame CSV breakdowns)
  4. `patient_40_diagnostics/` (101 visual error montages specifically for Patient 40; note that the entire validation split contains 122 frames: 101 from Patient 40, 15 from Patient 32, and 6 from Patient 18)

### Trap 6: Understanding Which Dice is Reported (Paper vs. Logs)
- **The Three Flavors of Dice:**
  1. **Macro Dice (`macro_dice`):** Unweighted class average: `(Ridge + Sil + Falc) / 3`. **This is the official metric reported in the TopoNet and D2GPLand papers** (Table 1: 65.19% DSC).
  2. **Foreground Union Dice (`fg_dice`):** Binary landmark union: `(pred > 0) vs (gt > 0)`. Measures geometric localization, ignoring class mix-ups.
  3. **Pooled Multi-Class Dice (`pred[1:].flatten()`):** Used in TopoNet's official `test.py`. Pools all foreground pixels into one numerator/denominator, heavily weighted toward long lines (Silhouette & Ridge: ~74% and ~68%), masking Falciform's lower score (~57%).
- **Why Val Macro Dice (66.2%) is lower than Patient 40 Dice (68.5%):**
  Patient 40 is high quality. The validation split also contains **Patient 32** (15 frames recorded with a 4K camera with extreme glare and viewpoint distortion), where all models score only ~40% Dice.

### Trap 7: Tool Execution Error with `write_to_file` and `ArtifactMetadata`
- **What happened:** `write_to_file` failed with: `invalid tool call error: ... is not a valid artifact path; artifacts must be in ~/.gemini/antigravity/brain/...`
- **Root Cause:** `ArtifactMetadata` is ONLY for internal brainstorm artifacts inside the brain directory. Never pass `ArtifactMetadata` when writing real code or markdown files into the workspace (`/Users/khoale/...`).

---

## 4. Architectural Findings & Scientific Insights

1. **Pretrained Swin-Tiny is the True Differentiator:**  
   Pure RGB Mask2Former beats multi-modal TopoNet on the test set (**65.73% vs 65.19%**) without depth maps, proving that rich pretrained Vision Transformer features outperform shallow CNNs trained from scratch on 921 frames.
2. **Multi-Scale Feature Engine is the Boundary Anchor:**  
   When single-scale stride-16 was used (`wo_multiscale`), **ASSD error exploded from 28.43 px to 36.65 px (+8.22 px boundary error)**. Stride-8 is indispensable for resolving thin 35-pixel curves.
3. **Query Self-Attention Prevents Test Generalization Collapse:**  
   Bypassing query self-attention (`wo_self_attn`) caused the worst test score across the suite (**64.12% Macro DSC**, -1.61% drop vs baseline), proving that latent inter-query attention ($Q \cdot Q^T$) acts as essential non-maximum suppression across unseen patients.
4. **Hard Case Anomaly (`08730`–`09000`):**  
   Under severe organ deformation, rapid laparoscope motion blur, and cautery smoke, `wo_multiscale` (locked to stride-16) achieved **higher Foreground Union Dice (40.4% vs 31.9%)** than multi-scale baseline. High-resolution stride-8 is vulnerable to high-frequency motion blur and glare edges, whereas stride-16 acts as a stable low-pass geometric tracker.

---

## 5. File & Directory Registry

- **Master Comparison Tables:** [`experiments/RESULTS.md`](RESULTS.md)
- **Detailed EXP_2 Ledger:** [`experiments/EXPERIMENT_2/results/RESULTS_LEDGER.md`](EXPERIMENT_2/results/RESULTS_LEDGER.md)
- **Detailed EXP_1 Ledger:** [`experiments/EXPERIMENT_1/results/RESULTS_LEDGER.md`](EXPERIMENT_1/results/RESULTS_LEDGER.md)
- **Interactive Visualizer (HTML):** [`experiments/patient_40_diagnostics_comparison.html`](patient_40_diagnostics_comparison.html)
- **Kaggle Notebooks (EXP_2):** `experiments/EXPERIMENT_2/Mask2Former_Run_*.ipynb`
- **HPC Cluster Slurm Scripts:** `experiments/EXPERIMENT_2/scripts/run_mask2former_*.sbatch`
- **Cluster Code Location:** `/data/khoalq/surgical_ai/` on `gpu-a240` (`100.82.42.48`)
- **Environment** Everything should be run under the venv surgical_ai on both the local and the server machine
- **Update result** Whenever you have a new experiment, you only do 2 things to log the result. First is that you only add 2 rows into the RESULT.md in the experiments folder, do not change anything else, and then you create a RESULT_LEDGER for that experiment. Do not change any log or any findings that I wrote in the RESULT.md

---

## 6. Checklist for the Next Agent

Before running or touching code:
- [ ] Check if the task involves modifying files in `/repos/`. **NEVER touch `/repos/`** (strictly immutable).
- [ ] Check label parsing: Are you using the typo-tolerant matcher for `"rigde"`?
- [ ] Check canvas resolution: Are you reading `imageHeight` and `imageWidth` from JSON metadata to prevent Patient 32 4K coordinate clipping?
- [ ] If running evaluation on Kaggle/NumPy 2.x, did you include the `np.Inf = np.inf` monkeypatch?
- [ ] Are you outputting the complete 4-file artifact deliverable (`.pth`, `summary_metrics.json`, CSVs, and `patient_40_diagnostics/`)?

---

## 7. EXPERIMENT_5 & EXPERIMENT_6 Generations (Chronicle & Discoveries)

### EXPERIMENT_5 (`experiments/EXPERIMENT_5/`): Junction-Steered Mask2Former
- **Architecture:** 4 learnable anatomical queries ($J_{\text{top}}, J_{\text{bottom}}, J_{\text{lat\_r}}, J_{\text{lat\_l}}$) attend to stride-16 features, then steer 100 Mask2Former queries via cross-attention with gated residual update:
  `Q_steered = LayerNorm(Q + alpha * delta_Q)`
- **Supervision:** 2-layer MLP predicting $(x, y) \in [0, 1]^2$ coordinates with Smooth-L1 loss + binary visibility BCE.
- **Benchmark Results:**
  - **Test Macro Dice:** **69.84%** (surpassing published BCRNet SOTA of 69.57% and Mask2Former baseline 65.73%).
  - **Val Macro Dice:** **71.98%** (Patient 40 Val: **72.61%**).
- **Key Flaw Diagnosed (Geometric Probing):**
  Linear probing (`analyze_j_geometry.py`) proved $J$ vectors strongly capture organ translation ($R^2 = 0.857$ for Y-centroid, $0.648$ for X-centroid) and scale ($R^2 = 0.741$), but the MLP coordinate head forced hallucination of $(x, y)$ coordinates when landmarks were absent or occluded, resulting in **147.4 px mean test landmark error** and phantom hot-spots.

### EXPERIMENT_6 (`experiments/EXPERIMENT_6/`): Heatmap-Guided Junction-Steered Mask2Former
- **Architecture:** Keeps `delta_q` steering 100% intact. Replaces the decoupled MLP coordinate head with **2D continuous spatial sigmoid heatmaps ($4 \times 64 \times 64$)** via dynamic spatial dot product:
  `dots = (J_proj · F_proj) / sqrt(d) + bias_prior`
  `pred_heatmaps = sigmoid(dots)`
- **Supervision:** CenterNet-style Gaussian Focal Loss:
  - Visible landmark ($v_k = 1$): 2D Gaussian peak ($\sigma = 2.0\text{ px}$, peak = $1.0$).
  - Absent landmark ($v_k = 0$): Clean all-zero plane ($0.0$), penalizing ghost activations and cleanly eliminating coordinate hallucination.
- **Benchmark Results:**
  - **Test Macro Dice:** **69.84%** (Tied SOTA, beats BCRNet 69.57%).
  - **Val Macro Dice:** **71.51%** (Patient 40: **72.06%**).
  - **Landmark Localization Error:** **81.57 px** on test set (**-44.7% reduction** from 147.42 px in EXP_5!).
  - **Convergence:** Best model saved at **Epoch 20** (converged 2x faster than EXP_5's Epoch 40).

---

## 8. New Critical Traps, Mistakes Made & Rules (EXP_5 & EXP_6)

### Trap 8: NEVER Modify, Delete, or Overwrite the User's Personal Log in `RESULTS.md`
- **What happened:** When adding EXP_5 benchmark numbers, the AI overwrote the user's handwritten log under `September 19, 2026`. The user had to specifically request restoring it.
- **Golden Rule:** When updating `experiments/RESULTS.md` for a new experiment:
  1. **ONLY add exactly 2 rows to the tables** (one row in Table 1 Validation, one row in Table 2 Test).
  2. **DO NOT touch, edit, or rephrase ANY text, notes, or dates written by the user.**
  3. All detailed experimental breakdowns and findings belong in `experiments/<EXP_ID>/RESULTS_LEDGER.md`, NOT in the main notes of `RESULTS.md`.

### Trap 9: Focal Loss Prior Bias in Dense Heatmap Heads
- **What happened:** In `DynamicHeatmapHead`, the dot product `(J_proj · F_proj) / sqrt(d)` had zero mean at initialization, so `sigmoid(0) = 0.5` across all 16,384 pixels ($4 \times 64 \times 64$). Summing negative focal loss over 16,384 pixels produced an initial loss of **~4,139**, creating a severe gradient explosion risk that could disrupt pretrained Swin weights.
- **How to fix / Golden Rule:** Follow RetinaNet and CenterNet: always initialize the bias of dense focal heatmap layers with $b = -2.19$ (or $-2.5$), such that $\text{sigmoid}(b) \approx 0.10$.
  ```python
  self.bias = nn.Parameter(torch.tensor(-2.19, dtype=torch.float32))
  dots = dots + self.bias
  ```
  With this bias and $\lambda_{\text{heatmap}} = 1.0$, the heatmap loss starts at balanced scale (~3.0 to 6.0), harmoniously matching Mask2Former loss (~2.5).

### Trap 10: Binary Overlap vs. Multi-Class Semantic Error Maps
- **What happened:** In `evaluate.py`, Panel 4 (Error Map) was implemented as:
  `tp = (pred_map > 0) & (gt_mask > 0)`
  On `Patient_40_08790`, the model predicted Silhouette (Green) where Ridge (Red in GT) was supposed to be. Panel 4 painted the boundary Green (TP), making it look like a correct prediction and confusing the user.
- **Clarification:** The quantitative metrics (`macro_dice` in `metrics.py`) are strictly class-aware and correctly caught the error (scoring 40.1% Dice). The flaw was purely cosmetic in the JPEG rendering.
- **Golden Rule:** When rendering multi-class error maps, separate correct class overlap from class confusion:
  ```python
  tp_correct  = (pred_map == gt_mask) & (gt_mask > 0)                 # Correct Class -> Green
  class_error = (pred_map > 0) & (gt_mask > 0) & (pred_map != gt_mask) # Wrong Class -> Yellow/Orange
  fp          = (pred_map > 0) & (gt_mask == 0)                       # False Positive -> Cyan
  fn          = (pred_map == 0) & (gt_mask > 0)                       # False Negative -> Red
  ```

### Trap 11: macOS BSD `rsync` Incompatibility (`--info=progress2`)
- **What happened:** Command `rsync -avzP --info=progress2` crashed on macOS with `rsync: unrecognized option '--info=progress2'`.
- **Root Cause:** macOS ships with BSD rsync 2.6.9, which does not support GNU rsync's `--info=progress2`.
- **How to fix:** On macOS, use `rsync -avzP` (`-P` already includes `--partial --progress`).

### Trap 12: User Execution Preference (Do Not Run Unsolicited Server/Git Commands)
- **What happened:** Attempting to automatically run `sbatch` or `git commit` prompted permission denials because the user explicitly stated they want to run execution commands themselves.
- **Golden Rule:** When server deployment or git commits are ready, provide the exact, copy-pasteable command to the user so they can review and execute it at their own pace.

### Trap 13: 3D vs 4D Shape Agnosticism in Heatmap Utility Functions
- **What happened:** `extract_peak_coords` crashed with `ValueError: too many values to unpack (expected 4)` when passed unbatched 3D tensors $(K, H, W)$ or when callers passed an extra dimension.
- **How to fix:** Always support both $(K, H, W)$ and $(B, K, H, W)$ gracefully:
  ```python
  is_batch = (len(pred_heatmaps.shape) == 4)
  if not is_batch:
      pred_heatmaps = pred_heatmaps.unsqueeze(0)
  # ... process ...
  if not is_batch:
      return coords.squeeze(0), visibilities.squeeze(0), confidences.squeeze(0)
  ```

