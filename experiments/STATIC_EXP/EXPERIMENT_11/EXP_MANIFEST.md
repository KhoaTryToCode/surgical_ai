# EXPERIMENT_11 Manifest: Targeted Inversion-Weighted Junction-Steered Mask2Former

## 1. Executive Summary & Core Motivation
In **EXPERIMENT_5**, the 2D continuous **Junction-Steered Mask2Former** achieved state-of-the-art overall performance (`Val Macro Dice: 68.04%`, `ASSD: 21.68 px`, `Patient 40 Dice: 70.36%`). However, an exhaustive per-frame failure audit across EXP_5, EXP_8, EXP_9, and EXP_10 revealed that **all four architectures fail identically on severe retracted/inverted liver views** (e.g. `Patient_40_08940` and `Patient_40_09000` crash to 0.0%–10.0% Dice).

A systematic dataset audit across all 921 training frames identified the root cause:
- **The 99:1 Dataset Bias:** Out of 921 training frames, **only 8 frames (0.87%) exhibit severe flap inversion** where laparoscopic graspers pull the Inferior Ridge physically above or level with the Silhouette. 99.2% of images exhibit canonical top-down anatomy (Silhouette strictly at the top, Ridge strictly at the bottom).
- **The Class-Swap Inversion Trap:** Because 99% of training gradients enforce the canonical top/bottom prior, the Transformer queries learn that "top query = Silhouette" and "bottom query = Ridge". When presented with an inverted visceral flap, the models detect the physical contours accurately, but assign Silhouette to the upper curve and Ridge to the lower curve—resulting in 0.0% overlap with ground truth.
- **Architectural Modification Limits:** Neither visibility gating (EXP_8), depth fusion (EXP_9), nor depth + junction steering (EXP_10) can overcome this bias because monocular depth maps only encode step discontinuities, not semantic tissue identity.

**EXPERIMENT_11** tests whether **Targeted Re-Weighting (`WeightedRandomSampler`)** on real deformed/inverted frames can force the network to decouple landmark identity from vertical screen coordinate without corrupting surgical realism.

---

## 2. Methodology & Scientific Design

### Single Independent Variable
- **Base Architecture:** Identical to **EXPERIMENT_5** (`JunctionSteeredMask2Former` with Swin-Tiny backbone, MSDeformAttn pixel decoder, 4-query `JunctionAnchorHead`, and dynamic query cross-attention steering).
- **Control Variable:** No changes to backbone, loss formulation ($\lambda_{\text{coord}}=5.0, \lambda_{\text{vis}}=1.0$), learning rates ($10^{-5} / 10^{-4}$), or resolution ($1024 \times 1024$).
- **Independent Variable:** **Unified Tier-1 Targeted Re-Weighting (`WeightedRandomSampler`)** + **Photometric ColorJitter**:
  - **All 24 audited non-standard/deformed frames are unified into Tier 1** and sampled with **15.0x weight**.
  - Standard canonical frames (897 frames) are sampled with **1.0x weight**.
  - Photometric jitter (brightness $\pm 15\%$, contrast $\pm 15\%$, saturation $\pm 15\%$) ensures the network learns invariant structural semantics rather than memorizing RGB pixel values.

### Audited Deformed Training Frames Ledger (Unified Tier 1: All 24 Frames, Weight = 15.0x)
#### Severe Inversion / Retracted Flap Frames (8 Frames)
| # | Filename | Patient | $\Delta Y$ (Ridge - Sil) | Ridge Peak $Y$ | Clinical Scenario Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `Patient_12_0266520` | Patient 12 | -0.000 | 0.008 | Ridge pulled across ceiling; resection cavity exposed beneath |
| 2 | `Patient_49_84900` | Patient 49 | -0.074 | 0.398 | Double metallic graspers & sutures lifting inferior edge above silhouette |
| 3 | `Patient_38_0368040` | Patient 38 | +0.029 | 0.111 | Heavy instrument traction pulling liver diagonally; 94.3% vertical span overlap |
| 4 | `Patient_20_0014880` | Patient 20 | +0.083 | 0.005 | Giant distended gallbladder elevates liver dome; Ridge loops across top border |
| 5 | `Patient_53_0133800` | Patient 53 | +0.098 | 0.161 | Dual graspers clamping and lifting visceral flap; sharp depth elevation discontinuity |
| 6 | `Patient_12_0267420` | Patient 12 | +0.112 | 0.125 | High-tension traction on segment boundary with tumor resection bed underneath |
| 7 | `Patient_53_0089640` | Patient 53 | +0.061 | 0.367 | Grasper with gauze pad retracts left lobe; 31.5° Falciform tilt, severe lateral rotation |
| 8 | `Patient_12_0265380` | Patient 12 | +0.158 | 0.110 | Retraction grasper visible on upper-left pulling margin upwards |

#### High-Traction & Margin Elevation Frames (16 Frames)
`Patient_49_84930`, `Patient_20_0016200`, `Patient_20_0016080`, `Patient_33_0210000`, `Patient_20_0015600`, `Patient_12_0265320`, `Patient_20_0015960`, `Patient_38_0038880`, `Patient_12_0266580`, `Patient_12_0265980`, `Patient_61_0049680`, `Patient_22_0140280`, `Patient_12_0265080`, `Patient_38_0039240`, `Patient_8_0018840`, `Patient_12_0265140`.

### Sampling Dynamics
- **Standard Training (EXP_5):** Deformed frames appeared in only ~2.6% of training steps.
- **EXP_11 Unified Re-Weighted Training:**
  - Unified Tier 1: 24 frames $\times 15.0 = 360$ probability mass (**28.64%** of epoch samples).
  - Standard Base: 897 frames $\times 1.0 = 897$ probability mass (**71.36%** of epoch samples).
  - **Exposure Frequency:** **28.64%** (approx 1 in every 3.5 samples, meaning roughly 1 out of every 2 training batches of batch size 2 will contain a deformed frame).

---

## 3. Implementation Details & Archival Standard
- **Training Script:** `experiments/EXPERIMENT_11/scripts/train.py`
- **Model Definition:** `experiments/EXPERIMENT_11/models/junction_steered_mask2former.py`
- **Dataset Class:** `experiments/EXPERIMENT_11/utils/dataset.py`
- **SLURM Batch Script:** `experiments/EXPERIMENT_11/scripts/run_server.sbatch`
- **Trap 14 Compliance:**
  - Unified output directory: `SAVE_DIR="/data/khoalq/checkpoints/exp11_deformed_weighted_m2f"`
  - Direct packaging: `results.zip` containing `metrics_summary.json`, all CSVs, and `patient_40_diagnostics/` saved inside `$SAVE_DIR`.
