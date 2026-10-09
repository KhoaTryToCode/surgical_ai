# Experiment Manifest: EXPERIMENT_15

## 1. Experiment Overview
- **Experiment ID:** `EXPERIMENT_15`
- **Name:** Temporal-Augmented Mask2Former ($T=3$)
- **Core Hypothesis:** Treating laparoscopic surgery as isolated static slices induces topological hallucination under surgical retraction (e.g. Patient 40 where the anterior ridge is hoisted to the upper margin). By augmenting vanilla Mask2Former with lightweight Spatiotemporal Query Cross-Attention over $T=3$ chronological keyframes within each patient ($(I_{t-2}, I_{t-1}, I_t)$), the model tracks continuous physical kinematics, resolving the topological inversion without complex 3D meshes or pre-op CT scans.
- **Base Checkpoint:** Pure vanilla `facebook/mask2former-swin-tiny-ade-semantic` (ImageNet/ADE20k base, zero surgical or junction priors for pure baseline isolation).

---

## 2. Architecture & Mathematical Formulation
- **Input:** $T=3$ video frames within the same patient: $I \in \mathbb{R}^{B \times 3 \times 3 \times H \times W}$.
- **Spatial Backbone:** Swin-Tiny + MSDeformAttn Pixel Decoder.
- **Temporal Query Block:**
  Queries from all $T=3$ frames interact via cross-attention with learnable gating parameter $\gamma$ (initialized to 0.05):
  `Q_t = LayerNorm(Q_t + gamma * CrossAttention(Q_t, Key=[Q_{t-1}, Q_{t-2}], Value=[Q_{t-1}, Q_{t-2}]))`
- **Output:** Space-time tubelet masks $(M_{t-2}, M_{t-1}, M_t)$ and class probabilities.
- **Loss:** Built-in Mask2Former Hungarian criterion across all clip frames.

---

## 3. Dataset & Training Configuration
- **Dataset:** L3D Benchmark (Patient-wise grouping).
- **Resolution:** $1024 \times 1024$ (Stroke Width: 35 px).
- **Epochs:** 30 epochs (or 60 epochs; Effective batch size = 4 clips / 12 frames).
- **Compute Profiles:**
  - **20GB Profile (`run_server_20gb.sbatch`):** `--gres=gpu:a100_3g.20gb:1`, batch size 1, accum 4, ~5.8 GB peak VRAM (14 GB free headroom, 100% zero OOM risk).
  - **10GB Profile (`run_server_10gb.sbatch`):** `--gres=gpu:a100_2g.10gb:1`, batch size 1, accum 4, ~5.8 GB peak VRAM.
  - **Full GPU Profile (`run_server.sbatch`):** `--gres=gpu:1`, batch size 2, accum 2.
- **Contiguity Threshold:** Consecutive frame gap $\le 240$ frames ($\le 8.0$ seconds at 30 fps; median $\approx 2.0\text{--}4.0$ seconds).
- **Available Clips:**
  - Train: 566 clips across 26 patients (sampled using `PatientStratifiedClipSampler`).
  - Validation: 93 clips (Patient 40 has 91 clips).
  - Test: 76 clips (Patient 41 has 74 clips).

---

## 4. Deliverables & Outputs
- `best_model.pth`: Optimal checkpoint based on Val Macro Dice & ASSD.
- `training_log.csv`: Epoch-by-epoch Train Loss, Val Dice, Val ASSD, Gamma.
- `val_predictions.csv` & `test_predictions.csv`: Per-frame metric logs.
- `metrics_summary.json`: Comprehensive report including per-patient breakdowns and temporal jitter.
- `diagnostics/worst_cases/`: Top 15 worst validation cases saved with side-by-side diagnostic overlays.
- `diagnostics/best_cases/`: Top 5 best cases.
- `results.zip`: Self-contained archive for 1-click download.
