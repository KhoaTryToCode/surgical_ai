# EXP_04: Landmark Master Tokens + Patch Bézier Decoder

## Architecture Summary
EXPERIMENT_4 resolves the severe anatomical hallucination and orientation inversion failures observed during liver retraction and flipped camera viewpoints (e.g. `Patient_40_08940`). 

Building upon the Swin-Tiny + MSDeformAttn pixel decoder and the $8 \times 8$ Patch Bézier decoder of EXPERIMENT_3, EXPERIMENT_4 introduces **3 Category-Level Landmark Master Tokens** ($T_{\text{ridge}}, T_{\text{sil}}, T_{\text{falc}}$) directly into the Transformer Decoder. The sequence length is expanded from 64 to **67 tokens**:
- Tokens 0..2: Coordinate-free Landmark Master Tokens representing Ridge (0), Silhouette (1), and Falciform (2).
- Tokens 3..66: 64 Spatially anchored patch queries initialized from stride-16 features via adaptive average pooling and 2D sinusoidal positional encoding.

### Transformer Self-Attention Dynamics
Across 6 Transformer Decoder layers, full Multi-Head Self-Attention is computed over all 67 tokens. This establishes:
1. **Landmark-to-Landmark Attention (3x3):** Explicitly tracks relative spatial and geometric relationships (e.g. whether Ridge is below Silhouette in normal pose or pulled above during retraction).
2. **Landmark-to-Patch Attention (3x64) & Patch-to-Landmark Attention (64x3):** Broadcasts organ-level orientation and presence context to all 64 local patch queries, conditioning local curve predictions on the global anatomical state.
3. **Patch-to-Patch Attention (64x64):** Preserves spatial curve continuity across adjacent tiles.

### Scale Engine
Features are locked to stride-16 (`single_scale=True` default) to prevent scale-hopping artifacts.

## Loss Formulations
The overall loss function is:
L_total = 2.0 * L_cls + 5.0 * L_ctrl + 2.0 * L_sample + 1.0 * L_cont + 0.5 * L_tan + 1.0 * L_lm

Where:
- L_cls: Focal Loss (gamma=2.0, alpha=0.25) across 64 patches.
- L_ctrl: Smooth-L1 loss (beta=0.02) on 4 Bézier control points for active patches.
- L_sample: L1 loss between 10 Bernstein-sampled curve points on predicted vs ground truth Bézier curves.
- Phase 1 (Epoch 1–30): L_cont = 0, L_tan = 0.
- Phase 2 (Epoch 31–60): L_cont (C0 boundary distance) and L_tan (C1 cosine tangent alignment) enforced between adjacent same-class active patches.
- L_lm: Auxiliary landmark supervision:
  L_lm = L_lm_bce + L_lm_com
  - L_lm_bce: Binary cross entropy on landmark presence logits.
  - L_lm_com: Smooth-L1 loss on normalized Center of Mass (cx, cy) coordinates in [0, 1]^2, masked strictly when target_presence == 1.

## Hyperparameter Table
| Hyperparameter | Value | Description |
|---|---|---|
| Input Image Size | 1024x1024 | RGB 3-channel laparoscopic frames |
| Grid Size | 8x8 | 64 spatial patch queries (128x128 px each) |
| Total Tokens | 67 | 3 landmark tokens + 64 patch queries |
| Decoder Layers | 6 | Pre-norm self-attention + cross-attention + FFN |
| Decoder Embedding Dim | 256 | Multi-head attention with 8 heads |
| Single Scale Lock | True (stride-16) | Eliminates multi-scale hopping noise |
| Epochs | 60 | Cosine Annealing schedule |
| Batch Size (Server A100) | 4 | No accumulation needed (effective batch=4) |
| Batch Size (Kaggle T4) | 1 | Accumulation steps = 4 (effective batch=4) |
| Learning Rate | 8e-5 | AdamW (weight_decay=3e-5) |
| Continuity Phase Start | Epoch 31 | Activates C0 continuity and C1 tangent alignment |
| Rasterization Stroke Width | 35 px | AA polylines on 1024x1024 canvas |

## Directory Structure
- `models/`:
  - `bezier_utils.py`: Bernstein basis, polyline resampling, robust Bézier fitting, discrete rasterizer.
  - `landmark_bezier_decoder.py`: 67-token transformer decoder with landmark & patch heads.
  - `landmark_bezier_model.py`: End-to-end model integrating Swin-Tiny backbone and decoder.
  - `landmark_losses.py`: Focal loss, control point loss, sampling loss, continuity/tangent losses, masked centroid & presence loss.
- `utils/`:
  - `dataset.py`: `LandmarkBezierDataset` computing 64-patch Bézier targets and 3-landmark centroids/presence on the fly.
  - `metrics.py`: Macro Dice, Mean IoU, ASSD, Patient 40 metrics, and 4-panel diagnostic montage renderer with centroid markers.
- `scripts/`:
  - `train.py`: Unified training loop with AMP FP16, AdamW, CosineAnnealing, checkpointing.
  - `evaluate.py`: Standalone evaluation script restoring `best_model.pth`.
  - `run_server.sbatch`: Slurm batch script for `gpu-a240` (A100-40GB).
  - `run_eval_server.sh`: Standalone server evaluation runner.
  - `generate_kaggle_notebook.py`: Self-contained notebook builder for Kaggle GPU.
- `notebooks/`:
  - `EXP4_LandmarkBezier_Kaggle.ipynb`: Self-contained Kaggle runner notebook.
- `results/`:
  - Output checkpoints (`best_model.pth`), metric summaries, diagnostic montages, and `results.zip`.

## Status
Codebase ready for training on Slurm A100 server and Kaggle GPU T4.
