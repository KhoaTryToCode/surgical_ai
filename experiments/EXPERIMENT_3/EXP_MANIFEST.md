# EXP_03: Mask2Former Backbone + Patch Bézier Decoder

## Architecture Summary
This experiment explores a hybrid architecture that uses a Mask2Former backbone for robust feature extraction combined with a Patch Bézier Decoder. The input images (1024x1024) are processed into an 8x8 patch grid (64 patches), and for each patch, the model predicts the class and control points of a Bézier curve representing a topological structure (Ridge, Silhouette, or Falciform). 

## Loss Formulations
The overall loss is a weighted sum of several components:
- Total Loss: L = 2.0 * L_cls + 5.0 * L_ctrl + 2.0 * L_sample + 1.0 * L_cont + 0.5 * L_tan
- Phase 1 (Epoch 1-30): Focus on L_cls, L_ctrl, L_sample
- Phase 2 (Epoch 31+): Includes continuity and tangent losses to enforce topological consistency.

## Hyperparameter Table
| Hyperparameter       | Value  |
|----------------------|--------|
| Image Size           | 1024x1024 |
| Patch Grid           | 8x8 (64 patches) |
| Epochs               | 60 |
| Batch Size           | 1 |
| Accumulation Steps   | 4 |
| Learning Rate        | 8e-5 |
| Weight Decay         | 3e-5 |
| Lambda Cls           | 2.0 |
| Lambda Ctrl          | 5.0 |
| Lambda Sample        | 2.0 |
| Lambda Cont          | 1.0 |
| Lambda Tan           | 0.5 |
| Optimizer            | AdamW |
| LR Scheduler         | CosineAnnealingLR |

## Directory Structure
- `utils/`: Datasets, metrics, rasterization, rendering utilities.
- `models/`: BézierPatchModel, BezierPatchLoss, and bezier_utils.
- `scripts/`: `train.py` main execution script.
- `results/`: Execution outputs, checkpoints, and patient_40 diagnostic renderings.

## Status
In Progress
