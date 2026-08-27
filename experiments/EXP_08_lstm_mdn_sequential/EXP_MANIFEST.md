# EXP_08: CNN-LSTM-MDN Sequential Surgical Landmark Detection

## Objective
Sequential autoregressive model treating surgical landmark detection as a stroke-drawing problem. A ResNet-18 backbone extracts spatial features, then an LSTM decoder with a Mixture Density Network (MDN) head predicts landmark polylines one point at a time, conditioned on local visual features bilinearly sampled at each predicted coordinate.

## Architecture Summary
```
Image (3, 512, 512)
    ↓
ResNet-18 Backbone (layer3) → Feature Map (256, 32, 32)
    ↓
For each landmark instance:
    ↓
    LSTM Initial State ← [GAP(F); ClassEmbedding]
    ↓
    Step 0: <INIT> token → LSTM → MDN Head → (π, μ, σ, eos)
    Step 1: [GT_p0; grid_sample(F, GT_p0)] → LSTM → MDN → (π, μ, σ, eos)
    ...
    Step K: [GT_p_{K-1}; grid_sample(F, GT_p_{K-1})] → LSTM → MDN → (π, μ, σ, eos)
    ↓
Loss: λ_mdn·L_mdn + λ_point·L_point + λ_dir·L_dir + λ_mask·L_mask + λ_eos·L_eos
```

## Key Design Decisions
1. **Teacher Forcing**: GT coordinates fed as input during training (no scheduled sampling)
2. **Absolute Coordinates**: Predicts (u, v) ∈ [0,1]^2 directly (no offset accumulation)
3. **Bilinear Grid Sample**: Local visual features extracted at sub-pixel precision via F.grid_sample
4. **Independent Bivariate Gaussian MDN**: No ρ_xy correlation in v1 for simplicity
5. **Learnable <INIT> Token**: Trainable embedding replaces missing p_{-1} at first step
6. **One LSTM Pass Per Instance**: Class-conditioned initialization, no Hungarian matching needed

## Loss Components
| Loss | Weight | Description |
|:-----|:------:|:------------|
| L_mdn | 1.0 | Gaussian Mixture NLL (log-sum-exp stabilized) |
| L_point | 5.0 | Expected-point Smooth-L1 regression |
| L_dir | 1.0 | Tangent direction cosine alignment |
| L_mask | 2.0 | Differentiable soft mask Dice (Gaussian splatting) |
| L_eos | 0.5 | End-of-sequence binary cross-entropy |

## Training Commands

### Kaggle (GPU)
```bash
cd experiments/EXP_08_lstm_mdn_sequential
python scripts/train_lstm_mdn.py \
    --dataset_dir /kaggle/input/laparoscopic-liver-landmarks \
    --epochs 80 \
    --batch_size 16 \
    --lr 1e-4 \
    --save_dir checkpoints/EXP_08 \
    --wandb
```

### Evaluation
```bash
python scripts/evaluate_lstm_mdn.py \
    --checkpoint checkpoints/EXP_08/best_model.pth \
    --dataset_dir /kaggle/input/laparoscopic-liver-landmarks \
    --split val
```

## Expected Performance Targets
| Metric | Target Range |
|:-------|:------------|
| Validation Dice | 0.30 – 0.50 |
| Polyline Error | < 80px at 512×512 |

## File Structure
```
EXP_08_lstm_mdn_sequential/
├── configs/exp08_config.py           # Hyperparameter dataclass
├── models/
│   ├── backbone_resnet.py            # ResNet-18 feature extractor
│   ├── lstm_mdn_decoder.py           # LSTM + MDN autoregressive decoder
│   ├── surgical_lstm_mdn.py          # Full model pipeline
│   └── mdn_losses.py                 # Multi-task loss suite
├── utils/dataset_sequential.py       # Dataset adapter (512×512)
├── scripts/
│   ├── train_lstm_mdn.py             # Training loop (teacher forcing)
│   └── evaluate_lstm_mdn.py          # Validation metrics
└── EXP_MANIFEST.md                   # This file
```

## References
- Ha & Eck (2017). "A Neural Representation of Sketch Drawings" — Sketch-RNN MDN formulation
- Graves (2013). "Generating Sequences with Recurrent Neural Networks" — Original MDN-LSTM framework
