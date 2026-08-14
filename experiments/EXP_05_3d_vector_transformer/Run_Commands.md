# Run Commands — EXP_05: Monocular 3D Vector Space Transformer with Masked Attention

Complete execution CLI commands for local macOS (Apple Silicon MPS / CPU) and Kaggle GPU environments.

---

## 1. Dataset Preparation Command (Kaggle Environment)

Run dataset preparation script to extract and unpack the Laparoscopic Liver 3D dataset:

```bash
!python /kaggle/working/surgical_ai/shared/utils/prepare_dataset.py --target_dir /kaggle/working/L3D
```

---

## 2. Automated GT Point Distribution & Optimal K Analysis

Run dataset analysis utility to scan raw GT annotations and verify optimal polyline vertex count $K$:

### Kaggle CUDA Environment:
```bash
!python /kaggle/working/surgical_ai/experiments/EXP_05_3d_vector_transformer/utils/dataset_analyzer.py \
  --dataset_dir /kaggle/working/L3D
```

### Local macOS Environment:
```bash
python experiments/EXP_05_3d_vector_transformer/utils/dataset_analyzer.py \
  --dataset_dir data/laparoscopic_liver
```

---

## 3. Training Commands (EXP_05 End-to-End)

Train the 3D Vector Space Transformer with deep supervision across decoder layers:

### Kaggle CUDA Environment (P100 / T4 GPU):
```bash
!python /kaggle/working/surgical_ai/experiments/EXP_05_3d_vector_transformer/scripts/train_3d_vector_transformer.py \
  --dataset_dir /kaggle/working/L3D \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-4 \
  --save_dir /kaggle/working/checkpoints/EXP_05
```

### Local macOS Environment (Apple Silicon MPS / CPU):
```bash
python experiments/EXP_05_3d_vector_transformer/scripts/train_3d_vector_transformer.py \
  --dataset_dir data/laparoscopic_liver \
  --batch_size 2 \
  --epochs 20 \
  --lr 1e-4 \
  --save_dir checkpoints/EXP_05
```

---

## 4. Evaluation Commands (3D Chamfer Distance Metric)

Evaluate model predictions against 3D ground-truth polylines:

### Kaggle CUDA Environment:
```bash
!python /kaggle/working/surgical_ai/experiments/EXP_05_3d_vector_transformer/scripts/evaluate_3d.py \
  --dataset_dir /kaggle/working/L3D \
  --checkpoint /kaggle/working/checkpoints/EXP_05/best_surgical_3d_vector.pth
```

### Local macOS Environment:
```bash
python experiments/EXP_05_3d_vector_transformer/scripts/evaluate_3d.py \
  --dataset_dir data/laparoscopic_liver \
  --checkpoint checkpoints/EXP_05/best_surgical_3d_vector.pth
```

---

## 5. Visualization Commands (Overlays & 3D Vector Predictions)

Generate visual overlays of predicted 3D polylines and 2D multi-class masks:

### Kaggle CUDA Environment:
```bash
!python /kaggle/working/surgical_ai/experiments/EXP_05_3d_vector_transformer/scripts/visualize_predictions_3d.py \
  --dataset_dir /kaggle/working/L3D \
  --checkpoint /kaggle/working/checkpoints/EXP_05/best_surgical_3d_vector.pth \
  --output_dir /kaggle/working/results/EXP_05_visualizations \
  --num_samples 10
```

### Local macOS Environment:
```bash
python experiments/EXP_05_3d_vector_transformer/scripts/visualize_predictions_3d.py \
  --dataset_dir data/laparoscopic_liver \
  --checkpoint checkpoints/EXP_05/best_surgical_3d_vector.pth \
  --output_dir results/EXP_05_visualizations \
  --num_samples 5
```
