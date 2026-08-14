# Run Commands — EXP_05: Monocular 3D Vector Space Transformer with Masked Attention

Complete execution CLI commands for Google Colab (A100 GPU + Google Drive mounting), Kaggle CUDA, and local macOS environments.

---

## 1. Google Colab (A100 GPU Execution + Google Drive Persistence)

### Cell 1: Mount Google Drive & Create Directories
```python
from google.colab import drive
drive.mount('/content/drive')

# Create persistent checkpoint and visualization directories on Google Drive
!mkdir -p "/content/drive/MyDrive/Surgical_AI/checkpoints/EXP_05"
!mkdir -p "/content/drive/MyDrive/Surgical_AI/results/EXP_05_visualizations"
```

### Cell 2: Clone Repository & Install Dependencies
```bash
%cd /content
!git clone https://github.com/KhoaTryToCode/surgical_ai.git || (cd surgical_ai && git pull)
%cd /content/surgical_ai
!pip install -q transformers scipy scikit-image opencv-python kagglehub
```

### Cell 3: Symlink Dataset to `/content/L3D`
```bash
# If dataset is uploaded to Google Drive under MyDrive/Surgical_AI/data:
!python shared/utils/prepare_dataset.py --target_dir /content/L3D
```

### Cell 4: Run Automated GT Point & Spline Analyzer
```bash
!python experiments/EXP_05_3d_vector_transformer/utils/dataset_analyzer.py \
  --dataset_dir /content/L3D
```

### Cell 5: Train EXP_05 on A100 GPU (Saving Checkpoints directly to Google Drive)
```bash
!python experiments/EXP_05_3d_vector_transformer/scripts/train_3d_vector_transformer.py \
  --dataset_dir /content/L3D \
  --batch_size 8 \
  --epochs 50 \
  --lr 1e-4 \
  --save_dir "/content/drive/MyDrive/Surgical_AI/checkpoints/EXP_05"
```

### Cell 6: Evaluate 3D Chamfer Distance Metrics
```bash
!python experiments/EXP_05_3d_vector_transformer/scripts/evaluate_3d.py \
  --dataset_dir /content/L3D \
  --checkpoint "/content/drive/MyDrive/Surgical_AI/checkpoints/EXP_05/best_surgical_3d_vector.pth"
```

### Cell 7: Generate Visual Overlays (Saving Directly to Google Drive)
```bash
!python experiments/EXP_05_3d_vector_transformer/scripts/visualize_predictions_3d.py \
  --dataset_dir /content/L3D \
  --checkpoint "/content/drive/MyDrive/Surgical_AI/checkpoints/EXP_05/best_surgical_3d_vector.pth" \
  --output_dir "/content/drive/MyDrive/Surgical_AI/results/EXP_05_visualizations" \
  --num_samples 10
```

---

## 2. Kaggle CUDA Environment Commands

### Unpack & Symlink Dataset
```bash
!python /kaggle/working/surgical_ai/shared/utils/prepare_dataset.py --target_dir /kaggle/working/L3D
```

### Train EXP_05
```bash
!python /kaggle/working/surgical_ai/experiments/EXP_05_3d_vector_transformer/scripts/train_3d_vector_transformer.py \
  --dataset_dir /kaggle/working/L3D \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-4 \
  --save_dir /kaggle/working/checkpoints/EXP_05
```

---

## 3. Local macOS Environment Commands (Apple Silicon MPS / CPU)

```bash
python experiments/EXP_05_3d_vector_transformer/scripts/train_3d_vector_transformer.py \
  --dataset_dir data/laparoscopic_liver \
  --batch_size 2 \
  --epochs 20 \
  --lr 1e-4 \
  --save_dir checkpoints/EXP_05
```
