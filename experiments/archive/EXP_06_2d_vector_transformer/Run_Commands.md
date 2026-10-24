# EXP_06 Run Commands: Direct 2D Vector Space Transformer

---

## 1. Google Colab Setup & Visual Smoke Test

### Step A: Pull Latest Code
```bash
!cd /content/surgical_ai && git pull
```

### Step B: Run 2D Visual Diagnostic Smoke Test
```python
!python experiments/EXP_06_2d_vector_transformer/scripts/visualize_pipeline_smoke_test_2d.py \
  --dataset_dir /content/L3D \
  --sample_idx 0 \
  --output /content/pipeline_diagnostic_2d.png

from IPython.display import Image, display
display(Image("/content/pipeline_diagnostic_2d.png"))
```

---

## 2. Launch 50-Epoch Training on Colab (NVIDIA A100 / T4)

```bash
!python experiments/EXP_06_2d_vector_transformer/scripts/train_2d_vector_transformer.py \
  --dataset_dir /content/L3D \
  --batch_size 8 \
  --epochs 50 \
  --lr 1e-4 \
  --save_dir "/content/drive/MyDrive/Surgical_AI/checkpoints/EXP_06" \
  --wandb
```

---

## 3. Quantitative Evaluation

```bash
!python experiments/EXP_06_2d_vector_transformer/scripts/evaluate_2d.py \
  --checkpoint "/content/drive/MyDrive/Surgical_AI/checkpoints/EXP_06/best_surgical_2d_vector.pth" \
  --dataset_dir /content/L3D
```
