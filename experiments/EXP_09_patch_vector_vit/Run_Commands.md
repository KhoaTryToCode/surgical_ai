# Run Commands: EXP_09 Patch-Level Bézier Vision Transformer (RGB-D)

## 1. Brand New Kaggle Notebook Setup & Execution (Step-by-Step)

In Kaggle notebook settings:
- **Accelerator:** GPU P100 or T4 x 2
- **Internet:** ON

### Cell 1: Environment & Dependency Setup
```python
# 1. Install required packages
!pip install -q timm kagglehub scipy opencv-python matplotlib wandb
```

### Cell 2: Clone or Navigate to Workspace
```bash
%%bash
# If cloning repository:
if [ ! -d "/kaggle/working/Surgical-AI" ]; then
    git clone https://github.com/khoatrytocode/surgical_ai.git /kaggle/working/Surgical-AI
fi
cd /kaggle/working/Surgical-AI
git pull origin main || true
```

### Cell 3: Prepare Dataset (Images + Labels + Depth Anything V2)
```python
import os
os.chdir("/kaggle/working/Surgical-AI")

# Sets up /kaggle/working/L3D symlinks; automatically downloads l3d-train, l3d-val, and l3d-depth via kagglehub if not mounted
!python shared/utils/prepare_dataset.py --target_dir /kaggle/working/L3D
```

### Cell 4: Fast GPU Visual Smoke Test
```python
os.chdir("/kaggle/working/Surgical-AI/experiments/EXP_09_patch_vector_vit")
!python scripts/visualize_smoke_test_patch_vit.py
```

### Cell 5: Launch Strongest ViT-Base Training (80 Epochs, vit_base_patch16_224, AMP)
```bash
%%bash
cd /kaggle/working/Surgical-AI/experiments/EXP_09_patch_vector_vit

python scripts/train_patch_vit.py \
    --backbone vit_base_patch16_224 \
    --dataset_dir /kaggle/working/L3D \
    --epochs 80 \
    --batch_size 16 \
    --lr 1e-4 \
    --backbone_lr_mult 0.1 \
    --amp \
    --use_depth \
    --save_dir checkpoints/EXP_09_base \
    --wandb \
    --wandb_key 83f4544a22543e319c6009abceaac90b634c68a3
```

### Cell 6: Quantitative & Qualitative Evaluation
```bash
%%bash
cd /kaggle/working/Surgical-AI/experiments/EXP_09_patch_vector_vit

python scripts/evaluate_patch_vit.py \
    --checkpoint checkpoints/EXP_09/best_model.pth \
    --dataset_dir /kaggle/working/L3D \
    --split val \
    --use_depth \
    --threshold 0.5 \
    --output_dir outputs/eval_results
```

---

## 2. Local macOS (Smoke Test & Fast Diagnostic)

```bash
cd "experiments/EXP_09_patch_vector_vit"
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/visualize_smoke_test_patch_vit.py
```
Output figure saved to: `outputs/smoke_test_patch_vit.png`.
