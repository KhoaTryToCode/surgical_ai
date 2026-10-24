# Run Commands: EXP_04 (Surgical-BeMapTR v2)

## 1. Kaggle Execution Commands (Standard T4 / P100 GPU)

```bash
# ── Cell 1: Setup Dataset & Environment ──
!rm -rf /kaggle/working/L3D
!mkdir -p /kaggle/working/L3D/train /kaggle/working/L3D/test /kaggle/working/L3D/val

!ln -sf /kaggle/input/datasets/khoatrytopublish/l3d-train/Train/images /kaggle/working/L3D/train/images
!ln -sf /kaggle/input/datasets/khoatrytopublish/l3d-train/Train/labels /kaggle/working/L3D/train/labels
!ln -sf /kaggle/input/datasets/khoatrytopublish/l3d-test/Test/images /kaggle/working/L3D/test/images
!ln -sf /kaggle/input/datasets/khoatrytopublish/l3d-test/Test/labels /kaggle/working/L3D/test/labels
!ln -sf /kaggle/input/datasets/khoatrytopublish/l3d-val/Val/images /kaggle/working/L3D/val/images
!ln -sf /kaggle/input/datasets/khoatrytopublish/l3d-val/Val/labels /kaggle/working/L3D/val/labels

!cd /kaggle/working/surgical_ai && git pull
!pip install timm medpy scikit-image wandb
```

```bash
# ── Cell 2: Run Surgical-BeMapTR v2 Training ──
%cd /kaggle/working/surgical_ai/experiments/EXP_04_surgical_bemaptr_pure_vector/scripts
!python train_bemaptr.py --data_path /kaggle/working/L3D --epochs 60 --batch_size 2 --lr 1e-4
```

---

## 2. Google Colab A100 (High-VRAM Ultra-Precision Execution)

With an A100 (40GB/80GB VRAM), we can train with **4x larger batch size** (`batch_size=8`) and **256x256 spatial centroid feature maps** (`--coord_feat_size 256` matching FPN P2 resolution directly) for maximum sub-pixel localization accuracy.

```bash
# ── Colab Cell 1: Clone Repo & Install Packages ──
!git clone https://github.com/KhoaTryToCode/surgical_ai.git
!pip install timm medpy scikit-image wandb
```

```bash
# ── Colab Cell 2: Run High-Precision A100 Training ──
%cd /content/surgical_ai/experiments/EXP_04_surgical_bemaptr_pure_vector/scripts
!python train_bemaptr.py --data_path /content/L3D --epochs 60 --batch_size 8 --coord_feat_size 256 --lr 2e-4
```

