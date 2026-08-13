# Run Commands: EXP_04b (Surgical-BeMapTR v3 Aux Edge Guidance)

## 1. Kaggle Execution Commands (T4 / P100 GPU)

```bash
# ── Cell 1: Pull Repo & Setup Dataset Symlinks ──
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
# ── Cell 2: Launch Surgical-BeMapTR v3 (Aux Edge) Training ──
%cd /kaggle/working/surgical_ai/experiments/EXP_04b_bemaptr_aux_edge/scripts
!python train_bemaptr_aux.py --data_path /kaggle/working/L3D --epochs 60 --batch_size 2 --lr 1e-4 --aux_weight 1.0
```

---

## 2. Google Colab Execution Commands (L4 / A100 GPU)

```bash
# ── Colab Cell 1: Clone Repo & Install Packages ──
!git clone https://github.com/KhoaTryToCode/surgical_ai.git
!pip install timm medpy scikit-image wandb
```

```bash
# ── Colab Cell 2: Run High-Precision Aux Edge Training ──
%cd /content/surgical_ai/experiments/EXP_04b_bemaptr_aux_edge/scripts
!python train_bemaptr_aux.py --data_path /content/L3D --epochs 60 --batch_size 6 --coord_feat_size 128 --lr 1.5e-4 --aux_weight 1.0
```
