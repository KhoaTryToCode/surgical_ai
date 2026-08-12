# Run Commands: EXP_04 (Surgical-BeMapTR)

## Kaggle Execution Commands

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
# ── Cell 2: Run Surgical-BeMapTR Training ──
%cd /kaggle/working/surgical_ai/experiments/EXP_04_surgical_bemaptr_pure_vector/scripts
!python train_bemaptr.py --data_path /kaggle/working/L3D --epochs 60
```
