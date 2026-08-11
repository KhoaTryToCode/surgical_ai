# Run Commands: EXP_03 (Surgical-GeMap v2)

## Kaggle Execution Commands

```bash
# ── Cell 1: Clone Repo & Setup Dataset Symlinks ──
!cd /kaggle/working && git clone https://github.com/KhoaTryToCode/surgical_ai.git

!rm -rf /kaggle/working/L3D
!mkdir -p /kaggle/working/L3D/train /kaggle/working/L3D/test /kaggle/working/L3D/val

!ln -sf /kaggle/input/datasets/khoatrytopublish/l3d-train/Train/images /kaggle/working/L3D/train/images
!ln -sf /kaggle/input/datasets/khoatrytopublish/l3d-train/Train/labels /kaggle/working/L3D/train/labels
!ln -sf /kaggle/input/datasets/khoatrytopublish/l3d-test/Test/images /kaggle/working/L3D/test/images
!ln -sf /kaggle/input/datasets/khoatrytopublish/l3d-test/Test/labels /kaggle/working/L3D/test/labels
!ln -sf /kaggle/input/datasets/khoatrytopublish/l3d-val/Val/images /kaggle/working/L3D/val/images
!ln -sf /kaggle/input/datasets/khoatrytopublish/l3d-val/Val/labels /kaggle/working/L3D/val/labels

!pip install timm medpy scikit-image wandb
```

```bash
# ── Cell 2: Run Surgical-GeMap v2 Training ──
%cd /kaggle/working/surgical_ai/experiments/EXP_03_pixel_guided_vector_transformer/scripts
!python train_surgical_gemap_v2.py --data_path /kaggle/working/L3D --epochs 60
```
