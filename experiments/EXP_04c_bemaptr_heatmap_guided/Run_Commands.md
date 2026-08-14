# Run Commands — EXP_04c: Heatmap-Guided Vector Prompt Learning

## 1. Local Training (macOS / Metal CPU)
```bash
python experiments/EXP_04c_bemaptr_heatmap_guided/scripts/train_bemaptr_guided.py \
    --data_path data/laparoscopic_liver/L3D \
    --epochs 60 \
    --batch_size 4 \
    --device cpu
```

## 2. Kaggle GPU Execution (Recommended)
```bash
# 1. Update repository
!cd /kaggle/working/surgical_ai && git pull

# 2. Run 60-Epoch Training
%cd /kaggle/working/surgical_ai/experiments/EXP_04c_bemaptr_heatmap_guided/scripts

!python train_bemaptr_guided.py \
    --data_path /kaggle/working/L3D \
    --epochs 60 \
    --batch_size 8 \
    --save_dir checkpoints_bemaptr_guided
```

## 3. Visualization & Prediction Inspection
```bash
%cd /kaggle/working/surgical_ai/experiments/EXP_04c_bemaptr_heatmap_guided/scripts

!python visualize_predictions_guided.py \
    --ckpt_path checkpoints_bemaptr_guided/best_surgical_bemaptr_guided.pth \
    --data_path /kaggle/working/L3D \
    --output_dir /kaggle/working/vis_results_guided \
    --num_samples 5 \
    --conf_threshold 0.05
```
