"""
Evaluation & Failure Diagnostics Script for EXPERIMENT_15.
Evaluates:
  - Validation Split (specifically monitoring Patient 40)
  - Benchmark Test Split (Patient 41, 31, 51)
Features:
  - Saves top 15 worst cases with qualitative side-by-side diagnostic overlays
  - Saves top 5 best cases
  - Computes temporal jitter and per-patient breakdowns
  - Exports metrics_summary.json, val_predictions.csv, test_predictions.csv
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_15_temporal_augmented_mask2former.utils.dataset_temporal import PatientTemporalL3DDataset
from experiments.EXPERIMENT_15_temporal_augmented_mask2former.utils.metrics import evaluate_frame_metrics, compute_clip_jitter
from experiments.EXPERIMENT_15_temporal_augmented_mask2former.models.temporal_mask2former import load_temporal_mask2former

COLOR_MAP_BGR = {
    0: (0, 0, 0),       # BG
    1: (0, 255, 0),     # Ridge: Green
    2: (255, 150, 0),   # Silhouette: Blue/Cyan
    3: (0, 0, 255)      # Falciform: Red
}

def rasterize_class_map(masks_queries_logits, class_queries_logits, canvas_size=1024):
    masks_queries_logits = masks_queries_logits.float()
    class_queries_logits = class_queries_logits.float()
    
    masks_prob = F.interpolate(
        masks_queries_logits.unsqueeze(0),
        size=(canvas_size, canvas_size),
        mode="bilinear",
        align_corners=False
    ).squeeze(0).sigmoid().cpu().numpy()
    
    class_probs = F.softmax(class_queries_logits, dim=-1).cpu().numpy()
    foreground_probs = class_probs[:, 1:] # (100, 3)
    
    sem_seg = np.zeros((3, canvas_size, canvas_size), dtype=np.float32)
    for q_idx in range(masks_prob.shape[0]):
        m = masks_prob[q_idx]
        for c_idx in range(3):
            c_score = foreground_probs[q_idx, c_idx]
            if c_score > 0.2:
                sem_seg[c_idx] = np.maximum(sem_seg[c_idx], m * c_score)
                
    pred_map = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    for c_idx in range(3):
        bin_mask = sem_seg[c_idx] > 0.4
        pred_map[bin_mask] = (c_idx + 1)
    return pred_map

def render_overlay(img_rgb, mask_map, alpha=0.5):
    h, w = img_rgb.shape[:2]
    if mask_map.shape[:2] != (h, w):
        mask_map = cv2.resize(mask_map, (w, h), interpolation=cv2.INTER_NEAREST)
        
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    overlay = img_bgr.copy()
    
    for c_idx, color in COLOR_MAP_BGR.items():
        if c_idx == 0: continue
        bin_m = (mask_map == c_idx).astype(np.uint8)
        if bin_m.sum() == 0: continue
        contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, contours, -1, color, 8)
        
    return cv2.addWeighted(img_bgr, 1.0 - alpha, overlay, alpha, 0)

def main():
    parser = argparse.ArgumentParser(description="Evaluate EXPERIMENT_15 Model")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to best_model.pth")
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--val_dir', type=str, default=None)
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--clip_len', type=int, default=3)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = os.path.dirname(os.path.abspath(args.ckpt))
    os.makedirs(args.out_dir, exist_ok=True)
    
    diag_worst_dir = os.path.join(args.out_dir, 'diagnostics/worst_cases')
    diag_best_dir = os.path.join(args.out_dir, 'diagnostics/best_cases')
    os.makedirs(diag_worst_dir, exist_ok=True)
    os.makedirs(diag_best_dir, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"📦 Loading model checkpoint: {args.ckpt} on {device}")
    model = load_temporal_mask2former(checkpoint_path=args.ckpt, clip_len=args.clip_len, device=device)
    model.eval()
    
    val_dataset = PatientTemporalL3DDataset('Val', data_dir=args.data_dir, split_dir=args.val_dir, clip_len=args.clip_len, image_size=args.image_size)
    test_dataset = PatientTemporalL3DDataset('Test', data_dir=args.data_dir, split_dir=args.test_dir, clip_len=args.clip_len, image_size=args.image_size)
    
    eval_splits = [('Val', val_dataset), ('Test', test_dataset)]
    all_summaries = {}
    
    for split_name, dataset in eval_splits:
        print(f"\n🔍 Evaluating {split_name} Split ({len(dataset)} clips)...")
        records = []
        raw_renders = []
        
        with torch.no_grad():
            for idx in range(len(dataset)):
                pixel_values, masks, meta = dataset[idx] # (T, 3, H, W), (T, H, W)
                pixel_values_batch = pixel_values.unsqueeze(0).to(device) # (1, T, 3, H, W)
                
                outputs = model(pixel_values_batch)
                masks_logits = outputs['masks_queries_logits'][0] # (T, 100, H/4, W/4)
                class_logits = outputs['class_queries_logits'][0] # (T, 100, 4)
                
                target_t = args.clip_len - 1
                pred_map = rasterize_class_map(masks_logits[target_t], class_logits[target_t], canvas_size=args.image_size)
                gt_map = masks[target_t].cpu().numpy()
                
                # Metrics
                frame_metrics = evaluate_frame_metrics(pred_map, gt_map)
                frame_metrics['patient'] = meta['patient']
                frame_metrics['frame_id'] = meta['frame_ids'][target_t]
                frame_metrics['img_path'] = meta['img_paths'][target_t]
                
                # Temporal jitter across the clip
                clip_preds = [rasterize_class_map(masks_logits[t], class_logits[t], canvas_size=args.image_size) for t in range(args.clip_len)]
                frame_metrics['clip_jitter'] = compute_clip_jitter(clip_preds)
                
                records.append(frame_metrics)
                
                # Keep target images for diagnostic rendering
                raw_renders.append({
                    'img_path': meta['img_paths'][target_t],
                    'pred_map': pred_map,
                    'gt_map': gt_map,
                    'patient': meta['patient'],
                    'frame_id': meta['frame_ids'][target_t],
                    'dice': frame_metrics['macro_dice'],
                    'assd': frame_metrics['macro_assd']
                })
                
        df = pd.DataFrame(records)
        csv_path = os.path.join(args.out_dir, f"{split_name.lower()}_predictions.csv")
        df.to_csv(csv_path, index=False)
        print(f"📄 Saved per-frame predictions to: {csv_path}")
        
        summary = {
            'frames': len(df),
            'macro_dice': float(df['macro_dice'].mean()),
            'macro_iou': float(df['macro_iou'].mean()),
            'macro_assd': float(df['macro_assd'].mean()),
            'mean_jitter': float(df['clip_jitter'].mean()),
            'dice_ridge': float(df['dice_ridge'].mean()),
            'dice_silhouette': float(df['dice_silhouette'].mean()),
            'dice_falciform': float(df['dice_falciform'].mean()),
            'patients': {}
        }
        
        # Per-patient breakdowns
        for pat in df['patient'].unique():
            pat_df = df[df['patient'] == pat]
            summary['patients'][pat] = {
                'frames': len(pat_df),
                'macro_dice': float(pat_df['macro_dice'].mean()),
                'macro_assd': float(pat_df['macro_assd'].mean())
            }
            print(f"   [{pat}] Macro Dice: {pat_df['macro_dice'].mean():.2%} | ASSD: {pat_df['macro_assd'].mean():.2f}px ({len(pat_df)} frames)")
            
        print(f"📊 Overall {split_name} Macro Dice: {summary['macro_dice']:.2%} | Macro ASSD: {summary['macro_assd']:.2f}px | Jitter: {summary['mean_jitter']:.2f}px")
        all_summaries[split_name.lower()] = summary
        
        # Diagnostics: Top 15 Worst Cases & Top 5 Best Cases (Validation split)
        if split_name == 'Val':
            print(f"\n🖼️ Rendering Top 15 Worst Cases for Validation failure analysis...")
            df_sorted = df.sort_values(by='macro_dice', ascending=True)
            worst_15_indices = df_sorted.index[:15]
            
            for rank, w_idx in enumerate(worst_15_indices, 1):
                item = raw_renders[w_idx]
                img_bgr = cv2.imread(item['img_path'])
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                gt_vis = render_overlay(img_rgb, item['gt_map'])
                pred_vis = render_overlay(img_rgb, item['pred_map'])
                
                # Add titles
                cv2.putText(gt_vis, f"GT (Rank {rank} Worst)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(pred_vis, f"PRED: Dice {item['dice']:.1%}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                # Side-by-side comparison panel
                montage = np.hstack([cv2.resize(gt_vis, (640, 360)), cv2.resize(pred_vis, (640, 360))])
                out_name = f"rank_{rank:02d}_{item['patient']}_{item['frame_id']:07d}_dice_{int(item['dice']*100)}.png"
                cv2.imwrite(os.path.join(diag_worst_dir, out_name), montage)
                
            print(f"✅ Saved Top 15 Worst case montages to: {diag_worst_dir}")
            
            # Top 5 Best Cases
            best_5_indices = df_sorted.index[-5:]
            for rank, b_idx in enumerate(reversed(best_5_indices), 1):
                item = raw_renders[b_idx]
                img_bgr = cv2.imread(item['img_path'])
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                gt_vis = render_overlay(img_rgb, item['gt_map'])
                pred_vis = render_overlay(img_rgb, item['pred_map'])
                cv2.putText(gt_vis, f"GT (Rank {rank} Best)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(pred_vis, f"PRED: Dice {item['dice']:.1%}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                montage = np.hstack([cv2.resize(gt_vis, (640, 360)), cv2.resize(pred_vis, (640, 360))])
                out_name = f"best_{rank:02d}_{item['patient']}_{item['frame_id']:07d}_dice_{int(item['dice']*100)}.png"
                cv2.imwrite(os.path.join(diag_best_dir, out_name), montage)
                
    with open(os.path.join(args.out_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print("✅ Evaluation complete! Summary saved to metrics_summary.json")
    
    # Package final results.zip including diagnostics montages
    import zipfile
    zip_path = os.path.join(args.out_dir, 'results.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(args.out_dir):
            for file in files:
                if file.endswith('.zip'):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, args.out_dir)
                zipf.write(file_path, arcname)
    print(f"📦 Final results archive packaged with diagnostics: {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")

if __name__ == '__main__':
    main()
