"""
Evaluation Script for EXPERIMENT_14 (Stratified Option 1 Junction-Steered Mask2Former).
Evaluates:
  - Re-stratified Validation Split (120 frames: Patient_38, Patient_32, Patient_18)
  - Frozen Benchmark Test Split (109 frames: Patient_41, Patient_31, Patient_21, Patient_51)
Produces:
  - metrics_summary.json (overall and per-patient breakdowns)
  - val_predictions.csv
  - test_predictions.csv
  - diagnostics/ (Visual predictions and ground truths for worst cases)
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# NumPy 2.0 monkeypatch
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
    np.PINF = np.inf
    np.NINF = -np.inf

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_14.utils.dataset import StratifiedL3DDataset, IMAGENET_MEAN, IMAGENET_STD
from experiments.EXPERIMENT_14.utils.metrics import evaluate_frame_metrics
from experiments.EXPERIMENT_14.models.junction_steered_mask2former import load_junction_steered_model

COLOR_MAP_RGB = {
    0: (0, 0, 0),       # BG
    1: (34, 197, 94),   # Ridge: Green
    2: (239, 68, 68),   # Silhouette: Red
    3: (59, 130, 246)   # Falciform: Blue
}

JUNCTION_COLORS_BGR = {
    0: (0, 215, 255), # Top: Gold
    1: (255, 255, 0), # Bottom: Cyan
    2: (255, 0, 255), # Lat_Right: Magenta
    3: (0, 140, 255)  # Lat_Left: Orange
}

def rasterize_class_map(masks_queries_logits, class_queries_logits, canvas_size=1024):
    """
    Standard Mask2Former post-processing: Argmax over class probabilities * sigmoid mask probabilities.
    """
    masks_queries_logits = masks_queries_logits.float()
    class_queries_logits = class_queries_logits.float()
    
    masks_prob = torch.sigmoid(masks_queries_logits) # (100, H/4, W/4)
    cls_prob = torch.softmax(class_queries_logits, dim=-1) # (100, 5)
    
    masks_interp = torch.nn.functional.interpolate(
        masks_prob.unsqueeze(0),
        size=(canvas_size, canvas_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0) # (100, H, W)
    
    # Exclude background/no-object class (0)
    sem_prob = torch.einsum('qc,qhw->chw', cls_prob[:, 1:4], masks_interp) # (3, H, W)
    
    bg_thresh = 0.5
    fg_mask = (sem_prob.max(dim=0)[0] > bg_thresh)
    pred_map = torch.zeros((canvas_size, canvas_size), dtype=torch.uint8)
    
    pred_labels = sem_prob.argmax(dim=0) + 1 # 1=Ridge, 2=Sil, 3=Falc
    pred_map[fg_mask] = pred_labels[fg_mask].to(torch.uint8)
    
def render_frame_diagnostic(orig_rgb_norm, gt_mask, pred_map, pred_j_coords, gt_j_coords, gt_j_vis, out_path, metrics):
    """
    Renders standard 4-panel diagnostic montage (Input, GT, Pred, Error Map).
    Matches EXPERIMENT_5 and EXPERIMENT_12/13 output standard.
    """
    img = (orig_rgb_norm.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    
    # Panel 2: GT Overlay
    p2 = bgr.copy()
    for c_id in [1, 2, 3]:
        mask_c = (gt_mask == c_id)
        if mask_c.any():
            col = COLOR_MAP_RGB[c_id][::-1]
            p2[mask_c] = (p2[mask_c] * 0.4 + np.array(col) * 0.6).astype(np.uint8)
    for k in range(4):
        if gt_j_vis[k] > 0.5:
            pt = (gt_j_coords[k] * float(W)).astype(int)
            cv2.circle(p2, tuple(pt), 14, (0, 0, 0), -1)
            cv2.circle(p2, tuple(pt), 10, JUNCTION_COLORS_BGR[k], -1)
            cv2.circle(p2, tuple(pt), 3, (255, 255, 255), -1)
            
    # Panel 3: Pred Overlay
    p3 = bgr.copy()
    for c_id in [1, 2, 3]:
        mask_c = (pred_map == c_id)
        if mask_c.any():
            col = COLOR_MAP_RGB[c_id][::-1]
            p3[mask_c] = (p3[mask_c] * 0.4 + np.array(col) * 0.6).astype(np.uint8)
    for k in range(4):
        pt = (pred_j_coords[k] * float(W)).astype(int)
        cv2.circle(p3, tuple(pt), 14, (0, 0, 0), -1)
        cv2.circle(p3, tuple(pt), 10, JUNCTION_COLORS_BGR[k], -1)
        cv2.circle(p3, tuple(pt), 3, (0, 0, 0), -1)
        
    # Panel 4: Error Map (TP=Green, FP=Cyan, FN=Red)
    p4 = bgr.copy()
    tp = (pred_map > 0) & (gt_mask > 0)
    fp = (pred_map > 0) & (gt_mask == 0)
    fn = (pred_map == 0) & (gt_mask > 0)
    p4[tp] = (p4[tp] * 0.3 + np.array([0, 255, 0]) * 0.7).astype(np.uint8)
    p4[fp] = (p4[fp] * 0.3 + np.array([255, 255, 0]) * 0.7).astype(np.uint8)
    p4[fn] = (p4[fn] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)
    
    top_row = np.hstack([bgr, p2])
    bot_row = np.hstack([p3, p4])
    montage = np.vstack([top_row, bot_row])
    
    cv2.putText(montage, "1. Input Laparoscopic Frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(montage, "2. Ground Truth + GT Junctions", (W + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(montage, f"3. Prediction (Dice: {metrics['macro_dice']:.1%}, ASSD: {metrics['macro_assd']:.1f}px)", (20, H + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
    cv2.putText(montage, "4. Error Map: Green=TP, Cyan=FP, Red=FN", (W + 20, H + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, montage, [cv2.IMWRITE_JPEG_QUALITY, 88])

def run_evaluation(model, dataloader, device, split_name="Val", out_dir=None, max_batches=None, save_diagnostics=False):
    model.eval()
    records = []
    latencies = []
    
    diag_dir = os.path.join(out_dir, f"diagnostics_{split_name.lower()}") if (out_dir and save_diagnostics) else None
    
    print(f"\n🔍 Evaluating {split_name} split ({len(dataloader.dataset)} frames)...")
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if max_batches is not None and idx >= max_batches:
                break
            pixel_values = batch['pixel_values'].to(device)
            masks = batch['mask'].numpy()
            gt_j_coords = batch['junction_coords'].numpy()
            gt_j_vis = batch['junction_vis'].numpy()
            filenames = batch['filename']
            
            t0 = time.time()
            outputs = model(pixel_values)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_batch = (time.time() - t0) * 1000.0 / float(pixel_values.shape[0])
            latencies.append(t_batch)
            
            pred_masks_logits = outputs['masks_queries_logits']
            pred_cls_logits = outputs['class_queries_logits']
            pred_j_coords_t = outputs['pred_junction_coords'].float().cpu().numpy()
            
            for b in range(pixel_values.shape[0]):
                pred_map = rasterize_class_map(pred_masks_logits[b], pred_cls_logits[b])
                
                m = evaluate_frame_metrics(
                    pred_map=pred_map,
                    target_map=masks[b],
                    pred_j_coords=pred_j_coords_t[b],
                    gt_j_coords=gt_j_coords[b],
                    gt_j_vis=gt_j_vis[b]
                )
                m['filename'] = filenames[b]
                # Extract patient ID from filename
                parts = filenames[b].split('_')
                pat_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "Unknown"
                m['patient'] = pat_id
                records.append(m)
                
                if diag_dir:
                    stem = Path(filenames[b]).stem
                    out_diag_path = os.path.join(diag_dir, f"{stem}_diag.jpg")
                    render_frame_diagnostic(
                        orig_rgb_norm=pixel_values[b].float().cpu().numpy(),
                        gt_mask=masks[b],
                        pred_map=pred_map,
                        pred_j_coords=pred_j_coords_t[b],
                        gt_j_coords=gt_j_coords[b],
                        gt_j_vis=gt_j_vis[b],
                        out_path=out_diag_path,
                        metrics=m
                    )
                
            if (idx + 1) % 20 == 0 or (idx + 1) == len(dataloader):
                cur_dice = np.mean([r['macro_dice'] for r in records])
                cur_assd = np.mean([r['macro_assd'] for r in records])
                print(f"   [{split_name} {idx+1}/{len(dataloader)}] Macro Dice: {cur_dice:.2%} | Macro ASSD: {cur_assd:.2f}px")

    df = pd.DataFrame(records)
    
    # Overall summary metrics
    summary = {
        'split': split_name,
        'total_frames': len(df),
        'macro_dice': float(df['macro_dice'].mean()),
        'macro_iou': float(df['macro_iou'].mean()),
        'macro_assd': float(df['macro_assd'].mean()),
        'ridge_dice': float(df['ridge_dice'].mean()),
        'sil_dice': float(df['sil_dice'].mean()),
        'falc_dice': float(df['falc_dice'].mean()),
        'fg_dice': float(df['fg_dice'].mean()),
        'mean_latency_ms': float(np.mean(latencies)),
        'fps': float(1000.0 / np.mean(latencies)),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU/MPS"
    }
    
    # Detailed per-patient breakdown
    pat_stats = {}
    for pat, pat_group in df.groupby('patient'):
        pat_stats[pat] = {
            'frames': len(pat_group),
            'macro_dice': float(pat_group['macro_dice'].mean()),
            'macro_iou': float(pat_group['macro_iou'].mean()),
            'macro_assd': float(pat_group['macro_assd'].mean()),
            'ridge_dice': float(pat_group['ridge_dice'].mean()),
            'sil_dice': float(pat_group['sil_dice'].mean()),
            'falc_dice': float(pat_group['falc_dice'].mean())
        }
    summary['per_patient_metrics'] = pat_stats
    
    # Track junction errors
    valid_j = df['mean_j_err_px'].dropna()
    if len(valid_j) > 0:
        summary['mean_junction_err_px'] = float(valid_j.mean())
        
    return summary, df

def main():
    parser = argparse.ArgumentParser(description="Evaluate EXPERIMENT_14 Stratified Junction-Steered Mask2Former")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to best_model.pth checkpoint")
    parser.add_argument('--data_dir', type=str, default=None, help="Path to L3D dataset root")
    parser.add_argument('--out_dir', type=str, default=None, help="Output directory for results")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = os.path.join(_WORKSPACE_ROOT, 'experiments/EXPERIMENT_14/results')
    os.makedirs(args.out_dir, exist_ok=True)
    
    device = torch.device(args.device)
    
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.out_dir, 'best_model.pth')
        
    print(f"📦 Loading checkpoint from: {args.checkpoint}")
    model = load_junction_steered_model(args.checkpoint, device=device)
    
    val_dataset = StratifiedL3DDataset(split='Val', data_dir=args.data_dir)
    test_dataset = StratifiedL3DDataset(split='Test', data_dir=args.data_dir)
    
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2)
    
    val_summary, val_df = run_evaluation(model, val_loader, device, split_name="Val", out_dir=args.out_dir)
    val_df.to_csv(os.path.join(args.out_dir, 'val_predictions.csv'), index=False)
    
    test_summary, test_df = run_evaluation(model, test_loader, device, split_name="Test", out_dir=args.out_dir)
    test_df.to_csv(os.path.join(args.out_dir, 'test_predictions.csv'), index=False)
    
    full_summary = {
        'val_summary': val_summary,
        'test_summary': test_summary,
        'checkpoint_path': args.checkpoint
    }
    with open(os.path.join(args.out_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(full_summary, f, indent=4)
        
    print(f"🎉 Evaluation Complete! Summary saved to {os.path.join(args.out_dir, 'metrics_summary.json')}")

if __name__ == '__main__':
    main()
