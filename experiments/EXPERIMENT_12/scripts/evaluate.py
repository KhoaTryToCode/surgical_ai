"""
Standalone Evaluation Script for EXPERIMENT_12 (Dual-Decoder Mask2Former).
Evaluates:
  - Validation Split (122 frames)
  - Unseen Test Split (109 frames)
Produces:
  - metrics_summary.json
  - val_predictions.csv
  - test_predictions.csv
  - patient_40_diagnostics/ (4-panel diagnostic visualizations for Patient 40 only)
"""
import os
import sys
import time
import json
import zipfile
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# NumPy 2.0 compatibility
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
    np.PINF = np.inf
    np.NINF = -np.inf

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_12.utils.dataset import L3DDataset, collate_fn_l3d, IMAGENET_MEAN, IMAGENET_STD
from experiments.EXPERIMENT_12.utils.metrics import evaluate_frame_metrics
from experiments.EXPERIMENT_12.models.dual_decoder_mask2former import load_dual_decoder_model

# Canonical colors
COLOR_MAP_RGB = {
    0: (0, 0, 0),       # BG
    1: (34, 197, 94),   # Ridge: Green (#22c55e)
    2: (239, 68, 68),   # Silhouette: Red (#ef4444)
    3: (59, 130, 246)   # Falciform: Blue (#3b82f6)
}

JUNCTION_COLORS_BGR = {
    0: (0, 215, 255), # Top: Gold
    1: (255, 255, 0), # Bottom: Cyan
    2: (255, 0, 255), # Lat_Right: Magenta
    3: (0, 140, 255)  # Lat_Left: Orange
}


def rasterize_class_map(masks_queries_logits, class_queries_logits, canvas_size=1024):
    """
    Standard Mask2Former post-processing:
    Argmax over query probabilities multiplied by sigmoid mask probabilities.
    """
    masks_queries_logits = masks_queries_logits.float()
    class_queries_logits = class_queries_logits.float()
    
    # 1. Resize mask logits to canvas_size
    masks = torch.nn.functional.interpolate(
        masks_queries_logits.unsqueeze(0),
        size=(canvas_size, canvas_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).sigmoid() # (100, H, W)
    
    # 2. Query classification probabilities
    cls_probs = torch.softmax(class_queries_logits, dim=-1) # (100, 5), last is BG
    
    # Exclude background class (index 4)
    fg_cls_probs = cls_probs[:, :4] # (100, 4)
    
    # Multiply: (100, 4, 1, 1) * (100, 1, H, W) -> (4, H, W)
    sem_probs = torch.einsum("qc,qhw->chw", fg_cls_probs, masks)
    
    # Thresholding
    pred_map = sem_probs.argmax(dim=0).cpu().numpy().astype(np.int64)
    max_prob = sem_probs.max(dim=0)[0].cpu().numpy()
    pred_map[max_prob < 0.25] = 0
    
    return pred_map


def render_patient_40_diagnostic(orig_rgb_norm, gt_mask, pred_map, pred_j_coords, gt_j_coords, gt_j_vis, out_path, metrics):
    """
    Renders a 4-panel diagnostic montage (1024x1024 x 4 = 2048x2048) for Patient 40 only:
      [ Panel 1: Input RGB ]               [ Panel 2: Ground Truth + GT Junctions ]
      [ Panel 3: Prediction + Pred Juncs ]  [ Panel 4: Error Overlay (TP, FP, FN)   ]
    """
    rgb = (orig_rgb_norm.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
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
        cv2.circle(p3, tuple(pt), 3, (255, 255, 255), -1)
        
    # Panel 4: Error map
    p4 = bgr.copy()
    tp_mask = (pred_map > 0) & (gt_mask > 0)
    fp_mask = (pred_map > 0) & (gt_mask == 0)
    fn_mask = (pred_map == 0) & (gt_mask > 0)
    
    p4[tp_mask] = (p4[tp_mask] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
    p4[fp_mask] = (p4[fp_mask] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
    p4[fn_mask] = (p4[fn_mask] * 0.4 + np.array([255, 255, 0]) * 0.6).astype(np.uint8)
    
    # Annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bgr, "Input Endoscopic View", (25, 45), font, 1.1, (255, 255, 255), 2)
    cv2.putText(p2, "Ground Truth + GT Junctions", (25, 45), font, 1.1, (255, 255, 255), 2)
    cv2.putText(p3, f"Dual-Decoder Pred (Dice: {metrics['macro_dice']:.3f})", (25, 45), font, 1.1, (255, 255, 255), 2)
    cv2.putText(p4, "Error Overlay (Green=TP, Red=FP, Cyan=FN)", (25, 45), font, 1.1, (255, 255, 255), 2)
    
    # 2x2 Montage
    top_row = np.hstack([bgr, p2])
    bot_row = np.hstack([p3, p4])
    montage = np.vstack([top_row, bot_row])
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, montage)


def evaluate_split(model, dataset, device, out_csv_path, p40_dir=None, canvas_size=1024, max_frames=None, num_workers=0):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_l3d)
    model.eval()
    
    results = []
    total_to_eval = min(len(dataset), max_frames) if max_frames else len(dataset)
    print(f"\n🔍 Evaluating on {dataset.split} split ({total_to_eval} frames)...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_frames is not None and i >= max_frames:
                break
            pixel_values = batch['pixel_values'].to(device)
            gt_masks = batch['mask'][0].numpy()
            gt_j_coords = batch['junction_coords'][0].numpy()
            gt_j_vis = batch['junction_vis'][0].numpy()
            filename = batch['filename'][0]
            is_p40 = batch['is_patient_40'][0]
            
            # Forward pass through landmark decoder
            outputs = model.forward_landmark(pixel_values)
            
            masks_queries_logits = outputs['masks_queries_logits'][0]
            class_queries_logits = outputs['class_queries_logits'][0]
            pred_j_coords = outputs['pred_junction_coords'][0].cpu().numpy()
            
            pred_map = rasterize_class_map(masks_queries_logits, class_queries_logits, canvas_size=canvas_size)
            
            m = evaluate_frame_metrics(
                pred_map=pred_map,
                target_map=gt_masks,
                pred_j_coords=pred_j_coords,
                gt_j_coords=gt_j_coords,
                gt_j_vis=gt_j_vis,
                canvas_size=canvas_size
            )
            
            m['filename'] = filename
            m['is_patient_40'] = is_p40
            results.append(m)
            
            if is_p40 and p40_dir is not None:
                stem = Path(filename).stem
                out_img = os.path.join(p40_dir, f"{stem}_diagnostic.png")
                orig_rgb_norm = pixel_values[0].cpu().numpy()
                render_patient_40_diagnostic(
                    orig_rgb_norm=orig_rgb_norm,
                    gt_mask=gt_masks,
                    pred_map=pred_map,
                    pred_j_coords=pred_j_coords,
                    gt_j_coords=gt_j_coords,
                    gt_j_vis=gt_j_vis,
                    out_path=out_img,
                    metrics=m
                )
                
            if (i + 1) % 25 == 0 or (i + 1) == len(dataset):
                print(f"   [{dataset.split}] Processed {i+1:3d}/{len(dataset):3d} frames...")
                
    df = pd.DataFrame(results)
    df.to_csv(out_csv_path, index=False)
    print(f"✅ Saved split metrics: {out_csv_path}")
    
    summary = {
        f"{dataset.split.lower()}_macro_dice": float(df['macro_dice'].mean()),
        f"{dataset.split.lower()}_ridge_dice": float(df['ridge_dice'].mean()),
        f"{dataset.split.lower()}_sil_dice": float(df['sil_dice'].mean()),
        f"{dataset.split.lower()}_falc_dice": float(df['falc_dice'].mean()),
        f"{dataset.split.lower()}_macro_assd": float(df['macro_assd'].mean()),
        f"{dataset.split.lower()}_mean_j_err_px": float(df['mean_j_err_px'].dropna().mean()) if 'mean_j_err_px' in df else None
    }
    
    p40_df = df[df['is_patient_40']]
    if len(p40_df) > 0:
        summary[f"{dataset.split.lower()}_p40_macro_dice"] = float(p40_df['macro_dice'].mean())
        summary[f"{dataset.split.lower()}_p40_sil_dice"] = float(p40_df['sil_dice'].mean())
        summary[f"{dataset.split.lower()}_p40_ridge_dice"] = float(p40_df['ridge_dice'].mean())
        summary[f"{dataset.split.lower()}_p40_falc_dice"] = float(p40_df['falc_dice'].mean())
        
    return summary


def run_evaluation(model, data_dir, out_dir, device, eval_splits='both', max_frames=None, num_workers=0):
    os.makedirs(out_dir, exist_ok=True)
    p40_dir = os.path.join(out_dir, 'patient_40_diagnostics')
    
    val_dataset = L3DDataset(split='Val', data_dir=data_dir, augment=False)
    val_csv = os.path.join(out_dir, 'val_predictions.csv')
    overall_summary = evaluate_split(model, val_dataset, device, val_csv, p40_dir=p40_dir, max_frames=max_frames, num_workers=num_workers)
    
    if eval_splits == 'both':
        test_dataset = L3DDataset(split='Test', data_dir=data_dir, augment=False)
        test_csv = os.path.join(out_dir, 'test_predictions.csv')
        test_summary = evaluate_split(model, test_dataset, device, test_csv, p40_dir=p40_dir, max_frames=max_frames, num_workers=num_workers)
        overall_summary.update(test_summary)
        
    summary_path = os.path.join(out_dir, 'metrics_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, indent=2)
    print(f"📊 Evaluation summary saved: {summary_path}")
    print(json.dumps(overall_summary, indent=2))
    return overall_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate EXPERIMENT_12 Dual-Decoder Mask2Former")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best_model.pth")
    parser.add_argument('--data_dir', type=str, default=None, help="L3D dataset root")
    parser.add_argument('--out_dir', type=str, default=None, help="Directory to save evaluation outputs")
    parser.add_argument('--splits', type=str, default='both', choices=['val', 'both'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.checkpoint)
        
    dev = torch.device(args.device)
    print(f"Loading checkpoint from: {args.checkpoint}")
    m = load_dual_decoder_model(args.checkpoint, device=dev)
    run_evaluation(m, args.data_dir, args.out_dir, dev, eval_splits=args.splits)
