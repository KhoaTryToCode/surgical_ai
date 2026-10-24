#!/usr/bin/env python3
"""
EXPERIMENT_4: Evaluation Metrics & Diagnostic Rendering
Computes standard TopoNet metrics: Macro Dice, Mean IoU, ASSD, Per-class Dice, Patient 40 metrics.
"""
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# NumPy 2.0 compatibility
for _a, _v in [('Inf', np.inf), ('NAN', np.nan), ('NaN', np.nan), ('PINF', np.inf), ('NINF', -np.inf)]:
    if not hasattr(np, _a): setattr(np, _a, _v)

from experiments.EXPERIMENT_4.models.bezier_utils import rasterize_bezier_predictions

def compute_dice_iou(pred_bin, gt_bin):
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    sum_area = pred_bin.sum() + gt_bin.sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    
    if sum_area == 0:
        return 1.0, 1.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0, 0.0
        
    dice = 2.0 * intersection / sum_area
    iou = intersection / union if union > 0 else 0.0
    return float(dice), float(iou)

def compute_assd(pred_mask, gt_mask, fallback=80.0):
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return fallback
    try:
        from surface_distance import metrics as sd_metrics
        dist = sd_metrics.compute_surface_distances(
            gt_mask.astype(bool), pred_mask.astype(bool), spacing_mm=(1.0, 1.0)
        )
        assd = sd_metrics.compute_average_surface_distance(dist)
        return float((assd[0] + assd[1]) / 2.0)
    except Exception:
        try:
            contours_p, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_g, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            edge_p = np.zeros_like(pred_mask, dtype=np.uint8)
            edge_g = np.zeros_like(gt_mask, dtype=np.uint8)
            cv2.drawContours(edge_p, contours_p, -1, 1, 1)
            cv2.drawContours(edge_g, contours_g, -1, 1, 1)
            
            if edge_p.sum() == 0 or edge_g.sum() == 0:
                return fallback
                
            dist_g = cv2.distanceTransform((1 - edge_g).astype(np.uint8), cv2.DIST_L2, 3)
            dist_p = cv2.distanceTransform((1 - edge_p).astype(np.uint8), cv2.DIST_L2, 3)
            
            d_p2g = np.mean(dist_g[edge_p > 0])
            d_g2p = np.mean(dist_p[edge_g > 0])
            return float((d_p2g + d_g2p) / 2.0)
        except Exception:
            return fallback

def compute_frame_metrics(pred_map, gt_2d):
    rd, riou = compute_dice_iou(pred_map == 1, gt_2d == 1)
    sd, siou = compute_dice_iou(pred_map == 2, gt_2d == 2)
    fd, fiou = compute_dice_iou(pred_map == 3, gt_2d == 3)
    fgd, fgiou = compute_dice_iou(pred_map > 0, gt_2d > 0)
    
    rassd = compute_assd(pred_map == 1, gt_2d == 1)
    sassd = compute_assd(pred_map == 2, gt_2d == 2)
    fassd = compute_assd(pred_map == 3, gt_2d == 3)
    
    macro_dice = (rd + sd + fd) / 3.0
    macro_iou  = (riou + siou + fiou) / 3.0
    macro_assd = (rassd + sassd + fassd) / 3.0
    
    return {
        'macro_dice': macro_dice,
        'macro_iou':  macro_iou,
        'macro_assd': macro_assd,
        'ridge_dice': rd, 'ridge_iou': riou, 'ridge_assd': rassd,
        'sil_dice':   sd, 'sil_iou':   siou, 'sil_assd':   sassd,
        'falc_dice':  fd, 'falc_iou':  fiou, 'falc_assd':  fassd,
        'fg_dice':    fgd, 'fg_iou':   fgiou
    }

def _render_patient40_panel(img_tensor, gt_2d, pred_map, fname, out_dir, pred_centroids=None, gt_centroids=None):
    os.makedirs(out_dir, exist_ok=True)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img = img_tensor * std + mean
    img = img.clamp(0, 1).numpy()
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    colors = {
        1: (0, 255, 0), # Green: Ridge (Class 1)
        2: (0, 0, 255), # Red: Silhouette (Class 2)
        3: (255, 140, 0)  # Blue: Falciform (Class 3)
    }
    
    def apply_overlay(base_img, mask):
        result = base_img.copy()
        alpha = 0.5
        for cls_idx, color in colors.items():
            idx = (mask == cls_idx)
            if np.any(idx):
                result[idx] = (base_img[idx].astype(np.float32) * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
        return result
        
    gt_vis   = apply_overlay(img_bgr, gt_2d)
    pred_vis = apply_overlay(img_bgr, pred_map)
    
    # Draw centroid crosshairs if provided
    H, W = img_bgr.shape[:2]
    if pred_centroids is not None:
        for cid in [1, 2, 3]:
            c_pred = pred_centroids[cid - 1]
            px, py = int(c_pred[0] * W), int(c_pred[1] * H)
            cv2.drawMarker(pred_vis, (px, py), colors[cid], markerType=cv2.MARKER_CROSS, markerSize=25, thickness=3)
            
    if gt_centroids is not None:
        for cid in [1, 2, 3]:
            c_gt = gt_centroids[cid - 1]
            if c_gt[0] > 0 or c_gt[1] > 0:
                gx, gy = int(c_gt[0] * W), int(c_gt[1] * H)
                cv2.drawMarker(gt_vis, (gx, gy), colors[cid], markerType=cv2.MARKER_TILTED_CROSS, markerSize=25, thickness=3)
                
    error_map = np.zeros_like(img_bgr)
    tp = (pred_map == gt_2d) & (gt_2d > 0)
    fp = (pred_map != gt_2d) & (pred_map > 0)
    fn = (pred_map != gt_2d) & (gt_2d > 0)
    
    error_map[tp] = (0, 255, 0) # Green = True Positive
    error_map[fp] = (0, 0, 255) # Red = False Positive
    error_map[fn] = (255, 0, 0) # Blue = False Negative
    
    h1 = np.hstack([img_bgr, gt_vis])
    h2 = np.hstack([pred_vis, error_map])
    v = np.vstack([h1, h2])
    
    fname_stem = Path(fname).stem
    cv2.imwrite(os.path.join(out_dir, f"{fname_stem}_diag.png"), v)

def evaluate_model(model, dataloader, device, split_name='Val', save_patient40_dir=None):
    model.eval()
    records = []
    latencies = []
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values, gt_2d_batch, _, _, _, target_lm_pres, target_lm_cent, filenames = batch
            pixel_values = pixel_values.to(device)
            
            t0 = time.time()
            pred_class, pred_bezier, pred_presence, pred_centroid = model(pixel_values)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latency = (time.time() - t0) / pixel_values.shape[0]
            latencies.append(latency)
            
            pred_cls_np  = pred_class.argmax(dim=-1).cpu().numpy()
            pred_bez_np  = pred_bezier.cpu().numpy()
            pred_cent_np = pred_centroid.cpu().numpy()
            gt_cent_np   = target_lm_cent.numpy()
            
            pred_maps = rasterize_bezier_predictions(pred_cls_np, pred_bez_np, grid_size=8, canvas_size=1024, stroke_width=35)
            gt_maps   = gt_2d_batch.numpy()
            
            for b, fname in enumerate(filenames):
                m = compute_frame_metrics(pred_maps[b], gt_maps[b])
                fname_base = os.path.basename(str(fname))
                is_p40 = 'Patient_40' in fname_base or '_40_' in fname_base
                m['filename'] = fname_base
                m['is_patient40'] = is_p40
                records.append(m)
                
                if save_patient40_dir and is_p40:
                    _render_patient40_panel(
                        pixel_values[b].cpu(), gt_maps[b], pred_maps[b], fname_base, save_patient40_dir,
                        pred_centroids=pred_cent_np[b], gt_centroids=gt_cent_np[b]
                    )
                    
    df = pd.DataFrame(records)
    p40_df = df[df['is_patient40']]
    
    mean_lat = np.mean(latencies[5:]) if len(latencies) > 5 else np.mean(latencies)
    fps = 1.0 / mean_lat if mean_lat > 0 else 0.0
    
    summary = {
        'split': split_name,
        'total_frames': len(records),
        'macro_dice': float(df['macro_dice'].mean()),
        'macro_iou':  float(df['macro_iou'].mean()),
        'macro_assd': float(df['macro_assd'].mean()),
        'ridge_dice': float(df['ridge_dice'].mean()),
        'sil_dice':   float(df['sil_dice'].mean()),
        'falc_dice':  float(df['falc_dice'].mean()),
        'fg_dice':    float(df['fg_dice'].mean()),
        'patient_40_dice': float(p40_df['macro_dice'].mean()) if len(p40_df) > 0 else 0.0,
        'patient_40_assd': float(p40_df['macro_assd'].mean()) if len(p40_df) > 0 else 80.0,
        'patient_40_count': len(p40_df),
        'mean_latency_ms': float(mean_lat * 1000.0),
        'fps': float(fps),
    }
    
    return summary, df
