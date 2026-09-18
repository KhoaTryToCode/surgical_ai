import numpy as np
for _attr, _val in [('Inf', np.inf), ('NAN', np.nan), ('NaN', np.nan), ('PINF', np.inf), ('NINF', -np.inf)]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _val)

import os
import time
from pathlib import Path
import cv2
import pandas as pd
import torch

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
    return dice, iou

def compute_assd(pred_mask, gt_mask, fallback=80.0):
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return fallback
    try:
        from surface_distance import metrics as sd_metrics
        dist = sd_metrics.compute_surface_distances(gt_mask.astype(bool), pred_mask.astype(bool), spacing_mm=(1.0, 1.0))
        assd = sd_metrics.compute_average_surface_distance(dist)
        return (assd[0] + assd[1]) / 2.0
    except ImportError:
        try:
            from medpy.metric.binary import assd
            return assd(pred_mask, gt_mask)
        except ImportError:
            try:
                import cv2
                def get_edge(mask):
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    edge = np.zeros_like(mask)
                    cv2.drawContours(edge, contours, -1, 1, 1)
                    return edge
                pred_edge = get_edge(pred_mask)
                gt_edge = get_edge(gt_mask)
                if pred_edge.sum() == 0 or gt_edge.sum() == 0:
                    return fallback
                
                dist_gt = cv2.distanceTransform((1 - gt_edge).astype(np.uint8), cv2.DIST_L2, 3)
                dist_pred = cv2.distanceTransform((1 - pred_edge).astype(np.uint8), cv2.DIST_L2, 3)
                
                assd_pred2gt = np.mean(dist_gt[pred_edge > 0])
                assd_gt2pred = np.mean(dist_pred[gt_edge > 0])
                
                return (assd_pred2gt + assd_gt2pred) / 2.0
            except Exception:
                return fallback
    except Exception:
        return fallback

def compute_frame_metrics(pred_map, gt_2d):
    metrics = {}
    
    ridge_p, ridge_g = (pred_map == 1), (gt_2d == 1)
    sil_p, sil_g = (pred_map == 2), (gt_2d == 2)
    falc_p, falc_g = (pred_map == 3), (gt_2d == 3)
    
    rd, riou = compute_dice_iou(ridge_p, ridge_g)
    sd, siou = compute_dice_iou(sil_p, sil_g)
    fd, fiou = compute_dice_iou(falc_p, falc_g)
    
    rassd = compute_assd(ridge_p, ridge_g, fallback=80.0)
    sassd = compute_assd(sil_p, sil_g, fallback=80.0)
    fassd = compute_assd(falc_p, falc_g, fallback=80.0)
    
    fg_p, fg_g = (pred_map > 0), (gt_2d > 0)
    fgd, fgiou = compute_dice_iou(fg_p, fg_g)
    
    metrics['ridge_dice'], metrics['ridge_iou'], metrics['ridge_assd'] = rd, riou, rassd
    metrics['sil_dice'], metrics['sil_iou'], metrics['sil_assd'] = sd, siou, sassd
    metrics['falc_dice'], metrics['falc_iou'], metrics['falc_assd'] = fd, fiou, fassd
    
    metrics['macro_dice'] = (rd + sd + fd) / 3.0
    metrics['macro_iou'] = (riou + siou + fiou) / 3.0
    metrics['macro_assd'] = (rassd + sassd + fassd) / 3.0
    
    metrics['fg_dice'] = fgd
    metrics['fg_iou'] = fgiou
    
    return metrics

def _render_patient40_panel(img_tensor, gt_2d, pred_map, fname, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img = img_tensor * std + mean
    img = img.clamp(0, 1).numpy()
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    colors = {
        0: (30, 30, 30),
        1: (0, 0, 255),   
        2: (0, 255, 0),   
        3: (255, 0, 0)    
    }
    
    def apply_overlay(base_img, mask):
        overlay = np.zeros_like(base_img)
        for cls_idx, color in colors.items():
            if cls_idx == 0: continue
            overlay[mask == cls_idx] = color
        alpha = 0.5
        result = cv2.addWeighted(base_img, 1, overlay, alpha, 0)
        for cls_idx, color in colors.items():
            if cls_idx == 0: continue
            result[mask == cls_idx] = cv2.addWeighted(base_img[mask == cls_idx], 1 - alpha, np.full_like(base_img[mask == cls_idx], color), alpha, 0)
        return result
        
    gt_vis = apply_overlay(img_bgr, gt_2d)
    pred_vis = apply_overlay(img_bgr, pred_map)
    
    error_map = np.zeros_like(img_bgr)
    tp = (pred_map == gt_2d) & (gt_2d > 0)
    fp = (pred_map != gt_2d) & (pred_map > 0)
    fn = (pred_map != gt_2d) & (gt_2d > 0)
    
    error_map[tp] = (0, 255, 0)
    error_map[fp] = (0, 0, 255)
    error_map[fn] = (255, 0, 0)
    
    h1 = np.hstack([img_bgr, gt_vis])
    h2 = np.hstack([pred_vis, error_map])
    v = np.vstack([h1, h2])
    
    fname_stem = Path(fname).stem
    cv2.imwrite(os.path.join(out_dir, f"{fname_stem}_diag.png"), v)

def evaluate_model(model, dataloader, device, split_name='Val', save_patient40_dir=None):
    model.eval()
    frame_records = []
    latencies = []
    warmup_done = 0

    with torch.no_grad():
        for pixel_values, gt_2d_batch, target_class, target_bezier, active_mask, filenames in dataloader:
            pixel_values = pixel_values.to(device)
            
            t0 = time.time()
            pred_class, pred_bezier = model(pixel_values)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dt = time.time() - t0
            
            if warmup_done >= 3:
                latencies.append(dt / pixel_values.shape[0])
            warmup_done += 1
            
            pred_class_idx = pred_class.argmax(dim=-1).cpu().numpy()
            pred_bezier_np = pred_bezier.detach().cpu().numpy()
            
            from experiments.EXPERIMENT_3.models.bezier_utils import rasterize_bezier_predictions
            pred_maps = rasterize_bezier_predictions(pred_class_idx, pred_bezier_np)
            gt_maps = gt_2d_batch.numpy()
            
            for b, fname in enumerate(filenames):
                metrics = compute_frame_metrics(pred_maps[b], gt_maps[b])
                fname_base = Path(fname).name if hasattr(fname, 'name') else os.path.basename(str(fname))
                is_p40 = 'Patient_40' in fname_base or '_40_' in fname_base
                metrics['filename'] = fname_base
                metrics['is_patient40'] = is_p40
                frame_records.append(metrics)
                
                if save_patient40_dir and is_p40:
                    _render_patient40_panel(
                        pixel_values[b].cpu(),
                        gt_maps[b],
                        pred_maps[b],
                        fname_base,
                        save_patient40_dir
                    )
                    
    df = pd.DataFrame(frame_records)
    p40_df = df[df['is_patient40'] == True]
    
    cuda = device.type == 'cuda'
    summary_dict = {
        'split': split_name,
        'total_frames': len(frame_records),
        'macro_dice': df['macro_dice'].mean(),
        'macro_iou': df['macro_iou'].mean(),
        'macro_assd': df['macro_assd'].mean(),
        'ridge_dice': df['ridge_dice'].mean(),
        'sil_dice': df['sil_dice'].mean(),
        'falc_dice': df['falc_dice'].mean(),
        'fg_dice': df['fg_dice'].mean(),
        'patient_40_dice': p40_df['macro_dice'].mean() if len(p40_df) > 0 else 0.0,
        'patient_40_assd': p40_df['macro_assd'].mean() if len(p40_df) > 0 else 80.0,
        'patient_40_count': len(p40_df),
        'mean_latency_ms': np.mean(latencies)*1000 if latencies else 0.0,
        'fps': 1.0/(np.mean(latencies)) if latencies else 0.0,
        'gpu_name': torch.cuda.get_device_name(0) if cuda else 'CPU'
    }
    
    return summary_dict, df
