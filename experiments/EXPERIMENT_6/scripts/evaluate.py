"""
Standalone and integrated evaluation script for EXPERIMENT_6 (Heatmap-Guided Junction-Steered Mask2Former).
Features:
  - Multi-class Macro Dice, IoU, ASSD per anatomical landmark
  - 4-panel visual diagnostics for Patient 40 (RGB, GT, Pred, Error map)
  - Automatic results packaging into results.zip
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
from pathlib import Path

_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_6.utils.dataset import L3DDataset
from experiments.EXPERIMENT_6.models.heatmap_steered_m2f import HeatmapSteeredMask2Former
from experiments.EXPERIMENT_6.utils.metrics import evaluate_frame_metrics

# Color Map (RGB): 1=Ridge (Red), 2=Silhouette (Green), 3=Falciform (Blue)
COLOR_MAP_RGB = {
    1: (255, 50, 50),
    2: (50, 255, 50),
    3: (50, 100, 255)
}
JUNCTION_COLORS_BGR = [
    (0, 0, 255),    # J_top: Red
    (0, 255, 255),  # J_bottom: Yellow
    (255, 0, 255),  # J_lat_r: Magenta
    (255, 255, 0)   # J_lat_l: Cyan
]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def rasterize_class_map(masks_queries_logits, class_queries_logits, canvas_size=1024):
    masks_queries_logits = masks_queries_logits.float()
    class_queries_logits = class_queries_logits.float()
    
    masks = torch.nn.functional.interpolate(
        masks_queries_logits.unsqueeze(0),
        size=(canvas_size, canvas_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).sigmoid() # (100, H, W)
    
    cls_probs = torch.softmax(class_queries_logits, dim=-1) # (100, 5), last is BG
    fg_cls_probs = cls_probs[:, :4] # (100, 4)
    sem_probs = torch.einsum("qc,qhw->chw", fg_cls_probs, masks) # (4, H, W)
    
    pred_map = sem_probs.argmax(dim=0).cpu().numpy().astype(np.int64) # (H, W)
    max_prob = sem_probs.max(dim=0)[0].cpu().numpy()
    pred_map[max_prob < 0.25] = 0
    return pred_map


def render_patient_40_diagnostic(orig_rgb_norm, gt_mask, pred_map, pred_j_coords, gt_j_coords, gt_j_vis, out_path, metrics):
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


def run_evaluation(model, dataloader, device, split_name="Val", out_dir=None):
    model.eval()
    records = []
    latencies = []
    p40_diag_dir = os.path.join(out_dir, 'patient_40_diagnostics') if out_dir else None
    
    print(f"\n🚀 Running {split_name} Evaluation ({len(dataloader.dataset)} frames)...")
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            pixel_values = batch['pixel_values'].to(device)
            masks = batch['mask'].numpy()
            gt_j_coords = batch['junction_coords'].numpy()
            gt_j_vis = batch['junction_vis'].numpy()
            filenames = batch['filename']
            is_p40s = batch['is_patient_40']
            
            t0 = time.time()
            use_amp = (device.type == 'cuda')
            amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                outputs = model(pixel_values)
            latencies.append((time.time() - t0) * 1000.0)
            
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
                m['is_patient_40'] = bool(is_p40s[b])
                records.append(m)
                
                if is_p40s[b] and p40_diag_dir:
                    diag_path = os.path.join(p40_diag_dir, f"{Path(filenames[b]).stem}_diag.jpg")
                    render_patient_40_diagnostic(
                        orig_rgb_norm=pixel_values[b].float().cpu().numpy(),
                        gt_mask=masks[b],
                        pred_map=pred_map,
                        pred_j_coords=pred_j_coords_t[b],
                        gt_j_coords=gt_j_coords[b],
                        gt_j_vis=gt_j_vis[b],
                        out_path=diag_path,
                        metrics=m
                    )
                    
            if (idx + 1) % 20 == 0 or (idx + 1) == len(dataloader):
                cur_dice = np.mean([r['macro_dice'] for r in records])
                cur_assd = np.mean([r['macro_assd'] for r in records])
                print(f"   [{split_name} {idx+1}/{len(dataloader)}] Macro Dice: {cur_dice:.2%} | Macro ASSD: {cur_assd:.2f}px")

    df = pd.DataFrame(records)
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
    
    p40_df = df[df['is_patient_40']]
    if len(p40_df) > 0:
        summary['patient_40_dice'] = float(p40_df['macro_dice'].mean())
        summary['patient_40_assd'] = float(p40_df['macro_assd'].mean())
        summary['patient_40_count'] = len(p40_df)
    else:
        summary['patient_40_dice'] = 0.0
        summary['patient_40_assd'] = 80.0
        summary['patient_40_count'] = 0
        
    valid_j = df['mean_j_err_px'].dropna()
    if len(valid_j) > 0:
        summary['mean_junction_err_px'] = float(valid_j.mean())
        
    return summary, df


def create_results_zip(source_dir, output_zip_path):
    print(f"\n📦 Packaging complete results archive into: {output_zip_path}...")
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path == output_zip_path:
                    continue
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    zip_size_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    print(f"✅ Successfully created {output_zip_path} ({zip_size_mb:.1f} MB)")
