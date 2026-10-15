"""
Evaluation Script for EXPERIMENT_9 (RGB-D Junction-Steered Mask2Former).

Features:
  - Exact evaluation protocol matching EXPERIMENT_8:
      * Macro Dice, IoU, ASSD per class (Ridge, Silhouette, Falciform)
      * Patient 40 diagnostic sub-metrics (Dice, ASSD, Landmark MAPE)
      * High-resolution 5-panel diagnostic montages for Patient 40:
          [ Panel 1: RGB ] [ Panel 2: Depth ] [ Panel 3: Ground Truth ] [ Panel 4: Pred ] [ Panel 5: Error Map ]
      * Returns (summary_dict, per_frame_df) compatible with train.py
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
from tqdm import tqdm

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_9.utils.dataset import L3DRGBDDataset, IMAGENET_MEAN, IMAGENET_STD, DEPTH_MEAN, DEPTH_STD
from experiments.EXPERIMENT_9.utils.metrics import evaluate_frame_metrics
from experiments.EXPERIMENT_9.models.rgbd_junction_steered_mask2former import RGBDJunctionSteeredMask2Former

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

def render_patient_40_diagnostic(orig_rgbd_norm, gt_mask, pred_map, pred_j_coords, pred_j_vis_probs, gt_j_coords, gt_j_vis, out_path, metrics, vis_thresh=0.5):
    """
    Renders a 5-panel horizontal diagnostic montage for Patient 40:
      [ Panel 1: RGB ] [ Panel 2: Depth ] [ Panel 3: GT ] [ Panel 4: Pred ] [ Panel 5: Error ]
    """
    # 1. RGB
    rgb = (orig_rgbd_norm[:3].transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    
    # 2. Depth
    depth_unnorm = (orig_rgbd_norm[3] * DEPTH_STD + DEPTH_MEAN) * 255.0
    depth_uint8 = np.clip(depth_unnorm, 0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    
    # 3. GT Overlay
    p3 = bgr.copy()
    for c_id in [1, 2, 3]:
        mask_c = (gt_mask == c_id)
        if mask_c.any():
            col = COLOR_MAP_RGB[c_id][::-1]
            p3[mask_c] = (p3[mask_c] * 0.4 + np.array(col) * 0.6).astype(np.uint8)
            
    # Draw GT Junctions
    for k in range(4):
        if gt_j_vis[k] > 0.5:
            pt = (gt_j_coords[k] * float(W)).astype(int)
            cv2.circle(p3, tuple(pt), 14, (0, 0, 0), -1)
            cv2.circle(p3, tuple(pt), 10, JUNCTION_COLORS_BGR[k], -1)
            
    # 4. Pred Overlay
    p4 = bgr.copy()
    for c_id in [1, 2, 3]:
        mask_c = (pred_map == c_id)
        if mask_c.any():
            col = COLOR_MAP_RGB[c_id][::-1]
            p4[mask_c] = (p4[mask_c] * 0.4 + np.array(col) * 0.6).astype(np.uint8)
            
    # Draw Predicted Junctions
    for k in range(4):
        prob = pred_j_vis_probs[k]
        if prob >= vis_thresh:
            pt = (pred_j_coords[k] * float(W)).astype(int)
            cv2.circle(p4, tuple(pt), 14, (0, 0, 0), -1)
            cv2.circle(p4, tuple(pt), 10, JUNCTION_COLORS_BGR[k], -1)
            cv2.putText(p4, f"{prob:.2f}", (pt[0] + 12, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    # 5. Semantic Error Map
    p5 = np.zeros_like(bgr)
    p5[(pred_map > 0) & (gt_mask > 0) & (pred_map == gt_mask)] = (0, 200, 0)   # Correct: Green
    p5[(pred_map > 0) & (gt_mask == 0)] = (0, 0, 255)                          # False Pos: Red
    p5[(pred_map == 0) & (gt_mask > 0)] = (0, 255, 255)                        # False Neg: Yellow
    p5[(pred_map > 0) & (gt_mask > 0) & (pred_map != gt_mask)] = (255, 0, 255) # Class Mismatch: Magenta
    
    cv2.putText(bgr, "1. Laparoscopic RGB", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(depth_color, "2. Depth Anything v2", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(p3, "3. Ground Truth Landmarks", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(p4, f"4. EXP_9 RGB-D Pred ({metrics['macro_dice']*100:.1f}%)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(p5, f"5. Error (ASSD: {metrics['macro_assd']:.1f}px)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    montage = np.hstack([bgr, depth_color, p3, p4, p5])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, montage)

def run_evaluation(model, dataloader, device, split_name="Val", out_dir=None, vis_thresh=0.5):
    """
    Runs full evaluation over the dataset.
    Returns (summary_dict, per_frame_df)
    """
    model.eval()
    records = []
    latencies = []
    
    p40_diag_dir = None
    if out_dir is not None:
        p40_diag_dir = os.path.join(out_dir, "patient_40_diagnostics")
        os.makedirs(p40_diag_dir, exist_ok=True)
        
    print(f"\n🔬 Starting Evaluation on {split_name} Split ({len(dataloader.dataset)} frames)...")
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc=f"Eval {split_name}")):
            pixel_values = batch['pixel_values'].to(device) # (B, 4, H, W)
            masks = batch['mask'].cpu().numpy()
            gt_j_coords = batch['junction_coords'].cpu().numpy()
            gt_j_vis = batch['junction_vis'].cpu().numpy()
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
            pred_j_vis_probs_t = outputs['pred_junction_vis_probs'].float().cpu().numpy()
            
            for b in range(pixel_values.shape[0]):
                pred_map = rasterize_class_map(pred_masks_logits[b], pred_cls_logits[b])
                
                m = evaluate_frame_metrics(
                    pred_map=pred_map,
                    target_map=masks[b],
                    pred_j_coords=pred_j_coords_t[b],
                    pred_j_vis=pred_j_vis_probs_t[b],
                    gt_j_coords=gt_j_coords[b],
                    gt_j_vis=gt_j_vis[b]
                )
                m['filename'] = filenames[b]
                m['is_patient_40'] = bool(is_p40s[b])
                records.append(m)
                
                if is_p40s[b] and p40_diag_dir:
                    diag_path = os.path.join(p40_diag_dir, f"{Path(filenames[b]).stem}_rgbd_diag.jpg")
                    render_patient_40_diagnostic(
                        orig_rgbd_norm=pixel_values[b].float().cpu().numpy(),
                        gt_mask=masks[b],
                        pred_map=pred_map,
                        pred_j_coords=pred_j_coords_t[b],
                        pred_j_vis_probs=pred_j_vis_probs_t[b],
                        gt_j_coords=gt_j_coords[b],
                        gt_j_vis=gt_j_vis[b],
                        out_path=diag_path,
                        metrics=m,
                        vis_thresh=vis_thresh
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
        'ridge_assd': float(df['ridge_assd'].mean()),
        'sil_assd': float(df['sil_assd'].mean()),
        'falc_assd': float(df['falc_assd'].mean()),
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
        
    c08730 = df[df['filename'].str.contains('08730')]
    if len(c08730) > 0:
        summary['case_08730_dice'] = float(c08730['macro_dice'].iloc[0])
        summary['case_08730_assd'] = float(c08730['macro_assd'].iloc[0])
        summary['case_08730_ridge_dice'] = float(c08730['ridge_dice'].iloc[0])
        summary['case_08730_sil_dice'] = float(c08730['sil_dice'].iloc[0])
        
    valid_j = df['mean_j_err_px'].dropna()
    if len(valid_j) > 0:
        summary['mean_junction_err_px'] = float(valid_j.mean())
    else:
        summary['mean_junction_err_px'] = None
        
    return summary, df

def main():
    parser = argparse.ArgumentParser(description="Evaluate EXPERIMENT_9 RGB-D Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--split", type=str, default="Val", choices=["Val", "Test", "Train"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="experiments/EXPERIMENT_9/results")
    args = parser.parse_args()
    
    dataset = L3DRGBDDataset(split=args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    model = RGBDJunctionSteeredMask2Former(num_labels=4)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(args.device)
    
    os.makedirs(args.out_dir, exist_ok=True)
    summary, df = run_evaluation(model, dataloader, args.device, split_name=args.split, out_dir=args.out_dir)
    
    df.to_csv(os.path.join(args.out_dir, f"{args.split.lower()}_predictions.csv"), index=False)
    with open(os.path.join(args.out_dir, f"{args.split.lower()}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
        
    print("\n" + "=" * 60)
    print(f"📊 EVALUATION SUMMARY ({args.split}):")
    print(f"  Macro Dice:      {summary['macro_dice']*100:.2f}%")
    print(f"  Macro ASSD:      {summary['macro_assd']:.2f} px")
    print(f"  Ridge Dice:      {summary['ridge_dice']*100:.2f}% | Sil Dice: {summary['sil_dice']*100:.2f}% | Falc Dice: {summary['falc_dice']*100:.2f}%")
    print(f"  Patient 40 Dice: {summary['patient_40_dice']*100:.2f}% | ASSD: {summary['patient_40_assd']:.2f} px")
    if 'case_08730_dice' in summary:
        print(f"  Case 08730 Dice: {summary['case_08730_dice']*100:.2f}% | ASSD: {summary['case_08730_assd']:.2f} px")
    print("=" * 60)

if __name__ == '__main__':
    main()
