"""
Standalone Evaluation Script for EXPERIMENT_13 (Junction-Steered Mask2Former on L3D-2K).
Controlled single-variable replication of EXPERIMENT_5.

Evaluates:
  - L3D-2K Validation Split (230 frames)
  - L3D-2K Test Split (238 frames)
  - Optional Original L3D Benchmark Test Split (109 frames)
Produces:
  - metrics_summary.json
  - val_predictions.csv
  - test_predictions.csv
  - patient_40_diagnostics/ (4-panel diagnostic visualizations for Patient 40 only)
  - results.zip archive
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

# NumPy 2.0 monkeypatch
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
    np.PINF = np.inf
    np.NINF = -np.inf

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_13.utils.dataset import L3D2KDataset, IMAGENET_MEAN, IMAGENET_STD
from experiments.EXPERIMENT_13.utils.metrics import evaluate_frame_metrics
from experiments.EXPERIMENT_13.models.junction_steered_mask2former import load_junction_steered_model

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
    ).squeeze(0).sigmoid()  # (100, H, W)
    
    # 2. Query classification probabilities
    cls_probs = torch.softmax(class_queries_logits, dim=-1)  # (100, 5), last is BG
    fg_cls_probs = cls_probs[:, :4]  # (100, 4)
    
    # Multiply: (100, 4, 1, 1) * (100, 1, H, W) -> (100, 4, H, W)
    sem_probs = torch.einsum("qc,qhw->chw", fg_cls_probs, masks)  # (4, H, W)
    
    pred_map = sem_probs.argmax(dim=0).cpu().numpy().astype(np.int64)  # (H, W)
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
            col = COLOR_MAP_RGB[c_id][::-1]  # RGB to BGR
            p2[mask_c] = (p2[mask_c] * 0.4 + np.array(col) * 0.6).astype(np.uint8)
    # Draw GT Junctions
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
    # Draw Pred Junctions
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
    
    p4[tp] = (p4[tp] * 0.3 + np.array([0, 255, 0]) * 0.7).astype(np.uint8)    # TP: Green
    p4[fp] = (p4[fp] * 0.3 + np.array([255, 255, 0]) * 0.7).astype(np.uint8)  # FP: Cyan
    p4[fn] = (p4[fn] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)    # FN: Red
    
    # Stitch 2x2 grid
    top_row = np.hstack([bgr, p2])
    bot_row = np.hstack([p3, p4])
    montage = np.vstack([top_row, bot_row])
    
    # Add title headers
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
            masks = batch['mask'].numpy()  # (B, H, W)
            gt_j_coords = batch['junction_coords'].numpy()  # (B, 4, 2)
            gt_j_vis = batch['junction_vis'].numpy()  # (B, 4)
            filenames = batch['filename']
            is_p40s = batch['is_patient_40']
            
            t0 = time.time()
            use_amp = (device.type == 'cuda')
            amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                outputs = model(pixel_values)
            latencies.append((time.time() - t0) * 1000.0)
            
            pred_masks_logits = outputs['masks_queries_logits']  # (B, 100, H/4, W/4)
            pred_cls_logits = outputs['class_queries_logits']    # (B, 100, 5)
            pred_j_coords_t = outputs['pred_junction_coords'].float().cpu().numpy()  # (B, 4, 2)
            
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
                
                # Render 4-panel diagnostic for Patient 40 only
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
    
    # Compute summary
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
    
    # Patient 40 breakdown
    p40_df = df[df['is_patient_40']]
    if len(p40_df) > 0:
        summary['patient_40_dice'] = float(p40_df['macro_dice'].mean())
        summary['patient_40_assd'] = float(p40_df['macro_assd'].mean())
        summary['patient_40_count'] = len(p40_df)
    else:
        summary['patient_40_dice'] = 0.0
        summary['patient_40_assd'] = 80.0
        summary['patient_40_count'] = 0
        
    # Junction error breakdown
    valid_j = df['mean_j_err_px'].dropna()
    if len(valid_j) > 0:
        summary['mean_junction_err_px'] = float(valid_j.mean())
        
    return summary, df


def main():
    parser = argparse.ArgumentParser(description="Evaluate EXPERIMENT_13 Junction-Steered Mask2Former on L3D-2K")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to best_model.pth checkpoint")
    parser.add_argument('--data_dir', type=str, default=None, help="Path to L3D-2K dataset root")
    parser.add_argument('--out_dir', type=str, default=None, help="Output directory for results and diagnostics")
    parser.add_argument('--batch_size', type=int, default=2, help="Evaluation batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader workers")
    parser.add_argument('--eval_original_l3d', type=str, default=None, help="Optional path to original L3D for benchmark cross-eval")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = os.path.join(_WORKSPACE_ROOT, 'experiments/EXPERIMENT_13/results')
    os.makedirs(args.out_dir, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"🖥️ Using device: {device}")
    
    # 1. Load Model
    model = load_junction_steered_model(args.checkpoint, device=device)
    
    # 2. Validation Set (L3D-2K)
    val_dataset = L3D2KDataset(split='val', data_dir=args.data_dir)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_summary, val_df = run_evaluation(model, val_loader, device, split_name="Val", out_dir=args.out_dir)
    
    val_csv_path = os.path.join(args.out_dir, 'val_predictions.csv')
    val_df.to_csv(val_csv_path, index=False)
    print(f"💾 Saved Validation predictions: {val_csv_path}")
    
    # 3. Test Set (L3D-2K)
    test_dataset = L3D2KDataset(split='test', data_dir=args.data_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_summary, test_df = run_evaluation(model, test_loader, device, split_name="Test", out_dir=args.out_dir)
    
    test_csv_path = os.path.join(args.out_dir, 'test_predictions.csv')
    test_df.to_csv(test_csv_path, index=False)
    print(f"💾 Saved Test predictions: {test_csv_path}")
    
    full_summary = {
        'val_summary': val_summary,
        'test_summary': test_summary,
        'checkpoint_path': args.checkpoint
    }
    
    # 4. Optional Cross-Evaluation on Original L3D Benchmark Test Set
    if args.eval_original_l3d and os.path.exists(args.eval_original_l3d):
        print(f"\n🔬 Running Cross-Evaluation on Original L3D Benchmark ({args.eval_original_l3d})...")
        l3d_test_ds = L3D2KDataset(split='Test', data_dir=args.eval_original_l3d)
        l3d_test_loader = DataLoader(l3d_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        orig_test_summary, orig_test_df = run_evaluation(model, l3d_test_loader, device, split_name="OrigL3D-Test", out_dir=args.out_dir)
        orig_test_csv = os.path.join(args.out_dir, 'original_l3d_test_predictions.csv')
        orig_test_df.to_csv(orig_test_csv, index=False)
        full_summary['original_l3d_test_summary'] = orig_test_summary
    
    # 5. Save JSON Summary
    summary_path = os.path.join(args.out_dir, 'metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(full_summary, f, indent=4)
    print(f"🎉 Complete Metrics Summary saved to: {summary_path}")
    
    # 6. Package results into results.zip (excluding .pth checkpoint weights)
    zip_path = os.path.join(args.out_dir, 'results.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(args.out_dir):
            for file in files:
                if file.endswith('.zip') or file.endswith('.pth'):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, args.out_dir)
                zipf.write(file_path, arcname)
    print(f"📦 Results archive packaged: {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")


if __name__ == '__main__':
    main()
