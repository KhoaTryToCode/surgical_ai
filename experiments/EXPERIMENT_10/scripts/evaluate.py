"""
Standalone Evaluation Script for EXPERIMENT_10 (Depth-Geometric Junction-Steered Mask2Former).
Evaluates:
  - Validation Split (122 frames)
  - Unseen Test Split (109 frames)
Produces:
  - metrics_summary.json
  - val_predictions.csv
  - test_predictions.csv
  - patient_40_diagnostics/ (Visualizations for difficult deformed frames)
  - results.zip
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
for _a, _v in [('Inf', np.inf), ('NAN', np.nan), ('NaN', np.nan), ('PINF', np.inf), ('NINF', -np.inf)]:
    if not hasattr(np, _a):
        setattr(np, _a, _v)

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_10.utils.dataset import L3DDataset, IMAGENET_MEAN, IMAGENET_STD
from experiments.EXPERIMENT_10.utils.metrics import evaluate_frame_metrics
from experiments.EXPERIMENT_10.models.depth_junction_steered_mask2former import load_depth_junction_steered_model

# Visualization color map (RGB)
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
    
    cls_probs = torch.softmax(class_queries_logits, dim=-1) # (100, num_classes + 1)
    fg_cls_probs = cls_probs[:, :4]                         # Exclude null class (idx 4)
    
    sem_probs = torch.einsum("qc,qhw->chw", fg_cls_probs, masks) # (4, H, W)
    pred_map = sem_probs.argmax(dim=0).cpu().numpy().astype(np.int64)
    max_prob = sem_probs.max(dim=0)[0].cpu().numpy()
    pred_map[max_prob < 0.25] = 0
    return pred_map


def render_overlay(orig_bgr, mask, alpha=0.5):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for c, rgb in COLOR_MAP_RGB.items():
        if c == 0:
            continue
        colored[mask == c] = (rgb[2], rgb[1], rgb[0]) # RGB -> BGR
    
    fg = (mask > 0)
    blended = orig_bgr.copy()
    blended[fg] = cv2.addWeighted(orig_bgr[fg], 1.0 - alpha, colored[fg], alpha, 0)
    return blended


def evaluate_split(model, dataset, device, output_dir=None, split_name="val", max_samples=None):
    model.eval()
    records = []
    diag_dir = os.path.join(output_dir, "patient_40_diagnostics") if output_dir else None
    if diag_dir and split_name == "val":
        os.makedirs(diag_dir, exist_ok=True)

    n_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    print(f"🔬 Evaluating {split_name.upper()} split ({n_samples} frames) on {device}...")

    with torch.no_grad():
        for i in range(n_samples):
            item = dataset[i]
            pixel_values = item['pixel_values'].unsqueeze(0).to(device)
            depth_map = item['depth_map'].unsqueeze(0).to(device)
            target_mask = item['mask'].numpy()
            fn = item['filename']
            is_p40 = item['is_patient_40']
            orig_bgr = item['orig_bgr'].numpy()
            
            # Forward pass
            outputs = model(pixel_values, depth_map)
            pred_map = rasterize_class_map(outputs['masks_queries_logits'][0], outputs['class_queries_logits'][0])
            
            pred_coords = outputs['pred_junction_coords'][0].cpu().numpy() # (4, 2)
            gt_coords = item['junction_coords'].numpy()                   # (4, 2)
            gt_vis = item['junction_vis'].numpy()                         # (4,)
            pred_vis_probs = outputs['pred_junction_vis_probs'][0].cpu().numpy()
            
            # Compute frame metrics
            frame_res = evaluate_frame_metrics(
                pred_map=pred_map,
                target_map=target_mask,
                pred_j_coords=pred_coords,
                gt_j_coords=gt_coords,
                gt_j_vis=gt_vis,
                canvas_size=1024
            )
            
            frame_res['filename'] = fn
            frame_res['is_patient_40'] = is_p40
            records.append(frame_res)
            
            # Render Patient 40 diagnostic montage
            if diag_dir and is_p40:
                p40_fn = os.path.join(diag_dir, f"diag_{Path(fn).stem}.jpg")
                overlay_pred = render_overlay(orig_bgr, pred_map)
                overlay_gt = render_overlay(orig_bgr, target_mask)
                
                # Draw predicted anchors (solid circles)
                for k in range(4):
                    px = int(np.clip(pred_coords[k, 0] * 1024, 0, 1023))
                    py = int(np.clip(pred_coords[k, 1] * 1024, 0, 1023))
                    col = JUNCTION_COLORS_BGR[k]
                    cv2.circle(overlay_pred, (px, py), 8, col, -1)
                    cv2.circle(overlay_pred, (px, py), 10, (255, 255, 255), 2)
                
                # Draw GT anchors on GT overlay
                for k in range(4):
                    if gt_vis[k] > 0.5:
                        gx = int(np.clip(gt_coords[k, 0] * 1024, 0, 1023))
                        gy = int(np.clip(gt_coords[k, 1] * 1024, 0, 1023))
                        col = JUNCTION_COLORS_BGR[k]
                        cv2.circle(overlay_gt, (gx, gy), 8, col, -1)
                        cv2.circle(overlay_gt, (gx, gy), 10, (0, 0, 0), 2)
                
                # 4-panel comparison: [Raw RGB | Depth | GT Mask | Predicted Mask]
                # Depth display: un-normalize depth to [0, 255] uint8
                depth_disp = item['depth_map'][0].numpy()
                depth_uint8 = np.clip((depth_disp * 0.2276 + 0.3905) * 255.0, 0, 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
                
                panel_top = np.hstack([cv2.resize(orig_bgr, (512, 512)), cv2.resize(depth_colored, (512, 512))])
                panel_bot = np.hstack([cv2.resize(overlay_gt, (512, 512)), cv2.resize(overlay_pred, (512, 512))])
                montage = np.vstack([panel_top, panel_bot])
                cv2.imwrite(p40_fn, montage)

    df = pd.DataFrame(records)
    summary = {
        'split': split_name,
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
    }
    
    p40_mask = df['is_patient_40']
    if p40_mask.sum() > 0:
        summary['patient_40_dice'] = float(df[p40_mask]['macro_dice'].mean())
        summary['patient_40_assd'] = float(df[p40_mask]['macro_assd'].mean())
        
    return df, summary


def run_evaluation(model_path=None, data_dir=None, output_dir="experiments/EXPERIMENT_10/results", device="cpu", max_samples=None):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    
    print("=" * 60)
    print("🚀 EXPERIMENT_10 Standalone Evaluation Pipeline")
    print(f"Target Device: {device}")
    print(f"Model Checkpoint: {model_path}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)
    
    model = load_depth_junction_steered_model(model_path, device=device)
    model.eval()
    
    # 1. Validation Evaluation
    val_dataset = L3DDataset(split='Val', data_dir=data_dir)
    df_val, val_summary = evaluate_split(model, val_dataset, device, output_dir=output_dir, split_name="val", max_samples=max_samples)
    val_csv_path = os.path.join(output_dir, "val_predictions.csv")
    df_val.to_csv(val_csv_path, index=False)
    print(f"✅ Saved validation predictions: {val_csv_path}")
    
    print("\n" + "=" * 50)
    print(f"📊 Validation Macro Dice : {val_summary['macro_dice']*100:.2f}%")
    print(f"📊 Validation ASSD       : {val_summary['macro_assd']:.2f} px")
    if 'patient_40_dice' in val_summary:
        print(f"📊 Patient 40 Dice       : {val_summary['patient_40_dice']*100:.2f}%")
    print("=" * 50)
    
    # 2. Test Split Evaluation (if directory exists)
    test_summary = {}
    try:
        test_dataset = L3DDataset(split='Test', data_dir=data_dir)
        df_test, test_summary = evaluate_split(model, test_dataset, device, output_dir=output_dir, split_name="test", max_samples=max_samples)
        test_csv_path = os.path.join(output_dir, "test_predictions.csv")
        df_test.to_csv(test_csv_path, index=False)
        print(f"✅ Saved test predictions: {test_csv_path}")
        print(f"📊 Test Macro Dice       : {test_summary['macro_dice']*100:.2f}%")
        print(f"📊 Test ASSD             : {test_summary['macro_assd']:.2f} px")
    except Exception as e:
        print(f"⚠️ Test split skipped or not found: {e}")
        
    # Save combined summary json
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({'val': val_summary, 'test': test_summary}, f, indent=2)
    print(f"✅ Saved metrics summary: {summary_path}")
    
    # Package into results.zip
    zip_path = os.path.join(output_dir, "results.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(summary_path, arcname="metrics_summary.json")
        zipf.write(val_csv_path, arcname="val_predictions.csv")
        if os.path.exists(os.path.join(output_dir, "test_predictions.csv")):
            zipf.write(os.path.join(output_dir, "test_predictions.csv"), arcname="test_predictions.csv")
    print(f"📦 Packaged deliverables: {zip_path}")
    return val_summary, test_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate EXPERIMENT_10 DepthJunctionSteeredMask2Former")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to best_model.pth")
    parser.add_argument('--data-dir', type=str, default=None, help="Path to L3D dataset root")
    parser.add_argument('--output-dir', type=str, default="experiments/EXPERIMENT_10/results", help="Output directory")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-samples', type=int, default=None, help="Max samples to evaluate (for smoke test)")
    args = parser.parse_args()
    
    run_evaluation(
        model_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        max_samples=args.max_samples
    )
