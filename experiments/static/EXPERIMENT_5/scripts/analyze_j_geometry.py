"""
Geometric Analysis of Junction Vectors for EXPERIMENT_5 (Method 1 & Method 3).

Method 1: Spatial Correlation Heatmap Overlays (J_k · F(x, y))
  - Computes cosine similarity between each of the 4 J vectors and the 2D feature map.
  - Generates 4-panel heatmap overlays on raw RGB frames for normal vs. deformed cases
    (e.g. Patient_40_03660, Patient_40_08730, Patient_40_08790).

Method 3: Quantitative Linear Probing (R^2 Score & Pearson Correlation)
  - Extracts 4 J vectors across all 101 Patient 40 validation frames.
  - Computes ground-truth physical geometric attributes:
      1. Scale / Zoom (Bounding Box Diagonal)
      2. Centroid X, Centroid Y (Center of mass)
      3. Liver Tilt / Orientation Angle (PCA principal axis of Ridge)
      4. Landmark Aspect Ratio (Height / Width)
  - Trains a cross-validated linear probe to predict geometry from J vectors.
  - Computes R^2 and Pearson r to mathematically prove geometric encoding.
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import pearsonr

_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_5.utils.dataset import L3DDataset
from experiments.EXPERIMENT_5.models.junction_steered_mask2former import JunctionSteeredMask2Former

JUNCTION_NAMES = ['J0_Top (Falc-Sil)', 'J1_Bot (Umbilical Notch)', 'J2_Lat_R (Right Tip)', 'J3_Lat_L (Left Tip)']
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def compute_physical_geometry(mask_np):
    """
    Computes objective physical geometric parameters from a dense segmentation mask (1024x1024).
    Classes: 1=Ridge, 2=Silhouette, 3=Falciform.
    """
    fg = (mask_np > 0)
    if not fg.any():
        return {
            'bbox_diag': 0.0,
            'centroid_x': 0.5,
            'centroid_y': 0.5,
            'ridge_angle_deg': 0.0,
            'aspect_ratio': 1.0,
            'fg_area_px': 0
        }
    
    ys, xs = np.where(fg)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    
    bbox_w = float(x_max - x_min + 1)
    bbox_h = float(y_max - y_min + 1)
    bbox_diag = float(np.sqrt(bbox_w**2 + bbox_h**2))
    aspect_ratio = float(bbox_h / (bbox_w + 1e-6))
    
    centroid_x = float(np.mean(xs)) / 1024.0
    centroid_y = float(np.mean(ys)) / 1024.0
    
    # Compute Ridge principal orientation angle
    ridge_mask = (mask_np == 1)
    if ridge_mask.sum() > 20:
        r_ys, r_xs = np.where(ridge_mask)
        pts = np.column_stack([r_xs, r_ys]).astype(np.float32)
        pts -= np.mean(pts, axis=0)
        cov = np.cov(pts, rowvar=False)
        # Angle of primary eigenvector
        eigvals, eigvecs = np.linalg.eigh(cov)
        v = eigvecs[:, np.argmax(eigvals)]
        angle_deg = float(np.degrees(np.arctan2(v[1], v[0])))
    else:
        angle_deg = 0.0
        
    return {
        'bbox_diag': bbox_diag,
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'ridge_angle_deg': angle_deg,
        'aspect_ratio': aspect_ratio,
        'fg_area_px': int(fg.sum())
    }


def render_method1_heatmaps(model, sample, out_path, device):
    """
    Method 1: Renders a 4-panel spatial correlation heatmap (J_k · F(x, y)) overlaid on RGB.
    """
    pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
    orig_rgb_norm = sample['pixel_values'].cpu().numpy()
    gt_mask = sample['mask'].numpy()
    gt_coords = sample['junction_coords'].numpy() # (4, 2)
    gt_vis = sample['junction_vis'].numpy()       # (4,)
    filename = sample['filename']
    
    # Unnormalize RGB for visualization
    rgb = (orig_rgb_norm.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H_img, W_img = bgr.shape[:2]
    
    with torch.no_grad():
        pixel_outputs = model.m2f.model.pixel_level_module(pixel_values, output_hidden_states=True)
        # stride-16 feature map (1, 256, 64, 64)
        feat = pixel_outputs.decoder_hidden_states[1]
        pred_coords, pred_vis, j_features = model.junction_head(feat)
        
    # feat: (1, 256, H_f, W_f)
    # j_features: (1, 4, 256)
    B, C, H_f, W_f = feat.shape
    feat_norm = F.normalize(feat, p=2, dim=1) # (1, 256, H_f, W_f)
    j_norm = F.normalize(j_features, p=2, dim=-1) # (1, 4, 256)
    
    # Spatial Cosine Similarity: einsum (1, 4, 256) x (1, 256, H_f, W_f) -> (1, 4, H_f, W_f)
    cos_sim = torch.einsum("bkc,bchw->bkhw", j_norm, feat_norm).squeeze(0) # (4, H_f, W_f)
    
    # Render 4 panels
    panels = []
    for k in range(4):
        sim_map = cos_sim[k].cpu().numpy()
        # Min-max normalize per channel
        s_min, s_max = sim_map.min(), sim_map.max()
        sim_norm = (sim_map - s_min) / (s_max - s_min + 1e-6)
        
        # Resize to 1024x1024
        heat_full = cv2.resize(sim_norm, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
        heat_u8 = np.clip(heat_full * 255.0, 0, 255).astype(np.uint8)
        color_map = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        
        # Alpha blend with original BGR image
        blend = cv2.addWeighted(bgr, 0.45, color_map, 0.55, 0)
        
        # Find peak location in heatmap
        peak_y, peak_x = np.unravel_index(np.argmax(heat_full), heat_full.shape)
        cv2.drawMarker(blend, (peak_x, peak_y), (0, 255, 255), cv2.MARKER_CROSS, 28, 3)
        cv2.putText(blend, "Heatmap Peak", (peak_x + 15, peak_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw GT junction if visible
        if gt_vis[k] > 0.5:
            gt_px = (int(gt_coords[k, 0] * W_img), int(gt_coords[k, 1] * H_img))
            cv2.circle(blend, gt_px, 14, (0, 0, 0), -1)
            cv2.circle(blend, gt_px, 10, (0, 255, 0), -1) # Green = True GT
            cv2.circle(blend, gt_px, 3, (255, 255, 255), -1)
            dist_err = float(np.linalg.norm(np.array([peak_x, peak_y]) - np.array(gt_px)))
            status_txt = f"GT: Visible (Peak Err: {dist_err:.1f}px)"
        else:
            status_txt = "GT: OFF-SCREEN / ABSENT"
            
        # Draw title header
        title = f"{JUNCTION_NAMES[k]}"
        cv2.putText(blend, title, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(blend, title, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
        
        cv2.putText(blend, status_txt, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200) if gt_vis[k] > 0.5 else (0, 140, 255), 2)
        panels.append(blend)
        
    # Stitch 2x2 grid
    top_row = np.hstack([panels[0], panels[1]])
    bot_row = np.hstack([panels[2], panels[3]])
    montage = np.vstack([top_row, bot_row])
    
    # Save image
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, montage, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"📸 Saved Method 1 Heatmap Montage: {out_path}")


def run_method3_linear_probing(j_matrix, geom_df, out_dir):
    """
    Method 3: Quantitative Linear Probing (R^2 & Pearson correlation).
    j_matrix: (N, 4, 256) or (N, 1024)
    geom_df: DataFrame of physical attributes across N frames.
    """
    N = j_matrix.shape[0]
    X_all = j_matrix.reshape(N, -1) # (N, 1024)
    
    targets = {
        'Camera Zoom / Scale (BBox Diag)': geom_df['bbox_diag'].values,
        'Centroid X Position': geom_df['centroid_x'].values,
        'Centroid Y Position': geom_df['centroid_y'].values,
        'Liver Ridge Tilt Angle (deg)': geom_df['ridge_angle_deg'].values,
        'Landmark Aspect Ratio (H/W)': geom_df['aspect_ratio'].values,
        'Foreground Area (px)': geom_df['fg_area_px'].values
    }
    
    results = []
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    
    # Add bias column to X_all: (N, 1025)
    X_homo = np.column_stack([X_all, np.ones(N, dtype=np.float32)])
    D = X_homo.shape[1]
    
    for target_name, y in targets.items():
        y = np.array(y, dtype=np.float32)
        y_preds = np.zeros_like(y)
        
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.setdiff1d(indices, val_idx)
            
            X_tr, y_tr = X_homo[train_idx], y[train_idx]
            X_te = X_homo[val_idx]
            
            # Ridge analytical solution: w = (X^T X + alpha * I)^(-1) X^T y
            alpha = 10.0
            reg_matrix = X_tr.T @ X_tr + alpha * np.eye(D, dtype=np.float32)
            reg_matrix[-1, -1] -= alpha  # do not regularize bias
            w = np.linalg.solve(reg_matrix, X_tr.T @ y_tr)
            y_preds[val_idx] = X_te @ w
            
        # Compute R^2 score
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_preds)**2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-6))
        
        # Pearson correlation
        r, p_val = pearsonr(y, y_preds)
        
        results.append({
            'Geometric Attribute': target_name,
            'Cross-Val R^2 Score': float(r2),
            'Pearson Correlation (r)': float(r),
            'P-Value': float(p_val),
            'Linear Decodable': "YES (Strong)" if r2 > 0.60 else ("MODERATE" if r2 > 0.30 else "WEAK")
        })
        
    df_res = pd.DataFrame(results)
    
    # Save outputs
    csv_path = os.path.join(out_dir, 'method3_linear_probing_results.csv')
    df_res.to_csv(csv_path, index=False)
    
    json_path = os.path.join(out_dir, 'method3_linear_probing_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*75)
    print("📊 METHOD 3: LINEAR PROBING RESULTS (Predicting Geometry from J-Vectors)")
    print("="*75)
    print(df_res.to_string(index=False))
    print("="*75)
    print(f"💾 Results saved to: {csv_path}")
    
    return df_res


def main():
    parser = argparse.ArgumentParser(description="Analyze geometric correlation of J vectors (EXP_5)")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to best_model.pth")
    parser.add_argument('--data_dir', type=str, default=None, help="Path to L3D root")
    parser.add_argument('--out_dir', type=str, default=None, help="Output directory")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Resolve paths
    if args.out_dir is None:
        args.out_dir = os.path.join(_WORKSPACE_ROOT, 'experiments/EXPERIMENT_5/results/geometry_analysis')
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Auto-detect checkpoint
    if args.checkpoint is None:
        candidates = [
            os.path.join(_WORKSPACE_ROOT, 'experiments/EXPERIMENT_5/results/results/best_model.pth'),
            os.path.join(_WORKSPACE_ROOT, 'experiments/EXPERIMENT_5/results/best_model.pth'),
            '/data/khoalq/checkpoints/exp5_junction_m2f/best_model.pth'
        ]
        for c in candidates:
            if os.path.exists(c):
                args.checkpoint = c
                break
                
    if args.checkpoint is None or not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
        
    device = torch.device(args.device)
    print(f"🖥️ Using device: {device}")
    print(f"📦 Loading model checkpoint from: {args.checkpoint}")
    
    # 1. Load Model
    model = JunctionSteeredMask2Former(num_labels=4).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("✅ Model weights loaded successfully!")
    
    # 2. Load Validation Dataset
    val_dataset = L3DDataset(split='Val', data_dir=args.data_dir)
    
    # Filter for Patient 40 frames
    p40_indices = [i for i, s in enumerate(val_dataset.json_files) if ('patient_40' in s.lower() or '_40_' in s.lower())]
    print(f"🔎 Found {len(p40_indices)} Patient 40 frames for geometric correlation testing.")
    
    # Target frames for Method 1 Heatmap Overlays
    target_stems = ['Patient_40_03660', 'Patient_40_08730', 'Patient_40_08790', 'Patient_40_149310']
    
    all_j_vectors = []
    geom_records = []
    
    print("\n🚀 Extracting J-vectors and physical geometry across Patient 40 frames...")
    
    for count, idx in enumerate(p40_indices):
        sample = val_dataset[idx]
        stem = Path(sample['filename']).stem
        
        # Method 1: Generate Heatmaps for key comparison frames
        if any(ts in stem for ts in target_stems):
            hm_out_path = os.path.join(args.out_dir, f"method1_heatmaps_{stem}.jpg")
            render_method1_heatmaps(model, sample, hm_out_path, device)
            
        # Extract J-vectors
        pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
        with torch.no_grad():
            pixel_outputs = model.m2f.model.pixel_level_module(pixel_values, output_hidden_states=True)
            feat = pixel_outputs.decoder_hidden_states[1] # (1, 256, 64, 64)
            _, _, j_features = model.junction_head(feat)   # (1, 4, 256)
            
        j_vec = j_features.squeeze(0).cpu().numpy() # (4, 256)
        all_j_vectors.append(j_vec)
        
        # Compute physical ground-truth geometry
        geom = compute_physical_geometry(sample['mask'].numpy())
        geom['filename'] = sample['filename']
        geom_records.append(geom)
        
        if (count + 1) % 25 == 0 or (count + 1) == len(p40_indices):
            print(f"   Processed [{count+1}/{len(p40_indices)}] frames...")
            
    j_matrix = np.array(all_j_vectors) # (101, 4, 256)
    geom_df = pd.DataFrame(geom_records)
    
    # 3. Run Method 3: Linear Probing
    df_res = run_method3_linear_probing(j_matrix, geom_df, args.out_dir)
    
    print("\n🎉 Geometric Analysis Complete! Outputs saved in:")
    print(f"   {args.out_dir}")

if __name__ == '__main__':
    main()
