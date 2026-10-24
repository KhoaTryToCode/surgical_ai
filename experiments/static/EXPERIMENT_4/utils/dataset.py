#!/usr/bin/env python3
"""
EXPERIMENT_4: Dataset loader with Landmark Ground Truth Generation
Produces:
  - pixel_values: (3, 1024, 1024) normalized RGB tensor
  - gt_2d: (1024, 1024) ground truth integer mask (thickness=35, native canvas resized nearest)
  - target_class: (64,) patch class assignments
  - target_bezier: (64, 4, 2) local Bézier control points in [0, 1]^2
  - active_mask: (64,) boolean mask of active patches
  - target_lm_presence: (3,) float {0, 1} indicating presence of Ridge (0), Silhouette (1), Falciform (2)
  - target_lm_centroid: (3, 2) float coordinates in [0, 1]^2
  - filename: basename of the image
"""
import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from experiments.EXPERIMENT_4.models.bezier_utils import resample_polyline, fit_bezier_to_patch

class LandmarkBezierDataset(Dataset):
    """
    Surgical Landmark Dataset producing local Bézier patch targets and global landmark centroids.
    """
    def __init__(self, data_dir, grid_size=8, canvas_size=1024):
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.canvas_size = canvas_size
        self.patch_size = canvas_size // grid_size
        self.num_patches = grid_size * grid_size
        
        image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.image_files = []
        for ext in image_exts:
            self.image_files.extend(glob.glob(os.path.join(data_dir, ext)))
            self.image_files.extend(glob.glob(os.path.join(data_dir, 'images', ext)))
        self.image_files = sorted(list(set(self.image_files)))
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        filename = os.path.basename(img_path)
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.canvas_size or img.shape[1] != self.canvas_size:
            img = cv2.resize(img, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_LINEAR)
            
        img_norm = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        pixel_values = torch.from_numpy(np.transpose(img_norm, (2, 0, 1))).float()
        
        gt_2d, target_class, target_bezier, active_mask, lm_presence, lm_centroid = self._load_targets(img_path)
        
        return pixel_values, gt_2d, target_class, target_bezier, active_mask, lm_presence, lm_centroid, filename

    def _load_targets(self, path):
        base_dir = os.path.dirname(path)
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)
        
        json_paths = [
            os.path.join(base_dir, name + '.json'),
            os.path.join(base_dir, '..', 'labels', name + '.json'),
            os.path.join(base_dir, 'labels', name + '.json'),
            os.path.join(base_dir.replace('images', 'labels'), name + '.json')
        ]
        
        json_path = None
        for p in json_paths:
            if os.path.exists(p):
                json_path = p
                break
                
        # Default empty targets
        gt_2d = torch.zeros((self.canvas_size, self.canvas_size), dtype=torch.int64)
        target_class = torch.zeros(self.num_patches, dtype=torch.int64)
        target_bezier = torch.zeros((self.num_patches, 4, 2), dtype=torch.float32)
        active_mask = torch.zeros(self.num_patches, dtype=torch.bool)
        target_lm_presence = torch.zeros(3, dtype=torch.float32)
        target_lm_centroid = torch.zeros((3, 2), dtype=torch.float32)
        
        if not json_path:
            return gt_2d, target_class, target_bezier, active_mask, target_lm_presence, target_lm_centroid
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # 1. Native canvas resolution extraction (Patient 32 4K canvas preservation)
        orig_h = data.get('imageHeight', 1080)
        orig_w = data.get('imageWidth', 1920)
        canvas_raw = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        scale_x = self.canvas_size / float(orig_w)
        scale_y = self.canvas_size / float(orig_h)
        
        curves_by_class = {1: [], 2: [], 3: []}
        all_points_by_class = {1: [], 2: [], 3: []}
        
        for shape in data.get('shapes', []):
            label = str(shape.get('label', '')).lower().strip()
            if label.startswith('r') or 'ridge' in label or 'rigde' in label:
                class_id = 1
            elif label.startswith('s') or 'sil' in label or 'margin' in label:
                class_id = 2
            elif label.startswith('f') or 'falc' in label or 'lig' in label or 'ligament' in label:
                class_id = 3
            else:
                continue
                
            points = shape.get('points', [])
            if len(points) < 2:
                continue
                
            # Draw on native canvas with thickness 35 (matching benchmark)
            for i in range(1, len(points)):
                pt1 = tuple(map(int, points[i - 1]))
                pt2 = tuple(map(int, points[i]))
                cv2.line(canvas_raw, pt1, pt2, int(class_id), thickness=35)
                
            # Scaled points for 1024x1024 canvas
            scaled_line = np.array([[p[0] * scale_x, p[1] * scale_y] for p in points], dtype=np.float32)
            resampled = resample_polyline(scaled_line, spacing=5.0)
            if len(resampled) >= 2:
                curves_by_class[class_id].append(resampled)
                # Store normalized coordinates in [0, 1]^2 for centroid calculation
                norm_pts = np.array([[p[0] / orig_w, p[1] / orig_h] for p in points], dtype=np.float32)
                all_points_by_class[class_id].append(norm_pts)
                
        # 2. Downsample ground truth canvas to 1024x1024 via nearest neighbor
        if canvas_raw.shape[0] != self.canvas_size or canvas_raw.shape[1] != self.canvas_size:
            gt_2d_np = cv2.resize(canvas_raw, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_NEAREST)
        else:
            gt_2d_np = canvas_raw
            
        # 3. Compute Ground Truth Landmark Centroids & Presence
        for cid in [1, 2, 3]:
            idx = cid - 1 # 0: Ridge, 1: Silhouette, 2: Falciform
            if len(all_points_by_class[cid]) > 0:
                cat_pts = np.concatenate(all_points_by_class[cid], axis=0)
                mean_coord = cat_pts.mean(axis=0) # [cx, cy]
                target_lm_presence[idx] = 1.0
                target_lm_centroid[idx] = torch.tensor(mean_coord, dtype=torch.float32)
            else:
                target_lm_presence[idx] = 0.0
                target_lm_centroid[idx] = torch.zeros(2, dtype=torch.float32)
                
        # 4. Build local Bézier patch targets
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                patch_idx = r * self.grid_size + c
                x_min = c * self.patch_size
                y_min = r * self.patch_size
                x_max = (c + 1) * self.patch_size
                y_max = (r + 1) * self.patch_size
                patch_bbox = [x_min, y_min, x_max, y_max]
                
                best_curve_pts = None
                best_class_id = 0
                max_pts_count = 0
                
                for cid in [1, 2, 3]:
                    for curve in curves_by_class[cid]:
                        inside_mask = (
                            (curve[:, 0] >= x_min) & (curve[:, 0] < x_max) &
                            (curve[:, 1] >= y_min) & (curve[:, 1] < y_max)
                        )
                        pts_inside = curve[inside_mask]
                        if len(pts_inside) > max_pts_count:
                            max_pts_count = len(pts_inside)
                            best_curve_pts = pts_inside
                            best_class_id = cid
                            
                if best_curve_pts is not None and max_pts_count >= 2:
                    bezier_ctrl = fit_bezier_to_patch(best_curve_pts, patch_bbox)
                    target_class[patch_idx] = best_class_id
                    target_bezier[patch_idx] = torch.from_numpy(bezier_ctrl).float()
                    active_mask[patch_idx] = True
                    
        return (torch.from_numpy(gt_2d_np.astype(np.int64)),
                target_class,
                target_bezier,
                active_mask,
                target_lm_presence,
                target_lm_centroid)
