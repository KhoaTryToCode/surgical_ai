import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from experiments.EXPERIMENT_3.models.bezier_utils import resample_polyline, fit_bezier_to_patch

class BezierPatchDataset(Dataset):
    """
    Dataset for Bezier Patch based model.
    """
    def __init__(self, data_dir, grid_size=8, canvas_size=1024, mode='train'):
        """
        Initializes the BezierPatchDataset.
        Args:
            data_dir (str): Directory containing data.
            grid_size (int): Size of the patch grid.
            canvas_size (int): Expected canvas size.
            mode (str): train/val mode.
        """
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.canvas_size = canvas_size
        self.patch_size = canvas_size // grid_size
        self.num_patches = grid_size * grid_size
        self.mode = mode
        
        # Find all image files
        image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.image_files = []
        for ext in image_exts:
            self.image_files.extend(glob.glob(os.path.join(data_dir, ext)))
            self.image_files.extend(glob.glob(os.path.join(data_dir, 'images', ext)))
        self.image_files = sorted(self.image_files)
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        """Returns the total number of images."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns a single dataset item.
        Args:
            idx (int): Item index.
        Returns:
            tuple: (pixel_values, gt_2d, target_class, target_bezier, active_mask, filename)
        """
        img_path = self.image_files[idx]
        filename = os.path.basename(img_path)
        
        pixel_values = self._load_image(img_path)
        gt_2d, target_class, target_bezier, active_mask = self._load_targets(img_path)
        
        return pixel_values, gt_2d, target_class, target_bezier, active_mask, filename

    def _load_image(self, path):
        """
        Loads and preprocesses an image.
        Args:
            path (str): Image path.
        Returns:
            Tensor (3, H, W) float32
        """
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_LINEAR)
        
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).float()

    def _load_targets(self, path):
        """
        Loads and constructs targets for an image.
        Args:
            path (str): Image path.
        Returns:
            Tuple containing targets (gt_2d, target_class, target_bezier, active_mask).
        """
        base_dir = os.path.dirname(path)
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)
        
        json_paths = [
            os.path.join(base_dir, name + '.json'),
            os.path.join(base_dir, '..', 'labels', name + '.json'),
            os.path.join(base_dir, 'labels', name + '.json')
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
        
        if not json_path:
            return gt_2d, target_class, target_bezier, active_mask
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        orig_h = data.get('imageHeight', self.canvas_size)
        orig_w = data.get('imageWidth', self.canvas_size)
        
        scale_x = self.canvas_size / orig_w
        scale_y = self.canvas_size / orig_h
        
        gt_2d_np = np.zeros((self.canvas_size, self.canvas_size), dtype=np.int64)
        
        polylines_by_class = {1: [], 2: [], 3: []}
        
        for shape in data.get('shapes', []):
            label = shape.get('label', '').lower().strip()
            if label.startswith('r') or 'ridge' in label or 'rigde' in label:
                class_id = 1
            elif label.startswith('s') or 'sil' in label or 'margin' in label:
                class_id = 2
            elif label.startswith('f') or 'falc' in label or 'ligament' in label:
                class_id = 3
            else:
                continue
                
            points = shape.get('points', [])
            if len(points) < 2:
                continue
                
            scaled_points = []
            for pt in points:
                scaled_points.append([pt[0] * scale_x, pt[1] * scale_y])
            scaled_points = np.array(scaled_points, dtype=np.int32)
            
            # Draw segment by segment for compatibility
            for i in range(len(scaled_points) - 1):
                pt1 = tuple(scaled_points[i])
                pt2 = tuple(scaled_points[i+1])
                cv2.line(gt_2d_np, pt1, pt2, int(class_id), thickness=35)
                
            polylines_by_class[class_id].append(scaled_points)
            
        # Build Bezier patch targets
        all_resampled_points = {1: [], 2: [], 3: []}
        for cid, lines in polylines_by_class.items():
            for line in lines:
                if len(line) >= 2:
                    resampled = resample_polyline(line, spacing=5.0)
                    all_resampled_points[cid].extend(resampled)
                    
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                patch_idx = r * self.grid_size + c
                x_min = c * self.patch_size
                y_min = r * self.patch_size
                x_max = (c + 1) * self.patch_size
                y_max = (r + 1) * self.patch_size
                
                points_in_patch = {1: [], 2: [], 3: []}
                for cid, pts in all_resampled_points.items():
                    for pt in pts:
                        px, py = pt[0], pt[1]
                        if x_min <= px < x_max and y_min <= py < y_max:
                            points_in_patch[cid].append(pt)
                            
                counts = {cid: len(pts) for cid, pts in points_in_patch.items()}
                if counts:
                    dominant_class = max(counts, key=counts.get)
                    
                    if counts[dominant_class] >= 2:
                        dom_pts = np.array(points_in_patch[dominant_class])
                        # fit_bezier_to_patch expects canvas-space points + [x_min,y_min,x_max,y_max]
                        patch_bbox = [x_min, y_min, x_max, y_max]
                        bezier_ctrl = fit_bezier_to_patch(dom_pts, patch_bbox)

                        target_class[patch_idx] = dominant_class
                        target_bezier[patch_idx] = torch.from_numpy(bezier_ctrl).float()
                        active_mask[patch_idx] = True
                    
        return torch.from_numpy(gt_2d_np), target_class, target_bezier, active_mask
