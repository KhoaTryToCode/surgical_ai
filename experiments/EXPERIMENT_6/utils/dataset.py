"""
L3D PyTorch Dataset for EXPERIMENT_6 (Heatmap-Guided Junction-Steered Mask2Former).
Provides:
  - RGB images (1024x1024, normalized with ImageNet stats)
  - Dense 3-class segmentation masks (0=BG, 1=Ridge, 2=Sil, 3=Falc)
  - Ground-truth 4-junction continuous Gaussian heatmaps (4, 64, 64)
  - Ground-truth coordinates & visibility flags
  - Automatic environment resolution (local macOS vs. Linux cluster vs. Kaggle)
"""
import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from experiments.EXPERIMENT_6.utils.junction_extractor import extract_gt_junctions
from experiments.EXPERIMENT_6.utils.heatmap_utils import generate_gaussian_heatmaps

# Standard ImageNet statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_l3d_root(candidate_root=None):
    candidates = [
        candidate_root,
        "data/L3D",
        "../data/L3D",
        "../../data/L3D",
        "/kaggle/input/laparoscopic-liver-landmark-dataset/L3D",
        "/kaggle/input/l3d-dataset/L3D",
        "/kaggle/input/l3d/L3D",
        "/data/khoalq/data/L3D"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath("data/L3D")


class L3DDataset(Dataset):
    def __init__(self, split='Train', data_dir=None, image_size=1024, heatmap_size=64, stroke_width=35, sigma=2.0):
        super().__init__()
        self.split = split
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.stroke_width = stroke_width
        self.sigma = sigma
        self.root_dir = resolve_l3d_root(data_dir)
        
        split_dir = os.path.join(self.root_dir, split)
        self.img_dir = os.path.join(split_dir, 'images')
        self.lbl_dir = os.path.join(split_dir, 'labels')
        
        self.json_files = sorted(glob.glob(os.path.join(self.lbl_dir, '*.json')))
        if len(self.json_files) == 0:
            raise RuntimeError(f"No JSON annotation files found in: {self.lbl_dir}")
            
        print(f"[{split}] Loaded {len(self.json_files)} frames from: {split_dir}")

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        stem = Path(json_path).stem
        img_path = os.path.join(self.img_dir, f"{stem}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, f"{stem}.png")
            
        # 1. Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
            
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize image to canvas_size
        if orig_w != self.image_size or orig_h != self.image_size:
            img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img_rgb
            
        # Normalize pixel values
        norm_img = (img_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        pixel_values = torch.from_numpy(norm_img).permute(2, 0, 1).float() # (3, H, W)
        
        # 2. Parse JSON annotations
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Dynamic canvas size: guarantees Patient 32 4K (2160x3840) is never truncated
        data_w = data.get('imageWidth', orig_w)
        data_h = data.get('imageHeight', orig_h)
        canvas_raw = np.zeros((data_h, data_w), dtype=np.uint8)
        
        # 3. Create dense raster mask
        # Draw contours with thickness 35 on raw canvas (exact TopoNet / EXP_1 paper standard)
        shapes = data.get('shapes', [])
        for shape in shapes:
            lbl = str(shape.get('label', '')).lower().strip()
            pts = shape.get('points', [])
            if len(pts) < 2:
                continue
                
            if lbl.startswith('r') or 'ridge' in lbl or 'rigde' in lbl or 'anterior' in lbl:
                color = 1
            elif lbl.startswith('s') or 'sil' in lbl or 'margin' in lbl or 'silhouette' in lbl:
                color = 2
            elif lbl.startswith('f') or lbl.startswith('l') or 'falc' in lbl or 'lig' in lbl:
                color = 3
            else:
                color = 0
                
            if color > 0:
                for i in range(1, len(pts)):
                    pt1 = tuple(map(int, pts[i - 1]))
                    pt2 = tuple(map(int, pts[i]))
                    cv2.line(canvas_raw, pt1, pt2, color, self.stroke_width)

        # Downsample to network resolution (1024x1024) using INTER_NEAREST (exact TopoNet / EXP_1 standard)
        if data_w != self.image_size or data_h != self.image_size:
            mask = cv2.resize(canvas_raw, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask = canvas_raw

        mask_tensor = torch.from_numpy(mask).long() # (H, W)
        
        # 4. Extract 4 GT Anatomical Junctions
        coords_norm, vis, _ = extract_gt_junctions(data, data_w, data_h, canvas_size=self.image_size)
        junction_coords = torch.from_numpy(coords_norm).float() # (4, 2)
        junction_vis = torch.from_numpy(vis).float()           # (4,)
        
        # 5. Generate Ground-Truth Continuous Gaussian Heatmaps (4, 64, 64)
        gt_heatmaps = generate_gaussian_heatmaps(
            coords_norm=junction_coords,
            visibility=junction_vis,
            map_size=self.heatmap_size,
            sigma=self.sigma
        ) # (4, 64, 64)
        
        is_p40 = ('patient_40' in stem.lower() or '_40_' in stem.lower())
        
        return {
            'pixel_values': pixel_values,
            'mask': mask_tensor,
            'junction_coords': junction_coords,
            'junction_vis': junction_vis,
            'gt_heatmaps': gt_heatmaps,
            'filename': f"{stem}.jpg",
            'is_patient_40': is_p40
        }
