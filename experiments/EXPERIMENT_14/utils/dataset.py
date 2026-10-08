"""
Stratified L3D PyTorch Dataset for EXPERIMENT_14 (Option 1 Re-stratified Junction-Steered Mask2Former).
Patient-level Re-stratification (Option 1: Patient_40 <-> Patient_38 Swap):
  - Train: Contains Patient_40 (101 frames) + all other non-test patients except Patient_38.
           Total: 923 frames, 83 flipped (8.99% vs population 8.92%).
  - Val:   Contains Patient_38 (99 frames), Patient_32 (15 frames), Patient_18 (6 frames).
           Total: 120 frames, 10 flipped (8.33% vs population 8.92%).
  - Test:  Official frozen test benchmark (Patients 21, 31, 41, 51). Total: 109 frames.

Features:
  - Preserves 100% architectural and metric compatibility with EXPERIMENT_5.
  - Zero disk mutation: reads from original Train and Val directories on disk without copying files.
  - Automatic environment resolution (local macOS vs. Kaggle CUDA).
"""
import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from experiments.EXPERIMENT_14.utils.junction_extractor import extract_gt_junctions

# Standard ImageNet statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def resolve_l3d_root(candidate_root=None):
    """
    Auto-detects the L3D dataset root path across local macOS, Linux, and Kaggle.
    """
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

class StratifiedL3DDataset(Dataset):
    """
    Stratified L3D Dataset for EXPERIMENT_14.
    Rebalances Train and Val anatomical pose distribution by swapping Patient_40 and Patient_38.
    """
    VAL_PATIENTS = {'patient_38', 'patient_32', 'patient_18'}
    
    def __init__(self, split='Train', data_dir=None, image_size=1024, stroke_width=35):
        super().__init__()
        self.split = split
        self.image_size = image_size
        self.stroke_width = stroke_width
        self.root_dir = resolve_l3d_root(data_dir)
        
        if split == 'Test':
            split_dir = os.path.join(self.root_dir, 'Test')
            self.items = []
            for jf in sorted(glob.glob(os.path.join(split_dir, 'labels', '*.json'))):
                stem = Path(jf).stem
                img_path = os.path.join(split_dir, 'images', f"{stem}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(split_dir, 'images', f"{stem}.png")
                self.items.append((jf, img_path))
        else:
            # Pool all Train and Val files from disk without altering original filesystem
            all_files = []
            for sp in ['Train', 'Val']:
                sp_dir = os.path.join(self.root_dir, sp)
                for jf in glob.glob(os.path.join(sp_dir, 'labels', '*.json')):
                    stem = Path(jf).stem
                    img_path = os.path.join(sp_dir, 'images', f"{stem}.jpg")
                    if not os.path.exists(img_path):
                        img_path = os.path.join(sp_dir, 'images', f"{stem}.png")
                    all_files.append((jf, img_path))
            
            self.items = []
            for jf, img_path in sorted(all_files):
                stem = Path(jf).stem.lower()
                is_val = any(vp in stem for vp in self.VAL_PATIENTS)
                if split == 'Val' and is_val:
                    self.items.append((jf, img_path))
                elif split == 'Train' and not is_val:
                    self.items.append((jf, img_path))
                    
        if len(self.items) == 0:
            raise RuntimeError(f"[{split}] No annotation files found in root: {self.root_dir}")
            
        print(f"[{split} - Stratified Option 1] Loaded {len(self.items)} frames from root: {self.root_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        json_path, img_path = self.items[idx]
        stem = Path(json_path).stem
            
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
        
        is_p40 = ('patient_40' in stem.lower() or '_40_' in stem.lower())
        is_p38 = ('patient_38' in stem.lower() or '_38_' in stem.lower())
        
        return {
            'pixel_values': pixel_values,
            'mask': mask_tensor,
            'junction_coords': junction_coords,
            'junction_vis': junction_vis,
            'filename': f"{stem}.jpg",
            'is_patient_40': is_p40,
            'is_patient_38': is_p38
        }

# Alias for drop-in compatibility
L3DDataset = StratifiedL3DDataset
