"""
L3D PyTorch Dataset for EXPERIMENT_8 (Human-Anchor-Steered Mask2Former).
Provides:
  - RGB images (1024x1024, normalized with ImageNet stats)
  - Dense 3-class segmentation masks (0=BG, 1=Ridge, 2=Sil, 3=Falc)
  - Ground-truth 4-junction coordinates in [0, 1]^2 loaded directly from train_biological_anchors_human.json
  - Ground-truth 4-junction binary visibility flags
  - Automatic environment resolution (local macOS vs. Cluster CUDA)
"""
import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

# Standard ImageNet statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ANCHOR_KEYS = ["J_top", "J_bottom", "J_lat_right", "J_lat_left"]

def resolve_l3d_root(candidate_root=None):
    """
    Auto-detects the L3D dataset root path across local macOS, Linux, and HPC Cluster.
    """
    candidates = [
        candidate_root,
        "data/L3D",
        "../data/L3D",
        "../../data/L3D",
        "/data/khoalq/data/L3D",
        "/kaggle/input/laparoscopic-liver-landmark-dataset/L3D",
        "/kaggle/input/l3d-dataset/L3D",
        "/kaggle/input/l3d/L3D"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath("data/L3D")

def resolve_anchor_json(candidate_path=None):
    """
    Auto-detects train_biological_anchors_human.json across local and cluster environments.
    """
    candidates = [
        candidate_path,
        "data/llm_annotate/train_biological_anchors_human.json",
        "../data/llm_annotate/train_biological_anchors_human.json",
        "../../data/llm_annotate/train_biological_anchors_human.json",
        "/data/khoalq/surgical_ai/data/llm_annotate/train_biological_anchors_human.json",
        "data/llm_annotate/train_biological_anchors.json"  # Fallback
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None

class L3DDataset(Dataset):
    def __init__(self, split='Train', data_dir=None, anchor_json_path=None, image_size=1024, stroke_width=35):
        super().__init__()
        self.split = split
        self.image_size = image_size
        self.stroke_width = stroke_width
        self.root_dir = resolve_l3d_root(data_dir)
        
        split_dir = os.path.join(self.root_dir, split)
        self.img_dir = os.path.join(split_dir, 'images')
        self.lbl_dir = os.path.join(split_dir, 'labels')
        
        self.json_files = sorted(glob.glob(os.path.join(self.lbl_dir, '*.json')))
        if len(self.json_files) == 0:
            raise RuntimeError(f"No JSON annotation files found in: {self.lbl_dir}")
            
        print(f"[{split}] Loaded {len(self.json_files)} frames from: {split_dir}")
        
        # Load Human-Refined Anchors for Training
        self.human_anchors_map = {}
        if self.split == 'Train':
            resolved_anchor_path = resolve_anchor_json(anchor_json_path)
            if resolved_anchor_path and os.path.exists(resolved_anchor_path):
                print(f"[Dataset] Loading human biological anchors from: {resolved_anchor_path}")
                with open(resolved_anchor_path, 'r', encoding='utf-8') as f:
                    anchor_data = json.load(f)
                for item in anchor_data.get('annotations', []):
                    fn = item.get('image_filename')
                    if fn:
                        self.human_anchors_map[fn] = item.get('anchors', {})
                print(f"[Dataset] Indexed {len(self.human_anchors_map)} human anchor annotations.")
            else:
                print(f"[Dataset Warning] No human anchor file found at: {resolved_anchor_path}. Defaulting to zeros.")

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
            
        # Dynamic canvas size: guarantees Patient 32 4K (2160x3840) is never truncated (Trap 3)
        data_w = data.get('imageWidth', orig_w)
        data_h = data.get('imageHeight', orig_h)
        canvas_raw = np.zeros((data_h, data_w), dtype=np.uint8)
        
        # 3. Create dense raster mask with typo-tolerant matching (Trap 1)
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

        # Downsample to network resolution (1024x1024) using INTER_NEAREST
        if data_w != self.image_size or data_h != self.image_size:
            mask = cv2.resize(canvas_raw, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask = canvas_raw

        mask_tensor = torch.from_numpy(mask).long() # (H, W)
        
        # 4. Extract 4 Biological Junctions from Human Ground Truth
        filename_jpg = f"{stem}.jpg"
        coords_arr = np.zeros((4, 2), dtype=np.float32)
        vis_arr = np.zeros((4,), dtype=np.float32)
        
        if filename_jpg in self.human_anchors_map:
            ann_anchors = self.human_anchors_map[filename_jpg]
            for k, anchor_name in enumerate(ANCHOR_KEYS):
                a = ann_anchors.get(anchor_name, {})
                v = float(a.get('v', 0))
                if v > 0.5:
                    raw_x = float(a.get('x', 0.0))
                    raw_y = float(a.get('y', 0.0))
                    # Normalize by raw image dimensions into [0, 1]^2
                    norm_x = np.clip(raw_x / float(data_w), 0.0, 1.0)
                    norm_y = np.clip(raw_y / float(data_h), 0.0, 1.0)
                    coords_arr[k] = [norm_x, norm_y]
                    vis_arr[k] = 1.0
                else:
                    coords_arr[k] = [0.0, 0.0]
                    vis_arr[k] = 0.0

        junction_coords = torch.from_numpy(coords_arr).float() # (4, 2)
        junction_vis = torch.from_numpy(vis_arr).float()       # (4,)
        
        # Patient 40 diagnostic flag (Trap 2)
        is_p40 = ('patient_40' in stem.lower() or '_40_' in stem.lower())
        
        return {
            'pixel_values': pixel_values,
            'mask': mask_tensor,
            'junction_coords': junction_coords,
            'junction_vis': junction_vis,
            'filename': filename_jpg,
            'is_patient_40': is_p40
        }
