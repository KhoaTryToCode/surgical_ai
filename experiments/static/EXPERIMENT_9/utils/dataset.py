"""
L3D PyTorch Dataset for EXPERIMENT_9 (RGB-D Human-Anchor-Steered Mask2Former).

Features:
  - 4-Channel Input: RGB + Depth Anything v2 relative depth map (1024x1024)
  - Normalized with ImageNet RGB stats and empirical depth stats:
      RGB:   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      Depth: mean=0.3905,                std=0.2276
  - Dense 3-class segmentation masks (0=BG, 1=Ridge, 2=Sil, 3=Falc) with stroke_width=35
  - Ground-truth 4-junction coordinates in [0, 1]^2 from train_biological_anchors_human.json
  - Ground-truth 4-junction binary visibility flags
  - Automatic environment resolution (local macOS vs. Cluster CUDA /data/khoalq vs Kaggle)
"""
import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

# Standard ImageNet statistics for RGB
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Empirical statistics for Depth Anything v2 on L3D
DEPTH_MEAN = 0.3905
DEPTH_STD = 0.2276

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


def resolve_depth_dir(root_dir, split):
    """
    Auto-detects depth map directory for Depth Anything v2.
    """
    split_lower = split.lower()
    candidates = [
        os.path.join(root_dir, "Depth", split_lower, "depth_anything_v2"),
        os.path.join(root_dir, "depth", split_lower, "depth_anything_v2"),
        os.path.join(root_dir, "..", "Depth", split_lower, "depth_anything_v2"),
        f"/data/khoalq/data/L3D/Depth/{split_lower}/depth_anything_v2",
        f"/kaggle/input/laparoscopic-liver-landmark-dataset/L3D/Depth/{split_lower}/depth_anything_v2",
        f"/kaggle/input/l3d-dataset/L3D/Depth/{split_lower}/depth_anything_v2",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.join(root_dir, "Depth", split_lower, "depth_anything_v2")


def resolve_anchor_json(candidate_path=None):
    """
    Auto-detects train_biological_anchors_human.json across local and cluster environments.
    """
    candidates = [
        candidate_path,
        "/data/khoalq/data/L3D/train_biological_anchors_human.json",
        "/data/khoalq/surgical_ai/data/llm_annotate/train_biological_anchors_human.json",
        "data/llm_annotate/train_biological_anchors_human.json",
        "../data/llm_annotate/train_biological_anchors_human.json",
        "../../data/llm_annotate/train_biological_anchors_human.json",
        "/data/khoalq/data/train_biological_anchors_human.json",
        "data/llm_annotate/train_biological_anchors.json"  # Fallback
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


class L3DRGBDDataset(Dataset):
    def __init__(self, split='Train', data_dir=None, anchor_json_path=None, image_size=1024, stroke_width=35):
        super().__init__()
        self.split = split
        self.image_size = image_size
        self.stroke_width = stroke_width
        self.root_dir = resolve_l3d_root(data_dir)
        
        split_dir = os.path.join(self.root_dir, split)
        self.img_dir = os.path.join(split_dir, 'images')
        self.lbl_dir = os.path.join(split_dir, 'labels')
        self.depth_dir = resolve_depth_dir(self.root_dir, split)
        
        self.json_files = sorted(glob.glob(os.path.join(self.lbl_dir, '*.json')))
        if len(self.json_files) == 0:
            raise RuntimeError(f"No JSON annotation files found in: {self.lbl_dir}")
            
        print(f"[{split}] Loaded {len(self.json_files)} frames from: {split_dir}")
        print(f"[{split}] Depth maps directory: {self.depth_dir} (Exists: {os.path.exists(self.depth_dir)})")
        
        # Load Human-Refined Anchors for Training (identical to EXP_8)
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
        
        # 1. Load RGB Image
        img_path = os.path.join(self.img_dir, f"{stem}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, f"{stem}.png")
            
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
            
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if orig_w != self.image_size or orig_h != self.image_size:
            img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img_rgb
            
        # Normalize RGB values
        norm_rgb = (img_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        rgb_tensor = torch.from_numpy(norm_rgb).permute(2, 0, 1).float() # (3, H, W)
        
        # 2. Load Depth Map (Depth Anything v2)
        depth_path = os.path.join(self.depth_dir, f"{stem}.png")
        if not os.path.exists(depth_path):
            depth_path = os.path.join(self.depth_dir, f"{stem}.jpg")
            
        if os.path.exists(depth_path):
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            if depth_raw.shape[0] != self.image_size or depth_raw.shape[1] != self.image_size:
                depth_raw = cv2.resize(depth_raw, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        else:
            # Fallback: Zero depth with warning
            depth_raw = np.full((self.image_size, self.image_size), int(DEPTH_MEAN * 255.0), dtype=np.uint8)
            
        # Normalize Depth values
        norm_depth = (depth_raw.astype(np.float32) / 255.0 - DEPTH_MEAN) / DEPTH_STD
        depth_tensor = torch.from_numpy(norm_depth).unsqueeze(0).float() # (1, H, W)
        
        # 3. Concatenate to 4-channel RGB-D Tensor (4, H, W)
        pixel_values = torch.cat([rgb_tensor, depth_tensor], dim=0) # (4, H, W)
        
        # 4. Parse JSON annotations
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Dynamic canvas size: guarantees Patient 32 4K (2160x3840) is never truncated
        data_w = data.get('imageWidth', orig_w)
        data_h = data.get('imageHeight', orig_h)
        canvas_raw = np.zeros((data_h, data_w), dtype=np.uint8)
        
        # Create dense raster mask with typo-tolerant matching
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
        
        # 5. Extract 4 Biological Junctions from Human Ground Truth (EXP_8 setup)
        img_fn = f"{stem}.jpg"
        human_anchors = self.human_anchors_map.get(img_fn, {})
        
        coords = np.zeros((4, 2), dtype=np.float32)
        vis = np.zeros(4, dtype=np.float32)
        
        for k_idx, k_name in enumerate(ANCHOR_KEYS):
            if k_name in human_anchors:
                info = human_anchors[k_name]
                if info.get('visible', False) and 'u' in info and 'v' in info:
                    coords[k_idx, 0] = float(np.clip(info['u'], 0.0, 1.0))
                    coords[k_idx, 1] = float(np.clip(info['v'], 0.0, 1.0))
                    vis[k_idx] = 1.0
                    
        coords_tensor = torch.from_numpy(coords).float()
        vis_tensor = torch.from_numpy(vis).float()
        
        is_patient_40 = "Patient_40" in stem or "patient_40" in stem
        
        return {
            'pixel_values': pixel_values,       # (4, 1024, 1024) RGB-D
            'mask': mask_tensor,                # (1024, 1024)
            'junction_coords': coords_tensor,   # (4, 2)
            'junction_vis': vis_tensor,         # (4,)
            'filename': stem,
            'is_patient_40': is_patient_40
        }
