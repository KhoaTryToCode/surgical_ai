"""
Patient-Wise Temporal L3D Dataset for EXPERIMENT_15.
Extracts chronological clips of T=3 frames strictly within the same patient.
Enforces contiguity threshold (max_gap <= 240 frames ~ 8.0s at 30fps).
Handles missing/disappearing landmarks with empty masks.
Portability: works seamlessly across local macOS, SLURM server, and Kaggle.
"""
import os
import glob
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def find_split_path(split_name, override_dir=None, data_dir=None):
    """
    Robust split resolver handling local, server, and multi-dataset Kaggle directories.
    """
    if override_dir and os.path.exists(override_dir) and os.path.exists(os.path.join(override_dir, 'labels')):
        return os.path.abspath(override_dir)
        
    target = split_name.lower().strip()
    
    # 1. Check inside provided data_dir
    if data_dir and os.path.exists(data_dir):
        for candidate in [split_name, split_name.capitalize(), target]:
            p = os.path.join(data_dir, candidate)
            if os.path.exists(p) and os.path.exists(os.path.join(p, 'labels')):
                return os.path.abspath(p)
                
    # 2. Known local and cluster candidates
    candidates = [
        f"/kaggle/input/datasets/khoatrytopublish/l3d-{target}/{split_name}",
        f"/kaggle/input/datasets/khoatrytopublish/l3d_{target}/{split_name}",
        f"/kaggle/input/datasets/khoatrytopublish/l3d{target}/{split_name}",
        f"/kaggle/input/l3d-{target}/{split_name}",
        f"/kaggle/input/l3d_{target}/{split_name}",
        f"/kaggle/input/l3d - {split_name}/{split_name}",
        f"/kaggle/input/l3d - {target}/{split_name}",
        f"/kaggle/input/l3d-{target}",
        f"/kaggle/input/l3d/{split_name}",
        f"/kaggle/input/laparoscopic-liver-landmark-dataset/L3D/{split_name}",
        f"/kaggle/input/l3d-dataset/L3D/{split_name}",
        f"/data/khoalq/data/L3D/{split_name}",
        f"data/L3D/{split_name}",
        f"../data/L3D/{split_name}",
        f"../../data/L3D/{split_name}",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.exists(os.path.join(c, 'labels')):
            return os.path.abspath(c)
            
    # 3. Recursive search under /kaggle/input
    if os.path.exists("/kaggle/input"):
        for m in glob.glob(f"/kaggle/input/**/{split_name}", recursive=True):
            if os.path.isdir(m) and os.path.exists(os.path.join(m, 'labels')):
                return os.path.abspath(m)
        for m in glob.glob(f"/kaggle/input/**/{target}", recursive=True):
            if os.path.isdir(m) and os.path.exists(os.path.join(m, 'labels')):
                return os.path.abspath(m)
        for p in glob.glob("/kaggle/input/**/labels", recursive=True):
            parent = os.path.dirname(p)
            if target in parent.lower():
                return os.path.abspath(parent)
                
    raise RuntimeError(f"Could not locate {split_name} split directory! Check your dataset inputs.")

class PatientTemporalL3DDataset(Dataset):
    """
    Temporal Dataset that extracts sliding clips of T=3 frames strictly within the same patient.
    """
    def __init__(self, split='Train', data_dir=None, split_dir=None, clip_len=3, max_gap=240, image_size=1024, stroke_width=35):
        super().__init__()
        self.split = split
        self.clip_len = clip_len
        self.max_gap = max_gap
        self.image_size = image_size
        self.stroke_width = stroke_width
        
        self.split_dir = find_split_path(split, override_dir=split_dir, data_dir=data_dir)
        self.clips = self._index_temporal_clips()
        print(f"[{split} - Temporal T={clip_len}] Loaded {len(self.clips)} clips from {self.split_dir}")

    def _index_temporal_clips(self):
        json_files = sorted(glob.glob(os.path.join(self.split_dir, 'labels', '*.json')))
        patient_dict = defaultdict(list)
        
        for jf in json_files:
            stem = Path(jf).stem
            m = re.match(r'(Patient_\d+)_(\d+)', stem)
            if m:
                pat = m.group(1)
                fid = int(m.group(2))
                
                # Check corresponding image path
                img_path = os.path.join(self.split_dir, 'images', f"{stem}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.split_dir, 'images', f"{stem}.png")
                    
                if os.path.exists(img_path):
                    patient_dict[pat].append((fid, jf, img_path))
                    
        clips = []
        for pat, frames in sorted(patient_dict.items()):
            frames.sort(key=lambda x: x[0])
            
            # Segment into contiguous sequences where gap <= max_gap
            segments = []
            cur = [frames[0]]
            for i in range(1, len(frames)):
                gap = frames[i][0] - frames[i-1][0]
                if gap <= self.max_gap:
                    cur.append(frames[i])
                else:
                    if len(cur) >= self.clip_len:
                        segments.append(cur)
                    cur = [frames[i]]
            if len(cur) >= self.clip_len:
                segments.append(cur)
                
            # Extract sliding windows of length clip_len
            for seg in segments:
                for s in range(len(seg) - self.clip_len + 1):
                    window = seg[s : s + self.clip_len]
                    clips.append({
                        'patient': pat,
                        'frame_ids': [w[0] for w in window],
                        'json_paths': [w[1] for w in window],
                        'img_paths': [w[2] for w in window]
                    })
        return clips

    def __len__(self):
        return len(self.clips)

    def _load_single_frame(self, img_path, json_path):
        # 1. Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if orig_w != self.image_size or orig_h != self.image_size:
            img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img_rgb
            
        norm_img = (img_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        pixel_tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float() # (3, H, W)
        
        # 2. Parse JSON & draw mask
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        data_w = data.get('imageWidth', orig_w)
        data_h = data.get('imageHeight', orig_h)
        canvas_raw = np.zeros((data_h, data_w), dtype=np.uint8)
        
        shapes = data.get('shapes', [])
        for shape in shapes:
            lbl = str(shape.get('label', '')).lower().strip()
            pts = shape.get('points', [])
            if len(pts) < 2:
                continue
                
            # Class mapping (with tolerance for 'rigde' typo in L3D annotations)
            if lbl.startswith('r') or 'ridge' in lbl or 'rigde' in lbl or 'anterior' in lbl:
                color = 1 # Ridge
            elif lbl.startswith('s') or 'sil' in lbl or 'margin' in lbl or 'silhouette' in lbl:
                color = 2 # Silhouette
            elif lbl.startswith('f') or lbl.startswith('l') or 'falc' in lbl or 'lig' in lbl:
                color = 3 # Falciform
            else:
                color = 0
                
            if color > 0:
                for i in range(1, len(pts)):
                    pt1 = tuple(map(int, pts[i - 1]))
                    pt2 = tuple(map(int, pts[i]))
                    cv2.line(canvas_raw, pt1, pt2, color, self.stroke_width)
                    
        if data_w != self.image_size or data_h != self.image_size:
            mask = cv2.resize(canvas_raw, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask = canvas_raw
            
        mask_tensor = torch.from_numpy(mask).long() # (H, W)
        return pixel_tensor, mask_tensor

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        
        frames_pixels = []
        frames_masks = []
        
        for img_p, json_p in zip(clip_info['img_paths'], clip_info['json_paths']):
            p_tensor, m_tensor = self._load_single_frame(img_p, json_p)
            frames_pixels.append(p_tensor)
            frames_masks.append(m_tensor)
            
        # Stack across time: pixel_values shape (T, 3, H, W), masks shape (T, H, W)
        pixel_values = torch.stack(frames_pixels, dim=0)
        masks = torch.stack(frames_masks, dim=0)
        
        meta = {
            'patient': clip_info['patient'],
            'frame_ids': clip_info['frame_ids'],
            'img_paths': clip_info['img_paths']
        }
        return pixel_values, masks, meta
