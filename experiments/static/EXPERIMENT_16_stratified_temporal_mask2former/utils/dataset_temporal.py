import os
import glob
import re
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Landmark class mapping
CLASS_MAP = {
    'liver_ridge': 1,
    'ridge': 1,
    'rigde': 1,  # Official annotator typo in L3D benchmark (1,661 shapes)
    'liver_silhouette': 2,
    'silhouette': 2,
    'falciform_ligament': 3,
    'falciform': 3,
    'ligament': 3  # Official annotator string in L3D benchmark (1,101 shapes)
}

# Re-stratification specification (Option 1: Patient_40 <-> Patient_38 Swap from EXP_14)
# Patient_40 (101 frames) is moved to Train to provide traction / grasper elevation training.
# Patient_38 (99 frames) is moved to Val.
# Patient_32 (15 frames) and Patient_18 (6 frames) remain in Val.
# Test set remains 100% frozen (Patients 21, 31, 41, 51; 109 frames).
VAL_PATIENTS = {'patient_38', 'patient_32', 'patient_18'}


def resolve_split_dir(split_name, custom_path=None):
    if custom_path and os.path.exists(custom_path):
        return os.path.abspath(custom_path)

    split_lower = split_name.lower()
    candidates = {
        'train': [
            '/kaggle/input/l3d-train/Train',
            '/kaggle/input/l3d-train/train',
            '/kaggle/input/datasets/khoatrytopublish/l3d-train/Train',
            '/kaggle/input/datasets/khoatrytopublish/l3d-train/train',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/train',
            '/kaggle/input/l3d-dataset/L3D/train',
            'data/L3D/train',
            '../data/L3D/train',
            '../../data/L3D/train',
            '/Users/khoale/Downloads/Surgical AI/data/L3D/train'
        ],
        'val': [
            '/kaggle/input/l3d-val/Val',
            '/kaggle/input/l3d-val/val',
            '/kaggle/input/datasets/khoatrytopublish/l3d-val/Val',
            '/kaggle/input/datasets/khoatrytopublish/l3d-val/val',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/val',
            '/kaggle/input/l3d-dataset/L3D/val',
            'data/L3D/val',
            '../data/L3D/val',
            '../../data/L3D/val',
            '/Users/khoale/Downloads/Surgical AI/data/L3D/val'
        ],
        'test': [
            '/kaggle/input/l3d-test/Test',
            '/kaggle/input/l3d-test/test',
            '/kaggle/input/datasets/khoatrytopublish/l3d-test/Test',
            '/kaggle/input/datasets/khoatrytopublish/l3d-test/test',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/test',
            '/kaggle/input/l3d-dataset/L3D/test',
            'data/L3D/test',
            '../data/L3D/test',
            '../../data/L3D/test',
            '/Users/khoale/Downloads/Surgical AI/data/L3D/test'
        ]
    }

    for path in candidates.get(split_lower, []):
        if os.path.exists(path):
            return os.path.abspath(path)

    if os.path.exists('/kaggle/input'):
        for root, dirs, _ in os.walk('/kaggle/input', followlinks=True):
            bname = os.path.basename(root).lower()
            if bname == split_lower or (split_lower in bname):
                if any(d.lower() == 'labels' for d in dirs):
                    return os.path.abspath(root)

    raise FileNotFoundError(f"Could not resolve directory for split '{split_name}' in Kaggle or local environments.")


def resolve_l3d_root():
    candidates = [
        "/kaggle/input/laparoscopic-liver-dataset/L3D",
        "/kaggle/input/l3d-dataset/L3D",
        "data/L3D",
        "../data/L3D",
        "../../data/L3D",
        "/Users/khoale/Downloads/Surgical AI/data/L3D"
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return "data/L3D"


class StratifiedTemporalL3DDataset(Dataset):
    """
    Stratified Temporal Laparoscopic Liver Landmark Dataset for EXPERIMENT_16.
    
    Inherits 100% of the Temporal Mask2Former architecture, TopoNet native canvas
    rasterization, and 100% causal temporal context padding from EXPERIMENT_15.
    
    Rebalances Train and Val anatomical pose distribution using EXPERIMENT_14 stratification:
      - Train: 923 frames across 32 patients (includes Patient_40; excludes Patient_38).
      - Val:   120 frames across 3 patients (Patient_38: 99, Patient_32: 15, Patient_18: 6).
      - Test:  109 frames across 4 patients (Patients 21, 31, 41, 51; 100% frozen benchmark).
    """
    def __init__(self, split='train', train_dir=None, val_dir=None, test_dir=None, clip_len=3, max_gap=240, image_size=1024, stroke_width=35):
        if os.path.exists(split) or '/' in split or '\\' in split:
            sp_lower = split.lower()
            if 'test' in sp_lower:
                self.split = 'test'
                test_dir = test_dir or split
            elif 'val' in sp_lower:
                self.split = 'val'
                val_dir = val_dir or split
            else:
                self.split = 'train'
                train_dir = train_dir or split
        else:
            self.split = split.lower()

        self.clip_len = clip_len
        self.max_gap = max_gap
        self.image_size = image_size
        self.stroke_width = stroke_width

        self.train_dir = resolve_split_dir('train', train_dir) if self.split in ['train', 'val'] else None
        self.val_dir = resolve_split_dir('val', val_dir) if self.split in ['train', 'val'] else None
        self.test_dir = resolve_split_dir('test', test_dir) if self.split == 'test' else None

        self.clips = self._index_temporal_clips()
        print(f"[{self.split.upper()} - Stratified Option 1] Loaded {len(self.clips)} causal clips (100% coverage).")

    def _index_temporal_clips(self):
        patient_dict = defaultdict(list)

        if self.split == 'test':
            # Frozen official test benchmark (Patients 21, 31, 41, 51)
            labels_dir = os.path.join(self.test_dir, 'labels')
            images_dir = os.path.join(self.test_dir, 'images')
            json_files = sorted(glob.glob(os.path.join(labels_dir, '*.json')))

            for jf in json_files:
                stem = Path(jf).stem
                m = re.match(r'(Patient_\d+)_(\d+)', stem)
                if not m:
                    continue
                pat, fid = m.group(1), int(m.group(2))
                img_path = os.path.join(images_dir, f"{stem}.png")
                if not os.path.exists(img_path):
                    img_path = os.path.join(images_dir, f"{stem}.jpg")
                if os.path.exists(img_path):
                    patient_dict[pat].append((fid, jf, img_path))
        else:
            # Pool all Train and Val files from disk without altering original filesystem
            all_files = []
            for s_dir in [self.train_dir, self.val_dir]:
                labels_dir = os.path.join(s_dir, 'labels')
                images_dir = os.path.join(s_dir, 'images')
                for jf in glob.glob(os.path.join(labels_dir, '*.json')):
                    stem = Path(jf).stem
                    img_path = os.path.join(images_dir, f"{stem}.png")
                    if not os.path.exists(img_path):
                        img_path = os.path.join(images_dir, f"{stem}.jpg")
                    if os.path.exists(img_path):
                        all_files.append((jf, img_path))

            for jf, img_path in sorted(all_files):
                stem = Path(jf).stem
                m = re.match(r'(Patient_\d+)_(\d+)', stem)
                if not m:
                    continue
                pat, fid = m.group(1), int(m.group(2))
                is_val_patient = any(vp in pat.lower() for vp in VAL_PATIENTS)

                if self.split == 'val' and is_val_patient:
                    patient_dict[pat].append((fid, jf, img_path))
                elif self.split == 'train' and not is_val_patient:
                    patient_dict[pat].append((fid, jf, img_path))

        # Build temporal clips with strictly causal context padding
        clips = []
        for pat in sorted(patient_dict.keys()):
            frames = patient_dict[pat]
            frames.sort(key=lambda x: x[0])  # Sort chronologically

            for i in range(len(frames)):
                target = frames[i]

                # Strictly causal padding: only look backward in time
                if i >= 1 and (frames[i][0] - frames[i - 1][0]) <= self.max_gap:
                    c1 = frames[i - 1]
                    if i >= 2 and (frames[i - 1][0] - frames[i - 2][0]) <= self.max_gap:
                        c0 = frames[i - 2]
                    else:
                        c0 = c1
                    window = [c0, c1, target]
                else:
                    window = [target, target, target]

                clips.append({
                    'patient': pat,
                    'frame_ids': [w[0] for w in window],
                    'json_paths': [w[1] for w in window],
                    'img_paths': [w[2] for w in window],
                    'target_idx': self.clip_len - 1
                })

        return clips

    def __len__(self):
        return len(self.clips)

    def _load_single_frame(self, img_path, json_path):
        # 1. Load Image
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
        pixel_tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float()  # (3, H, W)

        # 2. Parse JSON & Rasterize Mask using TopoNet's exact two-step method
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        canvas_h = data.get('imageHeight', orig_h)
        canvas_w = data.get('imageWidth', orig_w)
        if any(x in json_path for x in ['_31', '_36', '_25', '_29']):
            canvas_h, canvas_w = 2160, 3840
        elif canvas_h < 100 or canvas_w < 100:
            canvas_h, canvas_w = 1080, 1920

        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        shapes = data.get('shapes', [])
        for shape in shapes:
            lbl = str(shape.get('label', '')).lower().strip()
            if lbl.startswith('r'):
                color = 1
            elif lbl.startswith('s'):
                color = 2
            elif lbl.startswith('l') or 'falciform' in lbl:
                color = 3
            else:
                color = 0

            if color > 0:
                pts = shape.get('points', [])
                for i in range(1, len(pts)):
                    pt1 = tuple(map(int, pts[i - 1]))
                    pt2 = tuple(map(int, pts[i]))
                    cv2.line(canvas, pt1, pt2, color, int(self.stroke_width))

        # TopoNet INTER_NEAREST downsampling to 1024x1024 (effective stroke ~18.7px)
        gt_map = cv2.resize(canvas, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Mask2Former binary mask channels: 0: Ridge, 1: Silhouette, 2: Ligament
        mask_channels = np.stack([gt_map == 1, gt_map == 2, gt_map == 3], axis=0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_channels)  # (3, H, W)
        gt_map_tensor = torch.from_numpy(gt_map.astype(np.int64))  # (H, W)

        return pixel_tensor, mask_tensor, gt_map_tensor

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        pixel_tensors = []
        mask_tensors = []
        gt_map_tensors = []

        for img_p, json_p in zip(clip_info['img_paths'], clip_info['json_paths']):
            pt, mt, gm = self._load_single_frame(img_p, json_p)
            pixel_tensors.append(pt)
            mask_tensors.append(mt)
            gt_map_tensors.append(gm)

        pixel_values = torch.stack(pixel_tensors, dim=0)   # (T, 3, H, W)
        mask_labels = torch.stack(mask_tensors, dim=0)     # (T, 3, H, W)
        target_masks = mask_labels[-1]                     # (3, H, W)
        target_gt_map = gt_map_tensors[-1]                 # (H, W)

        return {
            'pixel_values': pixel_values,
            'mask_labels': mask_labels,
            'target_masks': target_masks,
            'gt_map': target_gt_map,
            'patient': clip_info['patient'],
            'frame_ids': clip_info['frame_ids']
        }

# Alias for 100% backward compatibility
PatientTemporalL3DDataset = StratifiedTemporalL3DDataset
