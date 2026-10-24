"""
L3D Dataset Loader for MinVIS (Experiment 17).
Strictly matches EXP 15 canvas rasterization (35px stroke width, INTER_NEAREST downsampling,
ImageNet normalization) and integrates with the canonical 5-fold patient-disjoint splits.
"""

import os
import glob
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Strict label mapping covering exact L3D annotations (including known 'rigde' typo)
CLASS_MAP = {
    'ridge': 1,
    'rigde': 1,
    'liver_ridge': 1,
    'silhouette': 2,
    'liver_silhouette': 2,
    'ligament': 3,
    'falciform': 3,
    'falciform_ligament': 3,
}


def resolve_split_dir(split_name: str, candidate_root: Optional[str] = None) -> str:
    """
    Resolves directory for a dataset split ('train', 'val', 'test') across
    Kaggle multi-dataset inputs (e.g. l3d-train/Train), unified datasets, and local directories.
    Strictly mirrors EXP 15 candidate resolution and case-insensitive discovery.
    """
    split_lower = split_name.lower()

    if candidate_root and os.path.exists(candidate_root):
        if any(os.path.exists(os.path.join(candidate_root, d)) for d in ['labels', 'Labels', 'LABELS']):
            return candidate_root
        try:
            for sub in os.listdir(candidate_root):
                if sub.lower() == split_lower:
                    full_p = os.path.join(candidate_root, sub)
                    if os.path.isdir(full_p) and any(os.path.exists(os.path.join(full_p, d)) for d in ['labels', 'Labels', 'LABELS']):
                        return full_p
        except OSError:
            pass

    # Specific candidate paths across Kaggle multi-dataset inputs and local environments
    candidates = {
        'train': [
            '/kaggle/input/l3d-train/Train',
            '/kaggle/input/l3d-train/train',
            '/kaggle/input/datasets/khoatrytopublish/l3d-train/Train',
            '/kaggle/input/datasets/khoatrytopublish/l3d-train/train',
            '/kaggle/input/laparoscopic-liver-landmarks/train',
            '/kaggle/input/laparoscopic-liver-landmarks/Train',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/train',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/Train',
            '/kaggle/input/l3d-dataset/L3D/train',
            '/kaggle/input/l3d-dataset/L3D/Train',
            'data/L3D/Train',
            'data/L3D/train',
            '../data/L3D/Train',
            '../data/L3D/train',
            '../../data/L3D/Train',
            '../../data/L3D/train',
            '/Users/khoale/Downloads/Surgical AI/data/L3D/Train',
            '/Users/khoale/Downloads/Surgical AI/data/L3D/train'
        ],
        'val': [
            '/kaggle/input/l3d-val/Val',
            '/kaggle/input/l3d-val/val',
            '/kaggle/input/datasets/khoatrytopublish/l3d-val/Val',
            '/kaggle/input/datasets/khoatrytopublish/l3d-val/val',
            '/kaggle/input/laparoscopic-liver-landmarks/val',
            '/kaggle/input/laparoscopic-liver-landmarks/Val',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/val',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/Val',
            '/kaggle/input/l3d-dataset/L3D/val',
            '/kaggle/input/l3d-dataset/L3D/Val',
            'data/L3D/Val',
            'data/L3D/val',
            '../data/L3D/Val',
            '../data/L3D/val',
            '../../data/L3D/Val',
            '../../data/L3D/val',
            '/Users/khoale/Downloads/Surgical AI/data/L3D/Val',
            '/Users/khoale/Downloads/Surgical AI/data/L3D/val'
        ],
        'test': [
            '/kaggle/input/l3d-test/Test',
            '/kaggle/input/l3d-test/test',
            '/kaggle/input/datasets/khoatrytopublish/l3d-test/Test',
            '/kaggle/input/datasets/khoatrytopublish/l3d-test/test',
            '/kaggle/input/laparoscopic-liver-landmarks/test',
            '/kaggle/input/laparoscopic-liver-landmarks/Test',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/test',
            '/kaggle/input/laparoscopic-liver-dataset/L3D/Test',
            '/kaggle/input/l3d-dataset/L3D/test',
            '/kaggle/input/l3d-dataset/L3D/Test',
            'data/L3D/Test',
            'data/L3D/test',
            '../data/L3D/Test',
            '../data/L3D/test',
            '../../data/L3D/Test',
            '../../data/L3D/test',
            '/Users/khoale/Downloads/Surgical AI/data/L3D/Test',
            '/Users/khoale/Downloads/Surgical AI/data/L3D/test'
        ]
    }

    for path in candidates.get(split_lower, []):
        if os.path.exists(path):
            return path

    # Dynamic search in /kaggle/input
    if os.path.exists('/kaggle/input'):
        for root, dirs, _ in os.walk('/kaggle/input', followlinks=True):
            bname = os.path.basename(root).lower()
            if bname == split_lower or (split_lower in bname):
                if any(d.lower() == 'labels' for d in dirs):
                    return root

    raise FileNotFoundError(f"Could not resolve directory for split '{split_name}' across Kaggle or local environments.")


def resolve_all_split_dirs(candidate_root: Optional[str] = None) -> Dict[str, str]:
    """
    Resolves train, val, and test split directories across Kaggle multi-dataset inputs,
    unified local datasets, or relative folders.
    Returns:
        dict: mapping from split name ('train', 'val', 'test') to resolved absolute/relative directory.
    """
    splits = {}
    for s in ['train', 'val', 'test']:
        try:
            splits[s] = resolve_split_dir(s, candidate_root=candidate_root)
        except FileNotFoundError:
            pass
    return splits


def rasterize_l3d_mask(
    json_path: str,
    orig_shape: Tuple[int, int],
    target_size: int = 1024,
    stroke_width: int = 35
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact two-step rasterization matching EXP 15 and TopoNet benchmark:
    1. Render 35px stroke on native canvas resolution (1080x1920 or 2160x3840).
    2. Downsample to target_size (1024x1024) using cv2.INTER_NEAREST.
    
    Returns:
      mask_channels: (3, target_size, target_size) float32 binary masks for Ridge, Silh, Lig.
      gt_map: (target_size, target_size) int64 discrete class map in {0, 1, 2, 3}.
    """
    orig_h, orig_w = orig_shape
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
            color = 1  # Ridge
        elif lbl.startswith('s'):
            color = 2  # Silhouette
        elif lbl.startswith('l') or 'falciform' in lbl:
            color = 3  # Ligament
        else:
            color = 0

        if color > 0:
            pts = shape.get('points', [])
            for i in range(1, len(pts)):
                pt1 = tuple(map(int, pts[i - 1]))
                pt2 = tuple(map(int, pts[i]))
                cv2.line(canvas, pt1, pt2, color, int(stroke_width))

    gt_map = cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    mask_channels = np.stack([gt_map == 1, gt_map == 2, gt_map == 3], axis=0).astype(np.float32)
    return mask_channels, gt_map.astype(np.int64)


class SingleFrameL3DDataset(Dataset):
    """
    Single-frame L3D Dataset for standard image-level training in MinVIS.
    Filters frames by a specific list of patient IDs across resolved split directories.
    """
    def __init__(
        self,
        patient_ids: List[str],
        data_root: Optional[str] = "data/L3D",
        split_dirs: Optional[List[str]] = None,
        image_size: int = 1024,
        stroke_width: int = 35
    ):
        self.patient_ids = set(patient_ids)
        self.image_size = image_size
        self.stroke_width = stroke_width

        if split_dirs is not None:
            self.split_dirs = [s for s in split_dirs if os.path.exists(s)]
        else:
            split_map = resolve_all_split_dirs(data_root)
            self.split_dirs = list(split_map.values())

        self.samples = self._discover_samples(self.split_dirs)

    def _discover_samples(self, split_dirs: List[str]) -> List[Tuple[str, int, str, str]]:
        samples = []
        for s_dir in split_dirs:
            if not os.path.exists(s_dir):
                continue

            labels_dir = None
            for cand in ['labels', 'Labels', 'LABELS']:
                p = os.path.join(s_dir, cand)
                if os.path.exists(p):
                    labels_dir = p
                    break
            if labels_dir is None:
                continue

            images_dir = None
            for cand in ['images', 'Images', 'IMAGES']:
                p = os.path.join(s_dir, cand)
                if os.path.exists(p):
                    images_dir = p
                    break
            if images_dir is None:
                continue

            for jf in sorted(glob.glob(os.path.join(labels_dir, '*.json'))):
                stem = Path(jf).stem
                m = re.match(r'(Patient_\d+)_(\d+)', stem)
                if not m:
                    continue
                pat, fid = m.group(1), int(m.group(2))
                if pat not in self.patient_ids:
                    continue

                img_path = os.path.join(images_dir, f"{stem}.png")
                if not os.path.exists(img_path):
                    img_path = os.path.join(images_dir, f"{stem}.jpg")
                if os.path.exists(img_path):
                    samples.append((pat, fid, img_path, jf))

        samples.sort(key=lambda x: (x[0], x[1]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pat, fid, img_path, json_path = self.samples[idx]

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if orig_w != self.image_size or orig_h != self.image_size:
            img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img_rgb

        mask_channels, gt_map = rasterize_l3d_mask(
            json_path, (orig_h, orig_w), target_size=self.image_size, stroke_width=self.stroke_width
        )

        norm_img = (img_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        pixel_tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float()  # (3, H, W)
        mask_tensor = torch.from_numpy(mask_channels).float()                # (3, H, W)
        gt_map_tensor = torch.from_numpy(gt_map).long()                     # (H, W)

        return {
            'pixel_values': pixel_tensor,
            'target_masks': mask_tensor,
            'gt_map': gt_map_tensor,
            'patient': pat,
            'frame_id': fid,
            'img_path': img_path
        }


class PatientVideoL3DDataset(Dataset):
    """
    Patient-grouped Video Clip Dataset for MinVIS sequential evaluation.
    Provides chronological video sequences per patient for online Hungarian query tracking.
    """
    def __init__(
        self,
        patient_ids: List[str],
        data_root: Optional[str] = "data/L3D",
        split_dirs: Optional[List[str]] = None,
        image_size: int = 1024,
        stroke_width: int = 35,
        clip_len: int = 3,
        max_gap: int = 240
    ):
        self.patient_ids = set(patient_ids)
        self.image_size = image_size
        self.stroke_width = stroke_width
        self.clip_len = clip_len
        self.max_gap = max_gap

        if split_dirs is not None:
            self.split_dirs = [s for s in split_dirs if os.path.exists(s)]
        else:
            split_map = resolve_all_split_dirs(data_root)
            self.split_dirs = list(split_map.values())

        self.clips = self._index_clips(self.split_dirs)

    def _index_clips(self, split_dirs: List[str]) -> List[Dict]:
        patient_dict = defaultdict(list)

        for s_dir in split_dirs:
            if not os.path.exists(s_dir):
                continue

            labels_dir = None
            for cand in ['labels', 'Labels', 'LABELS']:
                p = os.path.join(s_dir, cand)
                if os.path.exists(p):
                    labels_dir = p
                    break
            if labels_dir is None:
                continue

            images_dir = None
            for cand in ['images', 'Images', 'IMAGES']:
                p = os.path.join(s_dir, cand)
                if os.path.exists(p):
                    images_dir = p
                    break
            if images_dir is None:
                continue

            for jf in sorted(glob.glob(os.path.join(labels_dir, '*.json'))):
                stem = Path(jf).stem
                m = re.match(r'(Patient_\d+)_(\d+)', stem)
                if not m:
                    continue
                pat, fid = m.group(1), int(m.group(2))
                if pat not in self.patient_ids:
                    continue

                img_path = os.path.join(images_dir, f"{stem}.png")
                if not os.path.exists(img_path):
                    img_path = os.path.join(images_dir, f"{stem}.jpg")
                if os.path.exists(img_path):
                    patient_dict[pat].append((fid, jf, img_path))

        clips = []
        for pat in sorted(patient_dict.keys()):
            frames = patient_dict[pat]
            frames.sort(key=lambda x: x[0])

            for i in range(len(frames)):
                target = frames[i]
                if i >= 1 and (frames[i][0] - frames[i-1][0]) <= self.max_gap:
                    c1 = frames[i-1]
                    c0 = frames[i-2] if (i >= 2 and (frames[i-1][0] - frames[i-2][0]) <= self.max_gap) else c1
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

    def __len__(self) -> int:
        return len(self.clips)

    def _load_frame(self, img_path: str, json_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if orig_w != self.image_size or orig_h != self.image_size:
            img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img_rgb

        mask_channels, gt_map = rasterize_l3d_mask(
            json_path, (orig_h, orig_w), target_size=self.image_size, stroke_width=self.stroke_width
        )

        norm_img = (img_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        pixel_tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_channels).float()
        gt_map_tensor = torch.from_numpy(gt_map).long()

        return pixel_tensor, mask_tensor, gt_map_tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clip_info = self.clips[idx]
        pixel_tensors, mask_tensors, gt_map_tensors = [], [], []

        for img_p, json_p in zip(clip_info['img_paths'], clip_info['json_paths']):
            pt, mt, gm = self._load_frame(img_p, json_p)
            pixel_tensors.append(pt)
            mask_tensors.append(mt)
            gt_map_tensors.append(gm)

        pixel_values = torch.stack(pixel_tensors, dim=0)  # (T, 3, H, W)
        mask_labels = torch.stack(mask_tensors, dim=0)    # (T, 3, H, W)

        return {
            'pixel_values': pixel_values,
            'mask_labels': mask_labels,
            'target_masks': mask_labels[-1],               # (3, H, W)
            'gt_map': gt_map_tensors[-1],                  # (H, W)
            'patient': clip_info['patient'],
            'frame_ids': clip_info['frame_ids']
        }
