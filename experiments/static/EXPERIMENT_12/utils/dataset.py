"""
L3D PyTorch Dataset for EXPERIMENT_12 (Dual-Decoder Foundation Training).
Provides:
  - RGB images (1024x1024, normalized with ImageNet stats)
  - Optional Photometric ColorJitter during training
  - Dense 3-class segmentation masks (0=BG, 1=Ridge, 2=Sil, 3=Falc)
  - Ground-truth 4-junction coordinates in [0, 1]^2
  - Ground-truth 4-junction binary visibility flags
  - Unified Tier-1 Targeted Re-Weighting (all 24 audited deformed/inverted/traction frames)
"""
import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from experiments.EXPERIMENT_12.utils.junction_extractor import extract_gt_junctions

# Standard ImageNet statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Unified Tier-1 Audited Deformed & Retracted Frames (All 24 Frames)
TIER1_DEFORMED_FRAMES = {
    # 8 Severe Inversion / Flipped Flap Frames
    "Patient_12_0266520",
    "Patient_49_84900",
    "Patient_38_0368040",
    "Patient_20_0014880",
    "Patient_53_0133800",
    "Patient_12_0267420",
    "Patient_53_0089640",
    "Patient_12_0265380",

    # 16 High-Traction & Margin Elevation Frames
    "Patient_49_84930",
    "Patient_20_0016200",
    "Patient_20_0016080",
    "Patient_33_0210000",
    "Patient_20_0015600",
    "Patient_12_0265320",
    "Patient_20_0015960",
    "Patient_38_0038880",
    "Patient_12_0266580",
    "Patient_12_0265980",
    "Patient_61_0049680",
    "Patient_22_0140280",
    "Patient_12_0265080",
    "Patient_38_0039240",
    "Patient_8_0018840",
    "Patient_12_0265140",
}


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


class L3DDataset(Dataset):
    def __init__(self, split='Train', data_dir=None, image_size=1024, stroke_width=35, augment=False):
        super().__init__()
        self.split = split
        self.image_size = image_size
        self.stroke_width = stroke_width
        self.augment = augment and (split == 'Train')
        self.root_dir = resolve_l3d_root(data_dir)
        
        split_dir = os.path.join(self.root_dir, split)
        self.img_dir = os.path.join(split_dir, 'images')
        self.lbl_dir = os.path.join(split_dir, 'labels')
        
        self.json_files = sorted(glob.glob(os.path.join(self.lbl_dir, '*.json')))
        if len(self.json_files) == 0:
            raise RuntimeError(f"No JSON annotation files found in: {self.lbl_dir}")
            
        self.stems = [Path(p).stem for p in self.json_files]
        
        if self.augment:
            self.color_jitter = T.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.05
            )
        else:
            self.color_jitter = None
            
        print(f"[{split}] Loaded {len(self.json_files)} frames from: {split_dir} (augment={self.augment})")

    def __len__(self):
        return len(self.json_files)

    def get_sample_weights(self, deformed_weight=15.0, base_weight=1.0):
        weights = []
        n_deformed = 0
        n_base = 0
        
        for stem in self.stems:
            if stem in TIER1_DEFORMED_FRAMES:
                weights.append(float(deformed_weight))
                n_deformed += 1
            else:
                weights.append(float(base_weight))
                n_base += 1
                
        weights_tensor = torch.tensor(weights, dtype=torch.double)
        total_prob_mass = weights_tensor.sum().item()
        
        p_deformed = (n_deformed * deformed_weight) / total_prob_mass * 100.0
        p_base = (n_base * base_weight) / total_prob_mass * 100.0
        
        print("\n⚖️ [WeightedRandomSampler Allocation — Unified Tier 1]")
        print(f"   Tier-1 Deformed/Inverted : {n_deformed:3d} frames | weight={deformed_weight:4.1f}x | {p_deformed:5.2f}% of epoch sampling")
        print(f"   Standard Canonical Frames: {n_base:3d} frames | weight={base_weight:4.1f}x | {p_base:5.2f}% of epoch sampling")
        print(f"   Exposure Frequency       : Approx 1 every {100.0/p_deformed:.1f} samples (>{p_deformed:.1f}% of batches)\n")
        
        return weights_tensor

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        stem = self.stems[idx]
        img_path = os.path.join(self.img_dir, f"{stem}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, f"{stem}.png")
            
        # 1. Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
            
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if orig_w != self.image_size or orig_h != self.image_size:
            img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img_rgb
            
        if self.color_jitter is not None:
            pil_img = Image.fromarray(img_resized)
            pil_img = self.color_jitter(pil_img)
            img_resized = np.array(pil_img)
            
        norm_img = (img_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        pixel_values = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        
        # 2. Parse JSON annotations
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

        if data_w != self.image_size or data_h != self.image_size:
            mask = cv2.resize(canvas_raw, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask = canvas_raw

        mask_tensor = torch.from_numpy(mask).long()
        
        coords_norm, vis, _ = extract_gt_junctions(data, data_w, data_h, canvas_size=self.image_size)
        junction_coords = torch.from_numpy(coords_norm).float()
        junction_vis = torch.from_numpy(vis).float()
        
        is_p40 = ('patient_40' in stem.lower() or '_40_' in stem.lower())
        
        return {
            'pixel_values': pixel_values,
            'mask': mask_tensor,
            'junction_coords': junction_coords,
            'junction_vis': junction_vis,
            'filename': f"{stem}.jpg",
            'is_patient_40': is_p40
        }


def collate_fn_l3d(batch):
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    masks = torch.stack([b['mask'] for b in batch])
    junction_coords = torch.stack([b['junction_coords'] for b in batch])
    junction_vis = torch.stack([b['junction_vis'] for b in batch])
    filenames = [b['filename'] for b in batch]
    is_p40s = [b['is_patient_40'] for b in batch]
    
    mask_labels_list = []
    class_labels_list = []
    
    for b in range(len(batch)):
        m = masks[b]
        unique_classes = torch.unique(m)
        unique_classes = unique_classes[unique_classes > 0]
        
        if len(unique_classes) == 0:
            mask_labels_list.append(torch.zeros((1, m.shape[0], m.shape[1]), dtype=torch.float32))
            class_labels_list.append(torch.zeros(1, dtype=torch.int64))
        else:
            binary_masks = torch.stack([(m == c).float() for c in unique_classes])
            mask_labels_list.append(binary_masks)
            class_labels_list.append(unique_classes.long())
            
    return {
        'pixel_values': pixel_values,
        'mask': masks,
        'mask_labels': mask_labels_list,
        'class_labels': class_labels_list,
        'junction_coords': junction_coords,
        'junction_vis': junction_vis,
        'filename': filenames,
        'is_patient_40': is_p40s
    }
