"""
CholecSeg8k PyTorch Dataset for EXPERIMENT_12.
Provides:
  - 8,080 frames across 17 laparoscopic cholecystectomy videos
  - Dense 13-class semantic masks from *_endo_watershed_mask.png
  - Ultra-fast 256-element vectorized LUT remapping
  - Formatted mask_labels & class_labels for HuggingFace Mask2Former Hungarian criterion
"""
import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Canonical 13 semantic classes in CholecSeg8k
CHOLEC_CLASS_NAMES = {
    0: 'Black Background',
    1: 'Abdominal Wall',
    2: 'Liver',
    3: 'Gastrointestinal Tract',
    4: 'Fat',
    5: 'Grasper',
    6: 'Connective Tissue',
    7: 'Blood',
    8: 'Cystic Duct',
    9: 'L-hook Electrocautery',
    10: 'Gallbladder',
    11: 'Hepatic Vein',
    12: 'Liver Ligament'
}

# Build 256-element vector lookup table
# Unmapped values default to 255 (ignore index)
_LUT = np.full(256, 255, dtype=np.uint8)
_LUT[50]  = 0   # Black Background
_LUT[11]  = 1   # Abdominal Wall
_LUT[21]  = 2   # Liver
_LUT[13]  = 3   # Gastrointestinal Tract
_LUT[12]  = 4   # Fat
_LUT[31]  = 5   # Grasper
_LUT[23]  = 6   # Connective Tissue
_LUT[24]  = 7   # Blood
_LUT[25]  = 8   # Cystic Duct
_LUT[32]  = 9   # L-hook Electrocautery
_LUT[22]  = 10  # Gallbladder
_LUT[33]  = 11  # Hepatic Vein
_LUT[5]   = 12  # Liver Ligament
_LUT[255] = 255 # Watershed Boundary (Ignore)


def resolve_cholec_root(candidate_root=None):
    """
    Auto-detects the CholecSeg8k dataset root path across local macOS, Linux, and Kaggle.
    """
    candidates = [
        candidate_root,
        "data/cholecseg8k",
        "../data/cholecseg8k",
        "../../data/cholecseg8k",
        "/data/khoalq/data/cholecseg8k",
        "/kaggle/input/cholecseg8k",
        "/kaggle/input/newslab-kaist/cholecseg8k"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath("data/cholecseg8k")


class CholecSeg8kDataset(Dataset):
    def __init__(self, data_dir=None, image_size=1024, max_samples=None):
        super().__init__()
        self.image_size = image_size
        self.root_dir = resolve_cholec_root(data_dir)
        
        # Discover all endo frame paths
        self.frame_paths = sorted(glob.glob(os.path.join(self.root_dir, '*/*/*_endo.png')))
        if len(self.frame_paths) == 0:
            raise RuntimeError(f"No CholecSeg8k frames found in: {self.root_dir}")
            
        if max_samples is not None and max_samples < len(self.frame_paths):
            self.frame_paths = self.frame_paths[:max_samples]
            
        print(f"[CholecSeg8k] Loaded {len(self.frame_paths)} frames from: {self.root_dir}")

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        mask_path = frame_path.replace('_endo.png', '_endo_watershed_mask.png')
        
        # 1. Load image
        img_bgr = cv2.imread(frame_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at: {frame_path}")
            
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if orig_w != self.image_size or orig_h != self.image_size:
            img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img_rgb
            
        norm_img = (img_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        pixel_values = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        
        # 2. Load dense watershed mask
        ws_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if ws_img is None:
            raise FileNotFoundError(f"Failed to read watershed mask at: {mask_path}")
            
        # Fast vectorized class remapping via LUT
        mapped_mask = _LUT[ws_img]
        
        if orig_w != self.image_size or orig_h != self.image_size:
            mask_resized = cv2.resize(mapped_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mapped_mask
            
        mask_tensor = torch.from_numpy(mask_resized).long()
        
        return {
            'pixel_values': pixel_values,
            'mask': mask_tensor,
            'filename': os.path.basename(frame_path)
        }


def collate_fn_cholec(batch):
    """
    Collate function preparing Hungarian matching inputs for CholecSeg8k.
    Extracts individual binary masks for each foreground class in [1, 12].
    """
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    masks = torch.stack([b['mask'] for b in batch])
    filenames = [b['filename'] for b in batch]
    
    mask_labels_list = []
    class_labels_list = []
    
    for b in range(len(batch)):
        m = masks[b]
        # Valid foreground classes: > 0 and != 255 (ignore boundary)
        unique_classes = torch.unique(m)
        fg_classes = unique_classes[(unique_classes > 0) & (unique_classes != 255)]
        
        if len(fg_classes) == 0:
            mask_labels_list.append(torch.zeros((1, m.shape[0], m.shape[1]), dtype=torch.float32))
            class_labels_list.append(torch.zeros(1, dtype=torch.int64))
        else:
            binary_masks = torch.stack([(m == c).float() for c in fg_classes])
            mask_labels_list.append(binary_masks)
            class_labels_list.append(fg_classes.long())
            
    return {
        'pixel_values': pixel_values,
        'mask': masks,
        'mask_labels': mask_labels_list,
        'class_labels': class_labels_list,
        'filename': filenames
    }
