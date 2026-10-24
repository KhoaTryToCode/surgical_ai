import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class Mask2FormerDataset(Dataset):
    """
    Standardized L3D Dataset Reader for Pure Mask2Former (RGB-Only, NO Depth).
    Preserves dynamic canvas resolution to prevent Patient 32 4K coordinate clipping.
    """
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        
        # Resolve images directory flexibly (supports both nested 'images/' and flat folders)
        if os.path.exists(os.path.join(data_dir, 'images')):
            self.image_paths = sorted(glob.glob(os.path.join(data_dir, 'images', '*.[jJ][pP][gG]')) + 
                                      glob.glob(os.path.join(data_dir, 'images', '*.[pP][nN][gG]')))
        else:
            self.image_paths = sorted(glob.glob(os.path.join(data_dir, '*.[jJ][pP][gG]')) + 
                                      glob.glob(os.path.join(data_dir, '*.[pP][nN][gG]')))
            
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in dataset directory: {data_dir}")

        self.transform = transform if transform else self._default_transform

    @staticmethod
    def _default_transform(image, mask):
        to_tensor = T.ToTensor()
        return to_tensor(image), to_tensor(mask)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self.load_image(img_path)
        mask = self.load_mask(img_path)
        fname = os.path.basename(img_path)

        # Transform expects mask in shape (H, W, C)
        image_t, mask_t = self.transform(image, mask.transpose(1, 2, 0))

        return image_t, mask_t, fname

    @staticmethod
    def load_image(path):
        """Loads RGB image, resizes to (1024, 1024), returns uint8 in RGB format."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        return img

    @classmethod
    def load_mask(cls, path):
        """
        Dynamically reads JSON labels with Patient 32 4K canvas fix.
        Returns one-hot mask of shape (4, 1024, 1024).
        Channels: 0: Background, 1: Anterior Ridge, 2: Silhouette, 3: Falciform.
        """
        path = str(path)
        # Search for corresponding JSON file
        json_path = os.path.splitext(path)[0] + '.json'
        
        # If not found directly, search in sibling 'labels/' or 'label/' directory
        if not os.path.exists(json_path):
            parent_dir = os.path.dirname(path)
            base_fname = os.path.splitext(os.path.basename(path))[0]
            candidate_1 = os.path.join(parent_dir, '..', 'labels', base_fname + '.json')
            candidate_2 = os.path.join(parent_dir, 'labels', base_fname + '.json')
            if os.path.exists(candidate_1):
                json_path = candidate_1
            elif os.path.exists(candidate_2):
                json_path = candidate_2

        if not os.path.exists(json_path):
            # Fallback: empty background mask
            mask_2d = np.zeros((1024, 1024), dtype=np.uint8)
            return cls._to_one_hot(mask_2d)

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Dynamic Canvas Sizing: Prevents Patient 32 (2160x3840) truncation
        img_h = data.get('imageHeight', 1080)
        img_w = data.get('imageWidth', 1920)

        mask_2d = np.zeros((img_h, img_w), dtype=np.uint8)

        # Map classes according to TopoNet convention:
        # 1: Anterior Ridge ('ridge')
        # 2: Liver Silhouette ('silhouette' or 'margin')
        # 3: Falciform Ligament ('falciform' or 'ligament')
        class_mapping = {
            'ridge': 1,
            'anterior': 1,
            'anterior_ridge': 1,
            'silhouette': 2,
            'margin': 2,
            'liver_silhouette': 2,
            'falciform': 3,
            'ligament': 3,
            'falciform_ligament': 3
        }

        shapes = data.get('shapes', [])
        for shape in shapes:
            label = shape.get('label', '').lower()
            cls_idx = None
            for key, idx in class_mapping.items():
                if key in label:
                    cls_idx = idx
                    break

            if cls_idx is not None:
                points = shape.get('points', [])
                for i in range(1, len(points)):
                    pt1 = tuple(map(int, points[i - 1]))
                    pt2 = tuple(map(int, points[i]))
                    cv2.line(mask_2d, pt1, pt2, cls_idx, 35)


        # Resize to standardized (1024, 1024) canvas with Nearest Neighbor to preserve discrete labels
        mask_2d = cv2.resize(mask_2d, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        return cls._to_one_hot(mask_2d)

    @staticmethod
    def _to_one_hot(mask_2d, num_classes=4):
        """Converts 2D integer mask (H, W) to (num_classes, H, W) float32 one-hot."""
        h, w = mask_2d.shape
        one_hot = np.zeros((num_classes, h, w), dtype=np.float32)
        for c in range(num_classes):
            one_hot[c] = (mask_2d == c).astype(np.float32)
        return one_hot
