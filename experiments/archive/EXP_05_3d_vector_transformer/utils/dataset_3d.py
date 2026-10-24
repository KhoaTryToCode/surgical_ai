import os
import glob
import json
import math
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from .spline_utils import resample_polyline_arc_length, draw_polyline_mask

class Surgical3DVectorDataset(Dataset):
    """
    Surgical AI Dataset for Monocular 3D Vector Space Polyline Detection.
    Loads:
      - RGB Image: (3, 1024, 1024)
      - Depth Map: (1, 1024, 1024) relative depth from Depth Anything V2
      - Resampled 3D GT Polylines: (max_instances, K, 3) normalized to [-1, 1]^3
      - Class Labels: (max_instances,) integer class IDs
      - 2D Rasterized Masks: (max_instances, 1024, 1024) binary mask overlays
      - Valid Instances Mask: (max_instances,) boolean flag indicating active curves vs padded slots
    """
    def __init__(self, dataset_dir: str, num_instances: int = 10, num_points: int = 20, mode: str = "train"):
        self.dataset_dir = dataset_dir
        self.num_instances = num_instances
        self.num_points = num_points
        self.mode = mode

        # Find image files for current split (mode = train, val, test)
        split_dir = os.path.join(dataset_dir, mode)
        if os.path.exists(split_dir):
            self.img_files = sorted(glob.glob(os.path.join(split_dir, "images", "*.jpg")))
            if not self.img_files:
                self.img_files = sorted(glob.glob(os.path.join(split_dir, "**", "*.jpg"), recursive=True))
        else:
            self.img_files = sorted(glob.glob(os.path.join(dataset_dir, "images", "*.jpg")))
            if not self.img_files:
                self.img_files = sorted(glob.glob(os.path.join(dataset_dir, "**", "*.jpg"), recursive=True))

        self.transform_img = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def _get_label_path(self, img_path: str) -> str:
        return img_path.replace("images", "labels").replace(".jpg", ".json")

    def _get_depth_path(self, img_path: str) -> str:
        depth_p = img_path.replace("images", "depth").replace(".jpg", ".png")
        if os.path.exists(depth_p):
            return depth_p
        return img_path.replace(".jpg", "_depth.png")

    def __getitem__(self, idx: int):
        img_path = self.img_files[idx]
        
        # 1. Load RGB Image
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]
        rgb_resized = cv2.resize(rgb, (1024, 1024))
        image_tensor = self.transform_img(rgb_resized) # (3, 1024, 1024)

        # 2. Load Depth Map (or generate synthetic depth fallback)
        depth_path = self._get_depth_path(img_path)
        depth_norm = None
        if os.path.exists(depth_path) and os.path.getsize(depth_path) > 0:
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            if depth_img is not None and depth_img.size > 0:
                depth_resized = cv2.resize(depth_img, (1024, 1024))
                depth_norm = depth_resized.astype(np.float32) / 255.0

        if depth_norm is None or depth_norm.ndim != 2 or depth_norm.shape != (1024, 1024):
            # Synthetic fallback smooth depth map
            y_grid, x_grid = np.ogrid[:1024, :1024]
            depth_norm = (0.5 + 0.3 * (y_grid / 1024.0)).astype(np.float32)

        depth_tensor = torch.from_numpy(depth_norm).unsqueeze(0).float() # (1, 1024, 1024)

        # 3. Load GT Landmarks JSON
        gt_classes = []
        gt_polylines = []
        gt_masks = []

        label_path = self._get_label_path(img_path)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                data = json.load(f)

            for shape in data.get('shapes', []):
                pts = np.array(shape.get('points', []), dtype=np.float32)
                label = shape.get('label', '')
                if len(pts) < 2:
                    continue

                # Map class label to ID
                if label.startswith('r') or 'ridge' in label.lower():
                    cid = 1 # Ridge
                elif label.startswith('s') or 'silhouette' in label.lower():
                    cid = 2 # Silhouette
                elif label.startswith('l') or 'ligament' in label.lower() or 'vessel' in label.lower():
                    cid = 3 # Ligament / Vessel
                else:
                    cid = 1

                # Normalize 2D points to 1024x1024 scale
                pts[:, 0] = pts[:, 0] * (1024.0 / orig_w)
                pts[:, 1] = pts[:, 1] * (1024.0 / orig_h)

                # Resample 2D polyline to K points
                pts_2d_k = resample_polyline_arc_length(pts, K=self.num_points)

                # Unproject 2D (u,v,d) to canonical 3D space [-1, 1]^3 matching backbone pinhole math
                u_norm = (pts_2d_k[:, 0] - 512.0) / 512.0
                v_norm = (pts_2d_k[:, 1] - 512.0) / 512.0
                
                # Sample depth along the polyline vertices with safe bounds checking
                z_vals = []
                dh, dw = depth_norm.shape[:2]
                for pt in pts_2d_k:
                    px = int(np.clip(pt[0], 0, dw - 1))
                    py = int(np.clip(pt[1], 0, dh - 1))
                    z_vals.append(depth_norm[py, px])
                z_arr = np.array(z_vals, dtype=np.float32)

                # Canonical focal length (60 deg FOV -> f = 1.732)
                f_canon = 1.0 / math.tan(math.radians(60.0 / 2.0))
                z_canon = 0.1 + z_arr * 0.9
                x_canon = (u_norm * z_canon) / f_canon
                y_canon = (v_norm * z_canon) / f_canon

                x_norm = np.clip(x_canon, -1.0, 1.0)
                y_norm = np.clip(y_canon, -1.0, 1.0)
                z_norm = z_canon * 2.0 - 1.0

                pts_3d_k = np.stack([x_norm, y_norm, z_norm], axis=1) # (K, 3)

                # Rasterize 2D mask
                mask_2d = draw_polyline_mask(pts_2d_k, height=1024, width=1024, stroke_thickness=35)

                gt_classes.append(cid)
                gt_polylines.append(pts_3d_k)
                gt_masks.append(mask_2d)

        # Pad or truncate to fixed num_instances (N)
        num_gt = len(gt_classes)
        target_classes = torch.zeros(self.num_instances, dtype=torch.long) # 0 = Background
        target_polylines = torch.zeros((self.num_instances, self.num_points, 3), dtype=torch.float32)
        target_masks = torch.zeros((self.num_instances, 1024, 1024), dtype=torch.float32)
        valid_mask = torch.zeros(self.num_instances, dtype=torch.bool)

        for i in range(min(num_gt, self.num_instances)):
            target_classes[i] = gt_classes[i]
            target_polylines[i] = torch.from_numpy(gt_polylines[i])
            target_masks[i] = torch.from_numpy(gt_masks[i])
            valid_mask[i] = True

        return {
            "image": image_tensor,                   # (3, 1024, 1024)
            "depth": depth_tensor,                   # (1, 1024, 1024)
            "target_classes": target_classes,         # (N,)
            "target_polylines": target_polylines,     # (N, K, 3)
            "target_masks": target_masks,             # (N, 1024, 1024)
            "valid_mask": valid_mask,                 # (N,)
            "img_path": img_path
        }
