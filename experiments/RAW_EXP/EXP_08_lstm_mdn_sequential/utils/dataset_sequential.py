import os
import json
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def resample_polyline_2d(points: np.ndarray, num_points: int = 20) -> np.ndarray:
    """
    Resamples a 2D polyline (N, 2) into exactly K equidistant points along its arc-length.
    Coordinates are expected and returned in [0.0, 1.0]^2.
    
    Reused from EXP_06 dataset utility — identical arc-length interpolation logic.
    """
    if len(points) == 0:
        return np.zeros((num_points, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, num_points, axis=0).astype(np.float32)
    
    # Calculate cumulative arc-length distances
    diffs = np.diff(points, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=-1))
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_len = cum_dists[-1]
    
    if total_len < 1e-6:
        return np.repeat(points[:1], num_points, axis=0).astype(np.float32)
        
    target_dists = np.linspace(0, total_len, num_points)
    
    # Interpolate x and y independently
    x_interp = np.interp(target_dists, cum_dists, points[:, 0])
    y_interp = np.interp(target_dists, cum_dists, points[:, 1])
    
    return np.stack([x_interp, y_interp], axis=-1).astype(np.float32)


class SequentialLandmarkDataset(Dataset):
    """
    Dataset for EXP_08 CNN-LSTM-MDN Sequential Landmark Detection.
    
    Reuses the JSON annotation parsing from EXP_06 but adapts for:
    - 512×512 image resolution (reduced from 1024 for faster training)
    - 512×512 mask rasterization
    - Same polyline format: (N, K=20, 2) in [0.0, 1.0]^2
    
    Output dict per sample:
        image:            (3, 512, 512) normalized RGB tensor
        target_classes:   (N,) class IDs in {0, 1, 2, 3, 4}
        target_polylines: (N, K, 2) normalized 2D coordinates
        target_masks:     (N, 512, 512) rasterized binary masks
        valid_mask:       (N,) boolean active instance indicators
    """
    
    CLASS_MAP = {
        "falciform": 3, "falciform_ligament": 3, "ligament": 3, "vessel": 3,
        "ridge": 1, "anterior_ridge": 1, "liver_ridge": 1,
        "silhouette": 2, "liver_silhouette": 2,
        "gallbladder": 4, "gallbladder_boundary": 4
    }
    
    def __init__(
        self,
        dataset_dir: str,
        num_instances: int = 10,
        num_points: int = 20,
        mode: str = "train",
        image_size: int = 512,
        stroke_thickness: int = 18
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.num_instances = num_instances
        self.num_points = num_points
        self.mode = mode
        self.image_size = image_size
        self.stroke_thickness = stroke_thickness
        
        # Resolve split directory
        split_dir = os.path.join(dataset_dir, mode)
        if not os.path.exists(split_dir):
            split_dir = dataset_dir
        
        self.split_dir = split_dir
        self.img_dir = os.path.join(split_dir, "images")
        self.ann_dir = os.path.join(split_dir, "annotations")
        
        # Discover image files
        if os.path.exists(self.img_dir):
            self.image_paths = sorted(
                glob.glob(os.path.join(self.img_dir, "*.png")) +
                glob.glob(os.path.join(self.img_dir, "*.jpg"))
            )
        else:
            self.image_paths = sorted(
                glob.glob(os.path.join(split_dir, "**", "*.png"), recursive=True) +
                glob.glob(os.path.join(split_dir, "**", "*.jpg"), recursive=True)
            )
        
        # Pre-index all JSON annotation files
        self.json_index = {}
        for jf in glob.glob(os.path.join(split_dir, "**", "*.json"), recursive=True):
            j_name = os.path.splitext(os.path.basename(jf))[0]
            self.json_index[j_name] = jf
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        
        found_anns = sum(1 for p in self.image_paths if self._find_annotation_path(p) is not None)
        print(f"📊 [{mode.upper()} DATASET] {len(self.image_paths)} Images | {found_anns} Annotated JSONs | {image_size}×{image_size}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _find_annotation_path(self, img_path: str):
        """
        Locate JSON annotation for a given image path.
        Tries multiple directory conventions (labels/, annotations/, label/, co-located).
        """
        base_no_ext = os.path.splitext(img_path)[0]
        base_name = os.path.basename(base_no_ext)
        
        if base_name in self.json_index:
            return self.json_index[base_name]
        
        img_dir = os.path.dirname(img_path)
        candidates = [
            img_path.replace("images", "labels").replace(".jpg", ".json").replace(".png", ".json"),
            img_path.replace("images", "annotations").replace(".jpg", ".json").replace(".png", ".json"),
            img_path.replace("images", "label").replace(".jpg", ".json").replace(".png", ".json"),
            os.path.join(img_dir, f"{base_name}.json"),
            os.path.join(os.path.dirname(img_dir), "labels", f"{base_name}.json"),
            os.path.join(os.path.dirname(img_dir), "annotations", f"{base_name}.json"),
            os.path.join(os.path.dirname(img_dir), "label", f"{base_name}.json"),
            f"{base_no_ext}.json"
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None
    
    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        S = self.image_size  # 512
        
        # 1. Load RGB Image → resize to SxS
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            img_rgb = np.zeros((S, S, 3), dtype=np.float32)
            orig_h, orig_w = S, S
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img_rgb.shape[:2]
            if img_rgb.shape[:2] != (S, S):
                img_rgb = cv2.resize(img_rgb, (S, S), interpolation=cv2.INTER_LINEAR)
            img_rgb = img_rgb.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats → Tensor (3, S, S)
        img_norm = (img_rgb - self.mean) / self.std
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
        
        # 2. Parse Landmark Annotations
        ann_path = self._find_annotation_path(img_path)
        
        target_classes = np.zeros((self.num_instances,), dtype=np.int64)
        target_polylines = np.zeros((self.num_instances, self.num_points, 2), dtype=np.float32)
        target_masks = np.zeros((self.num_instances, S, S), dtype=np.float32)
        valid_mask = np.zeros((self.num_instances,), dtype=bool)
        
        if ann_path is not None and os.path.exists(ann_path):
            try:
                with open(ann_path, 'r') as f:
                    data = json.load(f)
                
                shapes = data.get("shapes", [])
                inst_idx = 0
                
                for shape in shapes:
                    if inst_idx >= self.num_instances:
                        break
                    
                    label = str(shape.get("label", "")).lower().replace("-", "_").replace(" ", "_")
                    
                    # Class label parsing (matches EXP_06 convention)
                    if label.startswith('r') or 'ridge' in label:
                        cls_id = 1  # Ridge
                    elif label.startswith('s') or 'silhouette' in label:
                        cls_id = 2  # Silhouette
                    elif label.startswith('l') or 'ligament' in label or 'falciform' in label or 'vessel' in label:
                        cls_id = 3  # Falciform Ligament / Vessel
                    elif 'gall' in label:
                        cls_id = 4  # Gallbladder
                    else:
                        cls_id = self.CLASS_MAP.get(label, 1)
                    
                    raw_pts = np.array(shape.get("points", []), dtype=np.float32)
                    if len(raw_pts) < 2:
                        continue
                    
                    # Scale raw points to [0, 1] normalized coordinates
                    pts_norm = raw_pts.copy()
                    pts_norm[:, 0] = np.clip(pts_norm[:, 0] / float(orig_w), 0.0, 1.0)
                    pts_norm[:, 1] = np.clip(pts_norm[:, 1] / float(orig_h), 0.0, 1.0)
                    
                    # Resample to exactly K equidistant points
                    poly_2d = resample_polyline_2d(pts_norm, num_points=self.num_points)
                    
                    # Rasterize binary stroke mask at SxS
                    mask_canvas = np.zeros((S, S), dtype=np.uint8)
                    pts_pix = (poly_2d * float(S)).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(mask_canvas, [pts_pix], isClosed=False, color=1, thickness=self.stroke_thickness)
                    
                    target_classes[inst_idx] = cls_id
                    target_polylines[inst_idx] = poly_2d
                    target_masks[inst_idx] = mask_canvas.astype(np.float32)
                    valid_mask[inst_idx] = True
                    
                    inst_idx += 1
                    
            except Exception:
                pass
        
        return {
            "image": img_tensor,
            "target_classes": torch.from_numpy(target_classes).long(),
            "target_polylines": torch.from_numpy(target_polylines).float(),
            "target_masks": torch.from_numpy(target_masks).float(),
            "valid_mask": torch.from_numpy(valid_mask).bool()
        }
