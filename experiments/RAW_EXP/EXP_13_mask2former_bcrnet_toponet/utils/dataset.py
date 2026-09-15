"""
EXP_13: Multi-Modal Dataset Module for Mask2Former-BCRNet
=========================================================
Provides RGB-D surgical video frames, 4-class segmentation masks,
and fitted 5th-order Bézier curves for the L3D dataset.
"""
import os
import json
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional

from .bezier_ops import fit_bezier_least_squares, evaluate_bezier_torch


def resample_polyline_by_arclength(points: np.ndarray, step_size_px: float = 8.0) -> np.ndarray:
    """Resample polyline to uniform arc-length spacing."""
    if len(points) < 2:
        return points
    diffs = np.diff(points, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
    total = cum_dist[-1]
    if total <= 1e-4:
        return np.repeat(points[:1], 2, axis=0)
    n = max(int(np.ceil(total / step_size_px)) + 1, 4)
    t = np.linspace(0.0, total, n)
    rx = np.interp(t, cum_dist, points[:, 0])
    ry = np.interp(t, cum_dist, points[:, 1])
    return np.column_stack([rx, ry]).astype(np.float32)


class Mask2FormerBCRNetDataset(Dataset):
    """
    Unified Dataset for Mask2Former + TopoNet Loss + BCRNet.
    Classes:
      0: Ridge (Anterior Liver Ridge)
      1: Silhouette (Liver Silhouette)
      2: Ligament (Falciform Ligament)
    Background is class 0 in the 4-class segmentation map:
      0: Background, 1: Ridge, 2: Silhouette, 3: Ligament
    """
    CLASS_MAP = {
        "ridge": 0, "anterior_ridge": 0, "liver_ridge": 0,
        "silhouette": 1, "liver_silhouette": 1,
        "falciform": 2, "falciform_ligament": 2, "ligament": 2, "vessel": 2,
    }
    NUM_CLASSES = 3
    CLASS_NAMES = ["Ridge", "Silhouette", "Ligament"]

    def __init__(
        self,
        dataset_dir: str,
        mode: str = "train",
        image_size: int = 512,
        spline_step_px: float = 8.0,
        stroke_thickness: int = 2,
        use_depth: bool = True,
        bezier_degree: int = 5,
        num_sample_pts: int = 26,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.image_size = image_size
        self.spline_step_px = spline_step_px
        self.stroke_thickness = stroke_thickness
        self.use_depth = use_depth
        self.bezier_degree = bezier_degree
        self.num_ctrl_pts = bezier_degree + 1
        self.num_sample_pts = num_sample_pts

        split_dir = os.path.join(dataset_dir, mode)
        if not os.path.exists(split_dir):
            if os.path.exists(os.path.join(dataset_dir, mode.capitalize())):
                split_dir = os.path.join(dataset_dir, mode.capitalize())
            elif os.path.exists(os.path.join(dataset_dir, mode.lower())):
                split_dir = os.path.join(dataset_dir, mode.lower())
            else:
                split_dir = dataset_dir
        self.split_dir = split_dir

        img_dir = os.path.join(split_dir, "images")
        if os.path.exists(img_dir):
            self.image_paths = sorted(
                glob.glob(os.path.join(img_dir, "*.png")) +
                glob.glob(os.path.join(img_dir, "*.jpg"))
            )
        else:
            all_imgs = sorted(
                glob.glob(os.path.join(split_dir, "**", "*.png"), recursive=True) +
                glob.glob(os.path.join(split_dir, "**", "*.jpg"), recursive=True)
            )
            self.image_paths = [p for p in all_imgs if "masks" not in p and "depth" not in p]

        self.json_index = {}
        for jf in glob.glob(os.path.join(split_dir, "**", "*.json"), recursive=True):
            self.json_index[os.path.splitext(os.path.basename(jf))[0]] = jf

        self.depth_index = {}
        if use_depth:
            depth_globs = (
                glob.glob(os.path.join(split_dir, "**", "depth_anything_v2", "*.png"), recursive=True) +
                glob.glob(os.path.join(split_dir, "**", "depth_AdelaiDepth", "*.png"), recursive=True) +
                glob.glob(os.path.join(split_dir, "**", "depth", "*.png"), recursive=True) +
                glob.glob(os.path.join(dataset_dir, "**", "depth_anything_v2", "*.png"), recursive=True) +
                glob.glob(os.path.join(dataset_dir, "**", "depth_AdelaiDepth", "*.png"), recursive=True)
            )
            for df in depth_globs:
                self.depth_index[os.path.splitext(os.path.basename(df))[0]] = df

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

        print(
            f"[EXP_13 Dataset] {len(self.image_paths)} images | mode={mode} | "
            f"RGB-D={use_depth} | Bézier degree={bezier_degree} (K={self.num_ctrl_pts} ctrl pts)"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_annotations(self, base_name: str, orig_w: int, orig_h: int) -> Dict[int, np.ndarray]:
        json_path = self.json_index.get(base_name)
        if json_path is None or not os.path.exists(json_path):
            return {}

        with open(json_path, "r") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        class_polylines: Dict[int, List[np.ndarray]] = {}

        for shape in shapes:
            label = shape.get("label", "").lower().strip()
            cls_id = None
            for k, v in self.CLASS_MAP.items():
                if k in label:
                    cls_id = v
                    break
            if cls_id is None:
                continue

            raw_pts = np.array(shape.get("points", []), dtype=np.float32)
            if len(raw_pts) < 2:
                continue

            sx = float(self.image_size) / float(orig_w)
            sy = float(self.image_size) / float(orig_h)
            pts_scaled = raw_pts * np.array([sx, sy], dtype=np.float32)
            class_polylines.setdefault(cls_id, []).append(pts_scaled)

        result: Dict[int, np.ndarray] = {}
        for cls_id, poly_list in class_polylines.items():
            best = max(poly_list, key=len)
            resampled = resample_polyline_by_arclength(best, self.spline_step_px)
            result[cls_id] = resampled

        return result

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # 1. Load RGB
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        orig_h, orig_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        rgb_norm = (rgb_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
        rgb_norm = (rgb_norm - self.mean) / self.std

        # 2. Load Depth
        if self.use_depth:
            depth_path = self.depth_index.get(base_name)
            if depth_path and os.path.exists(depth_path):
                depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_raw is not None:
                    depth_resized = cv2.resize(
                        depth_raw.astype(np.float32),
                        (self.image_size, self.image_size),
                        interpolation=cv2.INTER_LINEAR
                    )
                    d_min, d_max = depth_resized.min(), depth_resized.max()
                    if d_max > d_min:
                        depth_norm = (depth_resized - d_min) / (d_max - d_min)
                    else:
                        depth_norm = np.zeros_like(depth_resized)
                else:
                    depth_norm = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            else:
                depth_norm = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            depth_ch = depth_norm[np.newaxis, :, :]  # (1, H, W)
            input_tensor = np.concatenate([rgb_norm, depth_ch], axis=0)  # (4, H, W)
        else:
            input_tensor = rgb_norm

        # 3. Load Annotations & Rasterize Masks
        polylines = self._load_annotations(base_name, orig_w, orig_h)
        gt_masks = np.zeros((self.NUM_CLASSES, self.image_size, self.image_size), dtype=np.float32)
        gt_ctrl_pts = np.zeros((self.NUM_CLASSES, self.num_ctrl_pts, 2), dtype=np.float32)
        gt_sample_pts = np.zeros((self.NUM_CLASSES, self.num_sample_pts, 2), dtype=np.float32)
        gt_exists = np.zeros(self.NUM_CLASSES, dtype=np.float32)

        for cls_id in range(self.NUM_CLASSES):
            if cls_id in polylines and len(polylines[cls_id]) >= 2:
                pts_px = polylines[cls_id]
                pts_norm = pts_px / float(self.image_size)

                # Draw binary mask
                mask_cv = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
                pts_int = np.round(pts_px).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(mask_cv, [pts_int], isClosed=False, color=1, thickness=self.stroke_thickness)
                gt_masks[cls_id] = mask_cv.astype(np.float32)

                # Fit 5th-order Bézier curve
                ctrl = fit_bezier_least_squares(pts_norm, degree=self.bezier_degree)
                gt_ctrl_pts[cls_id] = ctrl
                gt_exists[cls_id] = 1.0

                # Sample N=26 reference points along fitted curve
                ctrl_t = torch.from_numpy(ctrl).unsqueeze(0)
                eval_pts = evaluate_bezier_torch(ctrl_t, num_samples=self.num_sample_pts).squeeze(0).numpy()
                gt_sample_pts[cls_id] = eval_pts

        # 4-class semantic target: 0: Background, 1: Ridge, 2: Silhouette, 3: Ligament
        # (Foreground classes 0..2 map to 1..3 in semantic map)
        gt_semantic_4cls = np.zeros((self.image_size, self.image_size), dtype=np.int64)
        for c in range(self.NUM_CLASSES):
            gt_semantic_4cls[gt_masks[c] > 0.5] = c + 1

        return {
            "image": torch.from_numpy(input_tensor).float(),
            "gt_masks": torch.from_numpy(gt_masks).float(),              # (3, H, W)
            "gt_semantic": torch.from_numpy(gt_semantic_4cls).long(),    # (H, W)
            "gt_ctrl_pts": torch.from_numpy(gt_ctrl_pts).float(),        # (3, K, 2)
            "gt_sample_pts": torch.from_numpy(gt_sample_pts).float(),    # (3, N, 2)
            "gt_exists": torch.from_numpy(gt_exists).float(),            # (3,)
            "image_path": img_path,
            "base_name": base_name,
            "orig_rgb": rgb_resized,
        }
