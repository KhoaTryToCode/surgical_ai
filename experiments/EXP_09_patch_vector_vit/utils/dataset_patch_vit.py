import os
import json
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    from models.bezier_utils import resample_polyline_by_arclength, fit_cubic_bezier_least_squares
except ImportError:
    from ..models.bezier_utils import resample_polyline_by_arclength, fit_cubic_bezier_least_squares


class PatchBezierLandmarkDataset(Dataset):
    """
    Dataset for EXP_09 Patch-Level Bézier Vector Vision Transformer.
    
    Extracts dense arc-length resampled points using cubic splines,
    partitions them into 16x16 pixel spatial patches, and computes
    ground truth Cubic Bézier control points [P0, P1, P2, P3] via closed-form least squares.
    
    Output dictionary per sample:
        image:          (3, 512, 512) normalized RGB tensor
        target_classes: (32, 32) long tensor with class IDs {0: bg, 1: ridge, 2: silhouette, 3: ligament, 4: gallbladder}
        target_beziers: (32, 32, 4, 2) float tensor of Bézier control points in [0, 1]^2
        active_mask:    (32, 32) boolean mask of patches containing a landmark
        target_masks:   (4, 512, 512) float tensor of ground truth raster masks for evaluation
    """
    CLASS_MAP = {
        "ridge": 1, "anterior_ridge": 1, "liver_ridge": 1,
        "silhouette": 2, "liver_silhouette": 2,
        "falciform": 3, "falciform_ligament": 3, "ligament": 3, "vessel": 3,
        "gallbladder": 4, "gallbladder_boundary": 4
    }
    
    def __init__(
        self,
        dataset_dir: str,
        mode: str = "train",
        image_size: int = 512,
        patch_size: int = 16,
        spline_step_px: float = 8.0,
        stroke_thickness: int = 2,
        use_depth: bool = True
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # 32
        self.spline_step_px = spline_step_px
        self.stroke_thickness = stroke_thickness
        self.use_depth = use_depth
        
        # Resolve split directory
        split_dir = os.path.join(dataset_dir, mode)
        if not os.path.exists(split_dir):
            split_dir = dataset_dir
        self.split_dir = split_dir
        
        # Discover images
        self.img_dir = os.path.join(split_dir, "images")
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
            
        # Pre-index JSON files and depth maps
        self.json_index = {}
        for jf in glob.glob(os.path.join(split_dir, "**", "*.json"), recursive=True):
            j_name = os.path.splitext(os.path.basename(jf))[0]
            self.json_index[j_name] = jf

        self.depth_index = {}
        if self.use_depth:
            for df in (
                glob.glob(os.path.join(split_dir, "**", "depth_anything_v2", "*.png"), recursive=True) +
                glob.glob(os.path.join(split_dir, "**", "depth_anything_v2", "*.jpg"), recursive=True) +
                glob.glob(os.path.join(split_dir, "**", "depth", "*.png"), recursive=True) +
                glob.glob(os.path.join(dataset_dir, "**", "depth_anything_v2", "*.png"), recursive=True)
            ):
                d_name = os.path.splitext(os.path.basename(df))[0]
                self.depth_index[d_name] = df
            
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        
        found_anns = sum(1 for p in self.image_paths if self._find_annotation_path(p) is not None)
        found_depth = sum(1 for p in self.image_paths if self._find_depth_path(p) is not None) if self.use_depth else 0
        print(f"📐 [PATCH-BÉZIER DATASET] {len(self.image_paths)} Images | {found_anns} Annotated JSONs | {found_depth} Depth Maps (RGB-D: {self.use_depth}) | Grid: {self.grid_size}×{self.grid_size}")

    def __len__(self) -> int:
        return max(len(self.image_paths), 1)

    def _find_depth_path(self, img_path: str):
        base_no_ext = os.path.splitext(img_path)[0]
        base_name = os.path.basename(base_no_ext)
        if base_name in self.depth_index:
            return self.depth_index[base_name]

        img_dir = os.path.dirname(img_path)
        candidates = [
            img_path.replace("images", "depth_anything_v2").replace(".jpg", ".png").replace(".jpeg", ".png"),
            img_path.replace("images", "depth").replace(".jpg", ".png").replace(".jpeg", ".png"),
            os.path.join(os.path.dirname(img_dir), "depth_anything_v2", f"{base_name}.png"),
            os.path.join(os.path.dirname(img_dir), "depth", f"{base_name}.png"),
            os.path.join(img_dir, f"{base_name}_depth.png")
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _find_annotation_path(self, img_path: str):
        base_no_ext = os.path.splitext(img_path)[0]
        base_name = os.path.basename(base_no_ext)
        if base_name in self.json_index:
            return self.json_index[base_name]
        
        img_dir = os.path.dirname(img_path)
        candidates = [
            img_path.replace("images", "labels").replace(".jpg", ".json").replace(".png", ".json"),
            img_path.replace("images", "annotations").replace(".jpg", ".json").replace(".png", ".json"),
            os.path.join(img_dir, f"{base_name}.json"),
            os.path.join(os.path.dirname(img_dir), "labels", f"{base_name}.json"),
            os.path.join(os.path.dirname(img_dir), "annotations", f"{base_name}.json"),
            f"{base_no_ext}.json"
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def __getitem__(self, idx: int) -> dict:
        S = self.image_size
        P = self.patch_size
        G = self.grid_size
        
        # If no images exist on disk, synthesize a realistic procedural curve for testing
        if len(self.image_paths) == 0:
            return self._generate_synthetic_sample(S, P, G)
            
        img_path = self.image_paths[idx]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            img_rgb = np.ones((S, S, 3), dtype=np.float32) * 0.5
            orig_h, orig_w = S, S
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img_rgb.shape[:2]
            if img_rgb.shape[:2] != (S, S):
                img_rgb = cv2.resize(img_rgb, (S, S), interpolation=cv2.INTER_LINEAR)
            img_rgb = img_rgb.astype(np.float32) / 255.0

        img_norm = (img_rgb - self.mean) / self.std
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()  # (3, S, S)
        
        # Optional 4th Channel: Depth Anything V2 Map
        if self.use_depth:
            depth_path = self._find_depth_path(img_path)
            if depth_path is not None and os.path.exists(depth_path):
                depth_raw = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                if depth_raw is not None:
                    if depth_raw.shape[:2] != (S, S):
                        depth_raw = cv2.resize(depth_raw, (S, S), interpolation=cv2.INTER_LINEAR)
                    # Normalize depth to zero-mean unit-variance
                    depth_norm = (depth_raw.astype(np.float32) / 255.0 - 0.5) / 0.25
                    depth_tensor = torch.from_numpy(depth_norm).unsqueeze(0).float()
                else:
                    depth_tensor = torch.zeros((1, S, S), dtype=torch.float32)
            else:
                depth_tensor = torch.zeros((1, S, S), dtype=torch.float32)
            img_tensor = torch.cat([img_tensor, depth_tensor], dim=0)  # (4, S, S)
        
        # Target arrays
        target_classes = np.zeros((G, G), dtype=np.int64)
        target_beziers = np.zeros((G, G, 4, 2), dtype=np.float32)
        active_mask = np.zeros((G, G), dtype=bool)
        target_masks = np.zeros((4, S, S), dtype=np.float32)
        
        ann_path = self._find_annotation_path(img_path)
        if ann_path is not None and os.path.exists(ann_path):
            try:
                with open(ann_path, 'r') as f:
                    data = json.load(f)
                    
                shapes = data.get("shapes", [])
                for shape in shapes:
                    label = str(shape.get("label", "")).lower().replace("-", "_").replace(" ", "_")
                    
                    if label.startswith('r') or 'ridge' in label:
                        cls_id = 1
                    elif label.startswith('s') or 'silhouette' in label:
                        cls_id = 2
                    elif label.startswith('l') or 'ligament' in label or 'falciform' in label or 'vessel' in label:
                        cls_id = 3
                    elif 'gall' in label:
                        cls_id = 4
                    else:
                        cls_id = self.CLASS_MAP.get(label, 1)
                        
                    raw_pts = np.array(shape.get("points", []), dtype=np.float32)
                    if len(raw_pts) < 2:
                        continue
                        
                    # Scale raw points to (S, S) image size
                    pts_scaled = raw_pts.copy()
                    pts_scaled[:, 0] = pts_scaled[:, 0] * (float(S) / float(orig_w))
                    pts_scaled[:, 1] = pts_scaled[:, 1] * (float(S) / float(orig_h))
                    
                    # 1. Cubic Spline Arc-Length Resampling
                    dense_pts = resample_polyline_by_arclength(pts_scaled, step_size_px=self.spline_step_px)
                    if len(dense_pts) < 2:
                        continue
                        
                    # 2. Render evaluation mask
                    pts_pix = np.clip(np.round(dense_pts), 0, S - 1).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(target_masks[cls_id - 1], [pts_pix], isClosed=False, color=1.0, thickness=self.stroke_thickness, lineType=cv2.LINE_AA)
                    
                    # 3. Partition into spatial patches and collect points per patch
                    patch_points_dict = {}
                    for pt in dense_pts:
                        r = int(np.clip(pt[1] // P, 0, G - 1))
                        c = int(np.clip(pt[0] // P, 0, G - 1))
                        key = (r, c)
                        if key not in patch_points_dict:
                            patch_points_dict[key] = []
                        patch_points_dict[key].append(pt)
                        
                    # 4. Fit cubic Bézier curve per traversed patch
                    for (r, c), pts_in_patch in patch_points_dict.items():
                        pts_arr = np.array(pts_in_patch, dtype=np.float32)
                        # Normalize to local patch coordinates [0, 1]^2
                        pts_local = pts_arr.copy()
                        pts_local[:, 0] = (pts_local[:, 0] - c * P) / float(P)
                        pts_local[:, 1] = (pts_local[:, 1] - r * P) / float(P)
                        pts_local = np.clip(pts_local, 0.0, 1.0)
                        
                        ctrl_pts = fit_cubic_bezier_least_squares(pts_local)
                        
                        target_classes[r, c] = cls_id
                        target_beziers[r, c] = ctrl_pts
                        active_mask[r, c] = True
            except Exception:
                pass

        return {
            "image": img_tensor,
            "target_classes": torch.from_numpy(target_classes).long(),
            "target_beziers": torch.from_numpy(target_beziers).float(),
            "active_mask": torch.from_numpy(active_mask).bool(),
            "target_masks": torch.from_numpy(target_masks).float(),
            "img_path": img_path
        }

    def _generate_synthetic_sample(self, S: int, P: int, G: int) -> dict:
        """Procedural synthetic sample for offline verification when real data is unavailable."""
        img_rgb = np.ones((S, S, 3), dtype=np.float32) * 0.2
        img_norm = (img_rgb - self.mean) / self.std
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
        
        if self.use_depth:
            synth_depth = np.linspace(0.1, 0.9, S).reshape(1, S).repeat(S, axis=0).astype(np.float32)
            depth_norm = (synth_depth - 0.5) / 0.25
            depth_tensor = torch.from_numpy(depth_norm).unsqueeze(0).float()
            img_tensor = torch.cat([img_tensor, depth_tensor], dim=0)
        
        target_classes = np.zeros((G, G), dtype=np.int64)
        target_beziers = np.zeros((G, G, 4, 2), dtype=np.float32)
        active_mask = np.zeros((G, G), dtype=bool)
        target_masks = np.zeros((4, S, S), dtype=np.float32)
        
        # Synthetic curved trajectory: an ellipse arc across the liver cavity
        t = np.linspace(0.2 * np.pi, 0.8 * np.pi, 60)
        cx, cy = S * 0.5, S * 0.5
        rx, ry = S * 0.35, S * 0.25
        dense_x = cx + rx * np.cos(t)
        dense_y = cy + ry * np.sin(t)
        dense_pts = np.stack([dense_x, dense_y], axis=1).astype(np.float32)
        
        cls_id = 1  # Ridge
        pts_pix = np.clip(np.round(dense_pts), 0, S - 1).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(target_masks[cls_id - 1], [pts_pix], isClosed=False, color=1.0, thickness=self.stroke_thickness, lineType=cv2.LINE_AA)
        
        patch_points_dict = {}
        for pt in dense_pts:
            r = int(np.clip(pt[1] // P, 0, G - 1))
            c = int(np.clip(pt[0] // P, 0, G - 1))
            key = (r, c)
            if key not in patch_points_dict:
                patch_points_dict[key] = []
            patch_points_dict[key].append(pt)
            
        for (r, c), pts_in_patch in patch_points_dict.items():
            pts_arr = np.array(pts_in_patch, dtype=np.float32)
            pts_local = pts_arr.copy()
            pts_local[:, 0] = (pts_local[:, 0] - c * P) / float(P)
            pts_local[:, 1] = (pts_local[:, 1] - r * P) / float(P)
            pts_local = np.clip(pts_local, 0.0, 1.0)
            
            ctrl_pts = fit_cubic_bezier_least_squares(pts_local)
            target_classes[r, c] = cls_id
            target_beziers[r, c] = ctrl_pts
            active_mask[r, c] = True
            
        return {
            "image": img_tensor,
            "target_classes": torch.from_numpy(target_classes).long(),
            "target_beziers": torch.from_numpy(target_beziers).float(),
            "active_mask": torch.from_numpy(active_mask).bool(),
            "target_masks": torch.from_numpy(target_masks).float(),
            "img_path": "synthetic_sample"
        }
