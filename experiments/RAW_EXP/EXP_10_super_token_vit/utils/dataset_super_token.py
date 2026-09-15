import os
import json
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from models.spline_utils import resample_polyline_by_arclength, fit_bezier_least_squares_np
except ImportError:
    from ..models.spline_utils import resample_polyline_by_arclength, fit_bezier_least_squares_np


class SuperTokenLandmarkDataset(Dataset):
    """
    Dataset for EXP_10 Super-Token Geometric Vision Transformer.
    
    Produces:
        image:               (4, 512, 512) normalized RGB-D tensor
        target_exists:       (4,) float tensor in {0, 1}
        target_ctrl_points:  (4, K, 2) global Bézier control points in [0, 1]^2
        target_attn_masks:   (4, 32, 32) patch attention ground truth
        target_render_masks: (4, 128, 128) soft raster target for training Dice loss
        target_eval_masks:   (4, 512, 512) full-res benchmark masks for official validation
        img_path:            str
    """
    CLASS_MAP = {
        "ridge": 0, "anterior_ridge": 0, "liver_ridge": 0,
        "silhouette": 1, "liver_silhouette": 1,
        "falciform": 2, "falciform_ligament": 2, "ligament": 2, "vessel": 2,
        "gallbladder": 3, "gallbladder_boundary": 3
    }
    
    def __init__(
        self,
        dataset_dir: str,
        mode: str = "train",
        image_size: int = 512,
        patch_size: int = 16,
        num_ctrl_points: int = 6,
        render_size: int = 128,
        stroke_thickness: int = 2,
        use_depth: bool = True
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # 32
        self.num_ctrl_points = num_ctrl_points
        self.render_size = render_size
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
            # Filter out masks if they were swept in
            self.image_paths = [p for p in self.image_paths if "masks" not in p and "depth" not in p]
            
        # Index JSON annotations, depth maps, and GT masks
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
                
        self.mask_index = {}
        for mf in (
            glob.glob(os.path.join(split_dir, "**", "masks_gt", "*.png"), recursive=True) +
            glob.glob(os.path.join(dataset_dir, "**", "masks_gt", "*.png"), recursive=True)
        ):
            m_name = os.path.splitext(os.path.basename(mf))[0]
            self.mask_index[m_name] = mf

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        
        found_anns = sum(1 for p in self.image_paths if self._find_annotation_path(p) is not None)
        found_depth = sum(1 for p in self.image_paths if self._find_depth_path(p) is not None) if self.use_depth else 0
        found_masks = sum(1 for p in self.image_paths if self._find_mask_path(p) is not None)
        print(f"🌟 [SUPER-TOKEN DATASET] {len(self.image_paths)} Images | {found_anns} JSONs | {found_masks} GT Masks | {found_depth} Depth Maps (RGB-D: {self.use_depth})")

    def __len__(self) -> int:
        return max(len(self.image_paths), 1)

    def _find_depth_path(self, img_path: str):
        base_no_ext = os.path.splitext(img_path)[0]
        base_name = os.path.basename(base_no_ext)
        if base_name in self.depth_index:
            return self.depth_index[base_name]

        img_dir = os.path.dirname(img_path)
        candidates = [
            img_path.replace("images", "depth_anything_v2").replace(".jpg", ".png"),
            img_path.replace("images", "depth").replace(".jpg", ".png"),
            os.path.join(os.path.dirname(img_dir), "depth_anything_v2", f"{base_name}.png"),
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
            os.path.join(img_dir, f"{base_name}.json"),
            os.path.join(os.path.dirname(img_dir), "labels", f"{base_name}.json"),
            f"{base_no_ext}.json"
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _find_mask_path(self, img_path: str):
        base_no_ext = os.path.splitext(img_path)[0]
        base_name = os.path.basename(base_no_ext)
        if base_name in self.mask_index:
            return self.mask_index[base_name]
        candidates = [
            img_path.replace("images", "masks_gt"),
            os.path.join(os.path.dirname(os.path.dirname(img_path)), "masks_gt", f"{base_name}.png")
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def __getitem__(self, idx: int) -> dict:
        S = self.image_size
        G = self.grid_size
        R = self.render_size
        K = self.num_ctrl_points
        C = 4
        
        if len(self.image_paths) == 0:
            return self._generate_synthetic_sample(S, G, R, K, C)
            
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
        
        # 4th Channel: Depth
        if self.use_depth:
            depth_path = self._find_depth_path(img_path)
            if depth_path is not None and os.path.exists(depth_path):
                depth_raw = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                if depth_raw is not None:
                    if depth_raw.shape[:2] != (S, S):
                        depth_raw = cv2.resize(depth_raw, (S, S), interpolation=cv2.INTER_LINEAR)
                    depth_norm = (depth_raw.astype(np.float32) / 255.0 - 0.5) / 0.25
                    depth_tensor = torch.from_numpy(depth_norm).unsqueeze(0).float()
                else:
                    depth_tensor = torch.zeros((1, S, S), dtype=torch.float32)
            else:
                depth_tensor = torch.zeros((1, S, S), dtype=torch.float32)
            img_tensor = torch.cat([img_tensor, depth_tensor], dim=0)  # (4, S, S)
            
        # Target containers
        target_exists = np.zeros(C, dtype=np.float32)
        target_ctrl_points = np.zeros((C, K, 2), dtype=np.float32)
        target_attn_masks = np.zeros((C, G, G), dtype=np.float32)
        target_render_masks = np.zeros((C, R, R), dtype=np.float32)
        target_eval_masks = np.zeros((C, S, S), dtype=np.float32)
        
        # 1. Check if official precomputed GT mask exists
        mask_path = self._find_mask_path(img_path)
        if mask_path is not None and os.path.exists(mask_path):
            gt_mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask_raw is not None:
                if gt_mask_raw.shape[:2] != (S, S):
                    gt_mask_raw = cv2.resize(gt_mask_raw, (S, S), interpolation=cv2.INTER_NEAREST)
                # Split channels if multi-class grayscale or binary
                bin_mask = (gt_mask_raw > 127).astype(np.float32)
                # If binary mask, assign to active classes based on JSON or default to ridge/silhouette
                target_eval_masks[0] = bin_mask
                
        # 2. Extract JSON polyline annotations and fit continuous Béziers
        ann_path = self._find_annotation_path(img_path)
        if ann_path is not None and os.path.exists(ann_path):
            try:
                with open(ann_path, 'r') as f:
                    data = json.load(f)
                    
                shapes = data.get("shapes", [])
                for shape in shapes:
                    label = str(shape.get("label", "")).lower().replace("-", "_").replace(" ", "_")
                    if label.startswith('r') or 'ridge' in label:
                        cls_idx = 0
                    elif label.startswith('s') or 'silhouette' in label:
                        cls_idx = 1
                    elif label.startswith('l') or 'ligament' in label or 'falciform' in label or 'vessel' in label:
                        cls_idx = 2
                    elif 'gall' in label:
                        cls_idx = 3
                    else:
                        cls_idx = self.CLASS_MAP.get(label, 0)
                        
                    raw_pts = np.array(shape.get("points", []), dtype=np.float32)
                    if len(raw_pts) < 2:
                        continue
                        
                    # Scale to (S, S) image size
                    pts_scaled = raw_pts.copy()
                    pts_scaled[:, 0] = pts_scaled[:, 0] * (float(S) / float(orig_w))
                    pts_scaled[:, 1] = pts_scaled[:, 1] * (float(S) / float(orig_h))
                    
                    # Arc-length resample
                    dense_pts = resample_polyline_by_arclength(pts_scaled, step_size_px=6.0)
                    if len(dense_pts) < 2:
                        continue
                        
                    target_exists[cls_idx] = 1.0
                    
                    # Fit degree-(K-1) Bézier curve in normalized [0, 1]^2 image space
                    norm_pts = dense_pts / float(S)
                    ctrl_pts = fit_bezier_least_squares_np(norm_pts, degree=K - 1)
                    target_ctrl_points[cls_idx] = np.clip(ctrl_pts, 0.0, 1.0)
                    
                    # Render evaluation mask (S, S)
                    pts_pix_S = np.clip(np.round(dense_pts), 0, S - 1).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(target_eval_masks[cls_idx], [pts_pix_S], isClosed=False, color=1.0, thickness=self.stroke_thickness)
                    
                    # Render soft target mask at render resolution (R, R)
                    pts_pix_R = np.clip(np.round(dense_pts * (float(R) / float(S))), 0, R - 1).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(target_render_masks[cls_idx], [pts_pix_R], isClosed=False, color=1.0, thickness=max(1, int(self.stroke_thickness * (R / S))))
                    
                    # Compute Patch-Level Attention Target (32, 32)
                    P = self.patch_size
                    for pt in dense_pts:
                        r = int(np.clip(pt[1] // P, 0, G - 1))
                        c = int(np.clip(pt[0] // P, 0, G - 1))
                        target_attn_masks[cls_idx, r, c] = 1.0
            except Exception:
                pass
                
        return {
            "image": img_tensor,
            "target_exists": torch.from_numpy(target_exists).float(),
            "target_ctrl_points": torch.from_numpy(target_ctrl_points).float(),
            "target_attn_masks": torch.from_numpy(target_attn_masks).float(),
            "target_render_masks": torch.from_numpy(target_render_masks).float(),
            "target_eval_masks": torch.from_numpy(target_eval_masks).float(),
            "img_path": img_path
        }

    def _generate_synthetic_sample(self, S: int, G: int, R: int, K: int, C: int) -> dict:
        """Procedural synthetic sample for offline smoke testing."""
        img_rgb = np.zeros((3, S, S), dtype=np.float32)
        depth = np.zeros((1, S, S), dtype=np.float32)
        img_tensor = torch.cat([torch.from_numpy(img_rgb), torch.from_numpy(depth)], dim=0)
        
        target_exists = torch.zeros(C, dtype=torch.float32)
        target_ctrl_points = torch.zeros((C, K, 2), dtype=torch.float32)
        target_attn_masks = torch.zeros((C, G, G), dtype=torch.float32)
        target_render_masks = torch.zeros((C, R, R), dtype=torch.float32)
        target_eval_masks = torch.zeros((C, S, S), dtype=torch.float32)
        
        # Synthesize a smooth curved landmark for class 0 (Ridge)
        target_exists[0] = 1.0
        ts = np.linspace(0.1, 0.9, K)
        for i, t in enumerate(ts):
            target_ctrl_points[0, i, 0] = float(t)
            target_ctrl_points[0, i, 1] = float(0.5 + 0.25 * np.sin(t * np.pi))
            
        target_attn_masks[0, 10:22, 5:27] = 1.0
        target_render_masks[0, 40:88, 20:108] = 1.0
        target_eval_masks[0, 160:352, 80:432] = 1.0
        
        return {
            "image": img_tensor,
            "target_exists": target_exists,
            "target_ctrl_points": target_ctrl_points,
            "target_attn_masks": target_attn_masks,
            "target_render_masks": target_render_masks,
            "target_eval_masks": target_eval_masks,
            "img_path": "synthetic_smoke_test_frame.png"
        }
