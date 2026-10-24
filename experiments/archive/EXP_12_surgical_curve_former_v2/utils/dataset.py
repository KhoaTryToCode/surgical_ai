"""
EXP_12: Dataset for SurgicalCurveFormer v2
==========================================
Self-contained dataset module for EXP_12 with:
1. 5th-order Bézier GT fitting (K=6 ctrl pts, degree=5)
2. ACPI target score map: GT midpoint indicator on f4 grid (H/32 x W/32)
3. 3-class output: Ridge(0), Silhouette(1), Ligament(2)
4. Render-size GT mask (128x128) for soft rasterizer Dice
"""
import os
import json
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import sys

exp12_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if exp12_root not in sys.path:
    sys.path.insert(0, exp12_root)

from models.bezier_ops import fit_bezier_least_squares, evaluate_bezier_torch


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


class SurgicalCurveFormerDataset(Dataset):
    """
    Dataset for SurgicalCurveFormer v2.
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
        acpi_stride: int = 32,
        render_size: int = 128,
        bezier_degree: int = 5,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.image_size = image_size
        self.spline_step_px = spline_step_px
        self.stroke_thickness = stroke_thickness
        self.use_depth = use_depth
        self.acpi_h = image_size // acpi_stride
        self.acpi_w = image_size // acpi_stride
        self.render_size = render_size
        self.bezier_degree = bezier_degree

        split_dir = os.path.join(dataset_dir, mode)
        if not os.path.exists(split_dir):
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
                glob.glob(os.path.join(split_dir, "**", "depth", "*.png"), recursive=True) +
                glob.glob(os.path.join(dataset_dir, "**", "depth_anything_v2", "*.png"), recursive=True)
            )
            for df in depth_globs:
                self.depth_index[os.path.splitext(os.path.basename(df))[0]] = df

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

        print(
            f"[EXP_12 Dataset] {len(self.image_paths)} images | mode={mode} | "
            f"RGB-D={use_depth} | ACPI grid={self.acpi_h}x{self.acpi_w} | "
            f"Bézier degree={bezier_degree} (K={bezier_degree+1} ctrl pts)"
        )

    def __len__(self) -> int:
        return max(len(self.image_paths), 1)

    def _find_json(self, img_path: str):
        name = os.path.splitext(os.path.basename(img_path))[0]
        if name in self.json_index:
            return self.json_index[name]
        for cand in [
            img_path.replace("images", "labels").replace(".jpg", ".json").replace(".png", ".json"),
            os.path.join(os.path.dirname(img_path), f"{name}.json"),
            os.path.join(self.split_dir, "labels", f"{name}.json"),
        ]:
            if os.path.exists(cand):
                return cand
        return None

    def _find_depth(self, img_path: str):
        name = os.path.splitext(os.path.basename(img_path))[0]
        if name in self.depth_index:
            return self.depth_index[name]
        for cand in [
            img_path.replace("images", "depth_anything_v2").replace(".jpg", ".png"),
            img_path.replace("images", "depth").replace(".jpg", ".png"),
            os.path.join(os.path.dirname(os.path.dirname(img_path)),
                         "depth_anything_v2", f"{name}.png"),
        ]:
            if os.path.exists(cand):
                return cand
        return None

    def _label_to_class(self, label: str) -> int:
        label = label.lower().strip().replace("-", "_").replace(" ", "_")
        if "ridge" in label or label.startswith("r"):
            return 0
        if "silhouette" in label or label.startswith("s"):
            return 1
        if "ligament" in label or "falciform" in label or "vessel" in label or label.startswith("l"):
            return 2
        return self.CLASS_MAP.get(label, -1)

    def __getitem__(self, idx: int) -> dict:
        S = self.image_size
        M = self.NUM_CLASSES
        R = self.render_size

        if len(self.image_paths) == 0:
            return self._synthetic_sample()

        img_path = self.image_paths[idx]

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            img_rgb = np.zeros((3, S, S), dtype=np.float32)
            orig_h, orig_w = S, S
        else:
            img_rgb_hw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img_rgb_hw.shape[:2]
            if img_rgb_hw.shape[:2] != (S, S):
                img_rgb_hw = cv2.resize(img_rgb_hw, (S, S), interpolation=cv2.INTER_LINEAR)
            img_rgb = (img_rgb_hw.astype(np.float32) / 255.0).transpose(2, 0, 1)

        img_norm = (img_rgb - self.mean) / self.std
        img_tensor = torch.from_numpy(img_norm).float()

        if self.use_depth:
            depth_path = self._find_depth(img_path)
            if depth_path is not None:
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
            img_tensor = torch.cat([img_tensor, depth_tensor], dim=0)

        target_masks   = np.zeros((M, S, S),         dtype=np.float32)
        target_ctrl    = np.zeros((M, self.bezier_degree + 1, 2), dtype=np.float32)
        active_mask    = np.zeros((M,),              dtype=bool)
        acpi_scores    = np.zeros((M, self.acpi_h, self.acpi_w), dtype=np.float32)
        render_masks   = np.zeros((M, R, R),         dtype=np.float32)

        ann_path = self._find_json(img_path)
        if ann_path is not None:
            try:
                with open(ann_path, "r") as f:
                    data = json.load(f)
                for shape in data.get("shapes", []):
                    label = str(shape.get("label", ""))
                    cls_id = self._label_to_class(label)
                    if cls_id < 0:
                        continue

                    raw_pts = np.array(shape.get("points", []), dtype=np.float32)
                    if len(raw_pts) < 2:
                        continue

                    pts = raw_pts.copy()
                    pts[:, 0] *= S / float(orig_w)
                    pts[:, 1] *= S / float(orig_h)

                    dense = resample_polyline_by_arclength(pts, self.spline_step_px)
                    if len(dense) < 2:
                        continue

                    pts_pix = np.clip(np.round(dense), 0, S - 1).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(
                        target_masks[cls_id],
                        [pts_pix],
                        isClosed=False, color=1.0,
                        thickness=self.stroke_thickness, lineType=cv2.LINE_AA
                    )

                    dense_norm = dense.copy()
                    dense_norm[:, 0] /= float(S)
                    dense_norm[:, 1] /= float(S)
                    dense_norm = np.clip(dense_norm, 0.0, 1.0)

                    ctrl = fit_bezier_least_squares(dense_norm, degree=self.bezier_degree)
                    target_ctrl[cls_id] = ctrl
                    active_mask[cls_id] = True

                    mid_norm = dense_norm[len(dense_norm) // 2]
                    mid_acpi_x = int(np.clip(mid_norm[0] * self.acpi_w, 0, self.acpi_w - 1))
                    mid_acpi_y = int(np.clip(mid_norm[1] * self.acpi_h, 0, self.acpi_h - 1))
                    acpi_scores[cls_id, mid_acpi_y, mid_acpi_x] = 1.0

                    scale = float(R) / float(S)
                    dense_r = dense.copy()
                    dense_r[:, 0] *= scale
                    dense_r[:, 1] *= scale
                    pts_r = np.clip(np.round(dense_r), 0, R - 1).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(
                        render_masks[cls_id],
                        [pts_r],
                        isClosed=False, color=1.0,
                        thickness=max(1, int(self.stroke_thickness * scale)),
                        lineType=cv2.LINE_AA
                    )
            except Exception:
                pass

        return {
            "image":               img_tensor,
            "target_masks":        torch.from_numpy(target_masks).float(),
            "target_ctrl_pts":     torch.from_numpy(target_ctrl).float(),
            "active_mask":         torch.from_numpy(active_mask).bool(),
            "acpi_target_score":   torch.from_numpy(acpi_scores).float(),
            "target_render_masks": torch.from_numpy(render_masks).float(),
            "img_path":            img_path,
        }

    def _synthetic_sample(self) -> dict:
        S = self.image_size
        M = self.NUM_CLASSES
        R = self.render_size
        K = self.bezier_degree + 1

        img = torch.zeros(4, S, S, dtype=torch.float32)
        target_masks   = torch.zeros(M, S, S,            dtype=torch.float32)
        target_ctrl    = torch.zeros(M, K, 2,            dtype=torch.float32)
        active_mask    = torch.zeros(M,                   dtype=torch.bool)
        acpi_scores    = torch.zeros(M, self.acpi_h, self.acpi_w, dtype=torch.float32)
        render_masks   = torch.zeros(M, R, R,            dtype=torch.float32)

        ctrl_ridge = np.array([
            [0.1, 0.2], [0.25, 0.25], [0.4, 0.3],
            [0.6, 0.3], [0.75, 0.25], [0.9, 0.2]
        ], dtype=np.float32)
        target_ctrl[0] = torch.from_numpy(ctrl_ridge)
        active_mask[0] = True
        acpi_scores[0, self.acpi_h // 4, self.acpi_w // 2] = 1.0

        pts_px = (ctrl_ridge * S).astype(np.int32)
        for i in range(len(pts_px) - 1):
            cv2.line(
                target_masks[0].numpy(),
                tuple(pts_px[i]), tuple(pts_px[i+1]),
                1.0, 2
            )

        return {
            "image":               img,
            "target_masks":        target_masks,
            "target_ctrl_pts":     target_ctrl,
            "active_mask":         active_mask,
            "acpi_target_score":   acpi_scores,
            "target_render_masks": render_masks,
            "img_path":            "synthetic_smoke_test.png",
        }
