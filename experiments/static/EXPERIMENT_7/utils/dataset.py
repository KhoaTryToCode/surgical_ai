import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.EXPERIMENT_7.utils.atlas_extractor import extract_atlas_points

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def generate_ruled_manifold_map(data, orig_w, orig_h, target_size=1024):
    """
    Generates continuous (u, v) manifold coordinates across the liver parenchyma.
    v in [0, 1]: 0.0 at Anterior Ridge, 1.0 at Posterior Silhouette.
    u in [0, 1]: 0.0 at Left Apex, 0.5 at Falciform Hinge, 1.0 at Right Apex.
    """
    sx = float(target_size) / float(orig_w)
    sy = float(target_size) / float(orig_h)
    shapes = data.get('shapes', [])
    
    ridge_pts, sil_pts, falc_pts = [], [], []
    for s in shapes:
        lbl = s.get('label', '').lower().strip()
        pts = s.get('points', [])
        if len(pts) < 2:
            continue
        pts_scaled = np.array([[p[0] * sx, p[1] * sy] for p in pts], dtype=np.float32)
        if lbl.startswith('r') or 'ridge' in lbl or 'rigde' in lbl or 'anterior' in lbl:
            ridge_pts.append(pts_scaled)
        elif lbl.startswith('s') or 'sil' in lbl or 'margin' in lbl or 'silhouette' in lbl:
            sil_pts.append(pts_scaled)
        elif lbl.startswith('f') or lbl.startswith('l') or 'falc' in lbl or 'lig' in lbl:
            falc_pts.append(pts_scaled)
            
    has_r = len(ridge_pts) > 0
    has_s = len(sil_pts) > 0
    has_f = len(falc_pts) > 0

    sil_all = np.vstack(sil_pts) if has_s else np.zeros((0, 2), dtype=np.float32)
    ridge_all = np.vstack(ridge_pts) if has_r else np.zeros((0, 2), dtype=np.float32)
    falc_all = np.vstack(falc_pts) if has_f else np.zeros((0, 2), dtype=np.float32)

    pts_for_x = []
    if has_s: pts_for_x.append(sil_all)
    if has_r: pts_for_x.append(ridge_all)
    if has_f: pts_for_x.append(falc_all)
    
    if len(pts_for_x) == 0:
        return np.zeros((2, target_size, target_size), dtype=np.float32), np.zeros((target_size, target_size), dtype=np.uint8), False
        
    all_combined = np.vstack(pts_for_x)
    x_min = max(0, int(np.floor(all_combined[:, 0].min())))
    x_max = min(target_size - 1, int(np.ceil(all_combined[:, 0].max())))
    xs = np.arange(x_min, x_max + 1)

    # 1. Silhouette profile
    if has_s:
        s_order = np.argsort(sil_all[:, 0])
        s_x, s_y = sil_all[s_order, 0], sil_all[s_order, 1]
        _, u_idx = np.unique(s_x, return_index=True)
        y_sil_interp = np.interp(xs, s_x[u_idx], s_y[u_idx])
    else:
        r_order = np.argsort(ridge_all[:, 0])
        r_x, r_y = ridge_all[r_order, 0], ridge_all[r_order, 1]
        _, u_idx = np.unique(r_x, return_index=True)
        y_ridge_temp = np.interp(xs, r_x[u_idx], r_y[u_idx])
        y_sil_interp = np.clip(y_ridge_temp - 120.0, 0, target_size - 1)

    # 2. Ridge profile
    if has_r:
        r_order = np.argsort(ridge_all[:, 0])
        r_x, r_y = ridge_all[r_order, 0], ridge_all[r_order, 1]
        _, u_idx = np.unique(r_x, return_index=True)
        y_ridge_interp = np.interp(xs, r_x[u_idx], r_y[u_idx])
    else:
        y_ridge_interp = np.clip(y_sil_interp + 120.0, 0, target_size - 1)

    # 3. Falciform position
    if has_f:
        falc_x_mid = float(np.mean(falc_all[:, 0]))
    else:
        falc_x_mid = float(x_min + x_max) / 2.0

    # Chirality detection
    mean_y_sil = float(np.mean(y_sil_interp))
    mean_y_ridge = float(np.mean(y_ridge_interp))
    is_flipped = bool(mean_y_ridge < mean_y_sil - 15.0)

    u_coord = np.zeros((target_size, target_size), dtype=np.float32)
    v_coord = np.zeros((target_size, target_size), dtype=np.float32)
    mask_liver = np.zeros((target_size, target_size), dtype=np.uint8)

    for x_val, y_s, y_r in zip(xs, y_sil_interp, y_ridge_interp):
        y_top = int(np.round(min(y_s, y_r)))
        y_bot = int(np.round(max(y_s, y_r)))
        if y_bot <= y_top:
            continue
        y_top_c = max(0, y_top)
        y_bot_c = min(target_size, y_bot)

        mask_liver[y_top_c:y_bot_c, x_val] = 1
        ys = np.arange(y_top_c, y_bot_c)
        denom = max(float(y_bot - y_top), 1.0)
        
        # v: 0.0 at Ridge, 1.0 at Silhouette
        if is_flipped:
            v_coord[y_top_c:y_bot_c, x_val] = (ys - y_top) / denom
        else:
            v_coord[y_top_c:y_bot_c, x_val] = (y_bot - ys) / denom

        # u: 0.0 at x_min, 0.5 at falc_x_mid, 1.0 at x_max
        if x_val <= falc_x_mid:
            u_val = 0.5 * (x_val - x_min) / max(falc_x_mid - x_min, 1.0)
        else:
            u_val = 0.5 + 0.5 * (x_val - falc_x_mid) / max(x_max - falc_x_mid, 1.0)
        u_coord[y_top_c:y_bot_c, x_val] = np.clip(u_val, 0.0, 1.0)

    u_coord[mask_liver == 0] = 0.0
    v_coord[mask_liver == 0] = 0.0
    
    uv_map = np.stack([u_coord, v_coord], axis=0)  # (2, target_size, target_size)
    return uv_map, mask_liver, has_f


class L3DManifoldDataset(Dataset):
    """
    L3D Dataset for EXPERIMENT_7 (Manifold-Steered Mask2Former).
    
    Guarantees 100% metric parity by drawing polylines on native canvas with
    thickness 35, followed by cv2.INTER_NEAREST resize to 1024x1024 (W_eff ~ 18.7 px).
    """
    def __init__(self, data_dir=None, split="Train", target_size=1024, is_train=True):
        self.data_dir = resolve_l3d_root(data_dir)
        self.split = split
        self.target_size = target_size
        self.is_train = is_train

        split_dir = os.path.join(self.data_dir, split)
        self.img_dir = os.path.join(split_dir, "images")
        self.lbl_dir = os.path.join(split_dir, "labels")

        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        if len(self.img_files) == 0:
            self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))

        print(f"[{split}] Loaded {len(self.img_files)} images from: {self.img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(self.lbl_dir, f"{stem}.json")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        orig_h, orig_w = img_bgr.shape[:2]

        with open(json_path, "r") as f:
            data = json.load(f)

        # ── 1. RASTERIZE GROUND TRUTH (STRICT TOPONET/EXP1/2/5 PARITY) ────────
        # Draw on native canvas with thickness 35, then downsample to 1024x1024
        native_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for s in data.get("shapes", []):
            lbl = s.get("label", "").lower().strip()
            pts = np.array(s.get("points", []), dtype=np.int32)
            if len(pts) < 2:
                continue

            if lbl.startswith("r") or "ridge" in lbl or "rigde" in lbl or "anterior" in lbl:
                cls_id = 1  # Ridge
            elif lbl.startswith("s") or "sil" in lbl or "margin" in lbl or "silhouette" in lbl:
                cls_id = 2  # Silhouette
            elif lbl.startswith("f") or lbl.startswith("l") or "falc" in lbl or "lig" in lbl:
                cls_id = 3  # Falciform
            else:
                continue

            cv2.polylines(native_mask, [pts], isClosed=False, color=cls_id, thickness=35)

        # Resize to 1024x1024 using INTER_NEAREST to prevent class blending
        mask = cv2.resize(native_mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        # ── 2. PREPARE RGB INPUT TENSOR ───────────────────────────────────────
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        # Normalize with ImageNet mean and std
        img_norm = (img_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        # ── 3. EXTRACT 11-QUERY STRUCTURAL TARGETS ────────────────────────────
        coords, visibilities, has_falc = extract_atlas_points(data, orig_w, orig_h, target_size=self.target_size)
        coords_tensor = torch.from_numpy(coords).float()              # (11, 2)
        vis_tensor = torch.from_numpy(visibilities).float()          # (11,)

        # ── 4. EXTRACT CONTINUOUS MANIFOLD (U, V) MAP ─────────────────────────
        uv_map, mask_liver, _ = generate_ruled_manifold_map(data, orig_w, orig_h, target_size=self.target_size)
        uv_tensor = torch.from_numpy(uv_map).float()                 # (2, 1024, 1024)
        liver_mask_tensor = torch.from_numpy(mask_liver).float()     # (1024, 1024)

        return {
            "image": img_tensor,
            "mask": torch.from_numpy(mask).long(),
            "coords": coords_tensor,
            "visibilities": vis_tensor,
            "uv_map": uv_tensor,
            "liver_mask": liver_mask_tensor,
            "has_falc": torch.tensor(1.0 if has_falc else 0.0, dtype=torch.float32),
            "stem": stem,
            "orig_size": (orig_w, orig_h)
        }
