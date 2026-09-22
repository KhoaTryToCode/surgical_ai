import os
import json
import numpy as np
import cv2

# The 11 Canonical Atlas Landmark Names and their canonical (u, v) coordinates:
# u in [0, 1]: 0.0=Left Tip, 0.5=Falciform, 1.0=Right Tip
# v in [0, 1]: 0.0=Anterior Ridge, 1.0=Posterior Silhouette
CANONICAL_ATLAS_NAMES = [
    "falc_root",    # 0: (0.50, 1.00) Superior Falciform root
    "falc_mid",     # 1: (0.50, 0.50) Falciform mid-body
    "falc_notch",   # 2: (0.50, 0.00) Umbilical notch at ridge
    "left_tip",     # 3: (0.00, 0.50) Left triangular ligament apex
    "left_sil",     # 4: (0.25, 1.00) Left silhouette margin mid
    "left_ridge",   # 5: (0.25, 0.00) Left ridge margin mid
    "left_body",    # 6: (0.25, 0.50) Left lobe parenchyma center
    "right_tip",    # 7: (1.00, 0.50) Right triangular ligament apex
    "right_sil",    # 8: (0.75, 1.00) Right silhouette margin mid
    "right_ridge",  # 9: (0.75, 0.00) Right ridge margin mid
    "right_body",   # 10:(0.75, 0.50) Right lobe parenchyma center
]

CANONICAL_UV = np.array([
    [0.50, 1.00],  # 0: falc_root
    [0.50, 0.50],  # 1: falc_mid
    [0.50, 0.00],  # 2: falc_notch
    [0.00, 0.50],  # 3: left_tip
    [0.25, 1.00],  # 4: left_sil
    [0.25, 0.00],  # 5: left_ridge
    [0.25, 0.50],  # 6: left_body
    [1.00, 0.50],  # 7: right_tip
    [0.75, 1.00],  # 8: right_sil
    [0.75, 0.00],  # 9: right_ridge
    [0.75, 0.50],  # 10: right_body
], dtype=np.float32)


def extract_atlas_points(data, orig_w, orig_h, target_size=1024):
    """
    Extracts the 11 canonical structural atlas points from L3D polygon annotations.
    
    Returns:
        coords: np.ndarray (11, 2) in normalized [0, 1]^2 canvas coordinates.
        visibilities: np.ndarray (11,) boolean/float flags (1.0 = visible, 0.0 = absent/occluded).
        has_falc: bool flag whether Falciform Ligament is physically annotated.
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
    
    ridge_all = np.vstack(ridge_pts) if has_r else np.zeros((0, 2), dtype=np.float32)
    sil_all = np.vstack(sil_pts) if has_s else np.zeros((0, 2), dtype=np.float32)
    falc_all = np.vstack(falc_pts) if has_f else np.zeros((0, 2), dtype=np.float32)
    
    coords = np.zeros((11, 2), dtype=np.float32)
    visibilities = np.zeros((11,), dtype=np.float32)
    
    # Canonical fallback centers (normalized to canvas [0, 1])
    coords[:, 0] = CANONICAL_UV[:, 0]
    coords[:, 1] = 1.0 - CANONICAL_UV[:, 1]  # flip v to image y (v=1 is top/low y, v=0 is bot/high y)

    # ── 1. FALCIFORM HINGE QUERIES (0, 1, 2) ──────────────────────────────────
    if has_f:
        # Sort along vertical axis (top of falciform has smaller y, bottom has larger y)
        f_order = np.argsort(falc_all[:, 1])
        f_sorted = falc_all[f_order]
        
        p_root = f_sorted[0]
        p_mid = f_sorted[len(f_sorted) // 2]
        p_notch = f_sorted[-1]
        
        coords[0] = p_root / float(target_size)
        coords[1] = p_mid / float(target_size)
        coords[2] = p_notch / float(target_size)
        visibilities[0:3] = 1.0
        falc_x = float(np.mean(falc_all[:, 0]))
    else:
        # If no falciform, estimate anatomical midline from available landmarks
        pts_list = []
        if has_r: pts_list.append(ridge_all)
        if has_s: pts_list.append(sil_all)
        if len(pts_list) > 0:
            comb = np.vstack(pts_list)
            falc_x = float(np.mean(comb[:, 0]))
        else:
            falc_x = target_size * 0.5
        visibilities[0:3] = 0.0  # Explicitly marked INVISIBLE

    # ── 2. LEFT LOBE QUERIES (3: left_tip, 4: left_sil, 5: left_ridge, 6: left_body)
    left_r = ridge_all[ridge_all[:, 0] <= falc_x] if has_r else np.zeros((0, 2))
    left_s = sil_all[sil_all[:, 0] <= falc_x] if has_s else np.zeros((0, 2))
    
    has_left_r = len(left_r) > 0
    has_left_s = len(left_s) > 0
    
    if has_left_r or has_left_s:
        # Left Apex (Tip)
        left_pts = np.vstack([p for p in [left_r, left_s] if len(p) > 0])
        left_tip_pt = left_pts[np.argmin(left_pts[:, 0])]
        coords[3] = left_tip_pt / float(target_size)
        visibilities[3] = 1.0
        
        # Left Silhouette
        if has_left_s:
            s_mid = left_s[len(left_s) // 2]
            coords[4] = s_mid / float(target_size)
            visibilities[4] = 1.0
            
        # Left Ridge
        if has_left_r:
            r_mid = left_r[len(left_r) // 2]
            coords[5] = r_mid / float(target_size)
            visibilities[5] = 1.0
            
        # Left Lobe Body (centroid of left lobe landmarks)
        coords[6] = np.mean(left_pts, axis=0) / float(target_size)
        visibilities[6] = 1.0

    # ── 3. RIGHT LOBE QUERIES (7: right_tip, 8: right_sil, 9: right_ridge, 10: right_body)
    right_r = ridge_all[ridge_all[:, 0] > falc_x] if has_r else np.zeros((0, 2))
    right_s = sil_all[sil_all[:, 0] > falc_x] if has_s else np.zeros((0, 2))
    
    has_right_r = len(right_r) > 0
    has_right_s = len(right_s) > 0
    
    if has_right_r or has_right_s:
        # Right Apex (Tip)
        right_pts = np.vstack([p for p in [right_r, right_s] if len(p) > 0])
        right_tip_pt = right_pts[np.argmax(right_pts[:, 0])]
        coords[7] = right_tip_pt / float(target_size)
        visibilities[7] = 1.0
        
        # Right Silhouette
        if has_right_s:
            s_mid = right_s[len(right_s) // 2]
            coords[8] = s_mid / float(target_size)
            visibilities[8] = 1.0
            
        # Right Ridge
        if has_right_r:
            r_mid = right_r[len(right_r) // 2]
            coords[9] = r_mid / float(target_size)
            visibilities[9] = 1.0
            
        # Right Lobe Body (centroid of right lobe landmarks)
        coords[10] = np.mean(right_pts, axis=0) / float(target_size)
        visibilities[10] = 1.0

    # Clip coordinates to valid [0, 1] range
    coords = np.clip(coords, 0.0, 1.0)
    return coords, visibilities, has_f
