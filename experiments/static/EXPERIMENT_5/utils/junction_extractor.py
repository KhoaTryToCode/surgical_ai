"""
Deterministic Anatomical Junction Extractor for L3D Dataset.
Extracts the 4 biological landmark junctions from JSON polyline annotations:
  1. J_top (idx 0): Falciform-Silhouette Root (Superior liver boundary)
  2. J_bottom (idx 1): Umbilical Notch (Inferior Falciform meets Ridge)
  3. J_lat_right (idx 2): Right Lateral Tip (Right Ridge meets Right Silhouette)
  4. J_lat_left (idx 3): Left Lateral Tip (Left Ridge meets Left Silhouette)
"""
import numpy as np

JUNCTION_NAMES = ['J_top', 'J_bottom', 'J_lat_right', 'J_lat_left']

def extract_gt_junctions(data, orig_w, orig_h, canvas_size=1024, threshold_px=100.0):
    """
    Extracts normalized coordinates (x, y) in [0, 1]^2 and binary visibility flags (4,)
    for the 4 anatomical junction keypoints from an L3D JSON annotation dict.

    Args:
        data (dict): JSON contents with 'shapes' list.
        orig_w (int): Original image width (handles 4K Patient 32).
        orig_h (int): Original image height.
        canvas_size (int): Working resolution (default 1024).
        threshold_px (float): Distance threshold in canvas pixels to consider curves connected.

    Returns:
        coords_norm (np.ndarray): Shape (4, 2), float32, normalized to [0, 1]^2.
        visibility (np.ndarray): Shape (4,), float32, in {0.0, 1.0}.
        coords_px (np.ndarray): Shape (4, 2), float32, in [0, canvas_size] pixels.
    """
    sx = float(canvas_size) / float(orig_w)
    sy = float(canvas_size) / float(orig_h)

    r_curves, s_curves, f_curves = [], [], []
    for shape in data.get('shapes', []):
        lbl = str(shape.get('label', '')).lower().strip()
        pts = np.array(shape.get('points', []), dtype=np.float32)
        if len(pts) < 2:
            continue
        # Scale to canvas coordinates
        scaled = np.column_stack([pts[:, 0] * sx, pts[:, 1] * sy])
        if lbl.startswith('r') or 'ridge' in lbl or 'rigde' in lbl:
            r_curves.append(scaled)
        elif lbl.startswith('s') or 'sil' in lbl:
            s_curves.append(scaled)
        elif lbl.startswith('f') or 'falc' in lbl or 'lig' in lbl:
            f_curves.append(scaled)

    coords_px = np.zeros((4, 2), dtype=np.float32)
    visibility = np.zeros(4, dtype=np.float32)

    # 1. J_top (Falciform meets Silhouette at superior root)
    best_d = 1e9
    j_top = None
    for fc in f_curves:
        for sc in s_curves:
            for f_pt in [fc[0], fc[-1]]:
                dists = np.linalg.norm(sc - f_pt, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < best_d:
                    best_d = dists[min_idx]
                    j_top = (f_pt + sc[min_idx]) / 2.0
    if best_d <= threshold_px and j_top is not None:
        coords_px[0] = j_top
        visibility[0] = 1.0

    # 2. J_bottom (Falciform meets Ridge at Umbilical Notch)
    best_d = 1e9
    j_bot = None
    for fc in f_curves:
        for rc in r_curves:
            for f_pt in [fc[0], fc[-1]]:
                dists = np.linalg.norm(rc - f_pt, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < best_d:
                    best_d = dists[min_idx]
                    j_bot = (f_pt + rc[min_idx]) / 2.0
    if best_d <= threshold_px and j_bot is not None:
        coords_px[1] = j_bot
        visibility[1] = 1.0

    # 3. J_lat_right & J_lat_left (Ridge meets Silhouette at both extremities)
    best_dr, pt_r = 1e9, None
    best_dl, pt_l = 1e9, None

    for rc in r_curves:
        for sc in s_curves:
            # Check ridge endpoints to silhouette curve
            for r_pt in [rc[0], rc[-1]]:
                d_s = np.linalg.norm(sc - r_pt, axis=1)
                idx = np.argmin(d_s)
                d = d_s[idx]
                pt = (r_pt + sc[idx]) / 2.0
                if pt[0] >= canvas_size * 0.45 and d < best_dr:
                    best_dr, pt_r = d, pt
                elif pt[0] < canvas_size * 0.45 and d < best_dl:
                    best_dl, pt_l = d, pt
            # Check silhouette endpoints to ridge curve
            for s_pt in [sc[0], sc[-1]]:
                d_r = np.linalg.norm(rc - s_pt, axis=1)
                idx = np.argmin(d_r)
                d = d_r[idx]
                pt = (s_pt + rc[idx]) / 2.0
                if pt[0] >= canvas_size * 0.45 and d < best_dr:
                    best_dr, pt_r = d, pt
                elif pt[0] < canvas_size * 0.45 and d < best_dl:
                    best_dl, pt_l = d, pt

    if best_dr <= threshold_px and pt_r is not None:
        coords_px[2] = pt_r
        visibility[2] = 1.0

    if best_dl <= threshold_px and pt_l is not None:
        coords_px[3] = pt_l
        visibility[3] = 1.0

    # Normalize coordinates to [0, 1]^2
    coords_norm = coords_px / float(canvas_size)

    return coords_norm, visibility, coords_px
