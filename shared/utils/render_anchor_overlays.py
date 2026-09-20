import os
import sys
import json
import glob
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

# Add workspace root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from experiments.EXPERIMENT_6.utils.junction_extractor import extract_gt_junctions

# Color scheme (BGR)
COLOR_RIDGE_BGR = (0, 0, 255)       # Red
COLOR_SIL_BGR = (0, 255, 0)         # Green
COLOR_FALC_BGR = (255, 120, 0)      # Blue

JUNCTION_COLORS = [
    (0, 0, 255),    # J_top: Red
    (0, 255, 255),  # J_bot: Yellow
    (255, 0, 255),  # J_lat_r: Magenta
    (255, 255, 0)   # J_lat_l: Cyan
]


def point_to_polyline_dist(pt, polyline):
    """Computes exact minimum distance from point pt to polyline segments."""
    min_d = 1e9
    for i in range(len(polyline) - 1):
        p1 = polyline[i]
        p2 = polyline[i + 1]
        seg = p2 - p1
        seg_len2 = np.dot(seg, seg)
        if seg_len2 < 1e-4:
            d = np.linalg.norm(pt - p1)
        else:
            t = np.clip(np.dot(pt - p1, seg) / seg_len2, 0.0, 1.0)
            proj = p1 + t * seg
            d = np.linalg.norm(pt - proj)
        if d < min_d:
            min_d = d
    return min_d


def verify_ray_along_curves(pt, vec, curves, initial_offset=0.0, test_steps=(25.0, 50.0)):
    """
    Ray-marches along pt + s * vec to verify that the vector actually tracks the annotated curve.
    Uses adaptive tolerance: since the junction point can be offset up to 70px from a curve endpoint,
    the allowed ray distance is scaled to max(initial_offset + 18.0, 48.0 px) to prevent rejecting
    valid curves that meet with a small gap.
    """
    if not curves:
        return False, 1e9
    allowed_dist = max(float(initial_offset) + 18.0, 48.0)
    dists = []
    for s in test_steps:
        q = pt + s * vec
        d_min = min(point_to_polyline_dist(q, c) for c in curves)
        dists.append(d_min)
    mean_d = float(np.mean(dists))
    return (mean_d <= allowed_dist), mean_d


def extract_3way_tangents_denoised(pt, stalk_curves, cross_curves, is_bottom=False, search_dist=140.0):
    """
    Extracts physically verified, denoised tangents for 3-way tripods (J_top, J_bottom).
    Applies:
      1. Single-endpoint selection (never picks both ends of short curves).
      2. Adaptive ray-march verification (discards vectors shooting into empty space while tolerating gap offsets).
      3. Anatomical orientation bounds (ridge cannot point upwards into parenchyma).
      4. Angular deduplication (collapses parallel duplicate vectors).
    """
    # 1. Stalk Tangent
    t_stalk = None
    best_d = 1e9
    for c in stalk_curves:
        d0 = np.linalg.norm(c[0] - pt)
        d1 = np.linalg.norm(c[-1] - pt)
        offset = min(d0, d1)
        if offset < search_dist and offset < best_d:
            best_d = offset
            if d0 <= d1:
                idx = min(len(c) - 1, 4)
                v = c[idx] - c[0]
            else:
                idx = max(0, len(c) - 5)
                v = c[idx] - c[-1]
            if np.linalg.norm(v) > 1e-2:
                v_norm = v / np.linalg.norm(v)
                # Check anatomical direction
                if is_bottom and v_norm[1] < 0.2:  # Notch stalk must point up towards root
                    is_valid, _ = verify_ray_along_curves(pt, v_norm, [c], initial_offset=offset)
                    if is_valid:
                        t_stalk = v_norm
                elif not is_bottom and v_norm[1] > -0.2:  # Root stalk must point down towards notch
                    is_valid, _ = verify_ray_along_curves(pt, v_norm, [c], initial_offset=offset)
                    if is_valid:
                        t_stalk = v_norm

    # 2. Cross Curves (Ridge for bottom, Silhouette for top)
    candidate_tangents = []
    for c in cross_curves:
        d0 = np.linalg.norm(c[0] - pt)
        d1 = np.linalg.norm(c[-1] - pt)
        offset = min(d0, d1)
        # CRITICAL FIX: Strictly choose ONLY the single closest endpoint
        if offset < search_dist:
            if d0 <= d1:
                idx = min(len(c) - 1, 4)
                v = c[idx] - c[0]
            else:
                idx = max(0, len(c) - 5)
                v = c[idx] - c[-1]
            if np.linalg.norm(v) > 1e-2:
                v_norm = v / np.linalg.norm(v)

                # ANATOMICAL GATE: At J_bottom, ridge arms CANNOT point straight up into parenchyma
                if is_bottom and v_norm[1] < -0.15:
                    continue  # Purge spurious upward vector

                # ADAPTIVE RAY-MARCH VERIFICATION
                is_valid, _ = verify_ray_along_curves(pt, v_norm, [c], initial_offset=offset)
                if is_valid:
                    candidate_tangents.append(v_norm)

    # 3. Pass-through detection if curve wasn't split at junction
    if len(candidate_tangents) < 2:
        for c in cross_curves:
            dists = np.linalg.norm(c - pt, axis=1)
            min_i = np.argmin(dists)
            if dists[min_i] < 60.0 and 3 <= min_i <= len(c) - 4:
                v_fwd = c[min(len(c) - 1, min_i + 4)] - c[min_i]
                v_bwd = c[max(0, min_i - 4)] - c[min_i]
                v_fwd = v_fwd / (np.linalg.norm(v_fwd) + 1e-6)
                v_bwd = v_bwd / (np.linalg.norm(v_bwd) + 1e-6)
                for v_cand in [v_fwd, v_bwd]:
                    if is_bottom and v_cand[1] < -0.15:
                        continue
                    is_valid, _ = verify_ray_along_curves(pt, v_cand, [c], initial_offset=dists[min_i])
                    if is_valid:
                        candidate_tangents.append(v_cand)
                break

    # 4. Deduplicate parallel tangents (collapse if cos > 0.85 / angle < 30 deg)
    unique_cross = []
    for ct in candidate_tangents:
        is_dup = False
        for u in unique_cross:
            if np.dot(ct, u) > 0.85:
                is_dup = True
                break
        if not is_dup:
            unique_cross.append(ct)

    tangents = []
    if t_stalk is not None:
        tangents.append(('Falciform Stalk', t_stalk, (255, 200, 0)))  # Cyan
    for ct in unique_cross:
        name = 'Right Arm' if ct[0] >= 0 else 'Left Arm'
        col = (255, 255, 255) if ct[0] >= 0 else (0, 255, 255)
        tangents.append((name, ct, col))

    return tangents, None, False


def extract_2way_corner_denoised(pt, r_curves, s_curves, is_left_tip=False, search_dist=140.0):
    """
    Extracts physically verified tangents and opening angle for 2-way tips (J_lat_right, J_lat_left).
    Ray-marches both candidate arms to ensure they track actual boundary contours.
    """
    def get_verified_tangent(curves):
        best_d = 1e9
        best_tangent = None
        for c in curves:
            d0 = np.linalg.norm(c[0] - pt)
            d1 = np.linalg.norm(c[-1] - pt)
            offset = min(d0, d1)
            if offset < search_dist and offset < best_d:
                best_d = offset
                if d0 <= d1:
                    idx = min(len(c) - 1, 4)
                    vec = c[idx] - c[0]
                else:
                    idx = max(0, len(c) - 5)
                    vec = c[idx] - c[-1]
                if np.linalg.norm(vec) > 1e-3:
                    v_norm = vec / np.linalg.norm(vec)
                    is_valid, _ = verify_ray_along_curves(pt, v_norm, [c], initial_offset=offset)
                    if is_valid:
                        best_tangent = v_norm
        return best_tangent

    t_ridge = get_verified_tangent(r_curves)
    t_sil = get_verified_tangent(s_curves)

    if t_ridge is None or t_sil is None:
        tangents = []
        if t_ridge is not None:
            tangents.append(('Ridge Arm', t_ridge, (0, 255, 255)))
        if t_sil is not None:
            tangents.append(('Silhouette Arm', t_sil, (255, 255, 255)))
        return tangents, None, False

    # Canonical chirality: right tip has cross > 0 in normal anatomy, left tip has cross < 0
    cross_prod = t_ridge[0] * t_sil[1] - t_ridge[1] * t_sil[0]
    if is_left_tip:
        cross_prod = -cross_prod

    cos_theta = np.clip(np.dot(t_ridge, t_sil), -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_theta)))
    is_flipped = bool(cross_prod < 0)

    tangents = [
        ('Ridge Arm', t_ridge, (0, 255, 255)),        # Yellow
        ('Silhouette Arm', t_sil, (255, 255, 255))    # White
    ]
    return tangents, angle_deg, is_flipped


def process_single_frame(args):
    img_path, json_path, out_dir = args
    stem = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, f"{stem}.jpg")

    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (1024, 1024))
    canvas = img.copy()

    with open(json_path, 'r') as f:
        data = json.load(f)
    orig_w = data.get('imageWidth', 1920)
    orig_h = data.get('imageHeight', 1080)
    sx = 1024.0 / float(orig_w)
    sy = 1024.0 / float(orig_h)

    r_curves, s_curves, f_curves = [], [], []
    for shape in data.get('shapes', []):
        lbl = str(shape.get('label', '')).lower().strip()
        pts = np.array(shape.get('points', []), dtype=np.float32)
        if len(pts) < 2:
            continue
        scaled = np.column_stack([pts[:, 0] * sx, pts[:, 1] * sy])
        if lbl.startswith('r') or 'ridge' in lbl or 'rigde' in lbl:
            r_curves.append(scaled)
        elif lbl.startswith('s') or 'sil' in lbl:
            s_curves.append(scaled)
        elif lbl.startswith('f') or 'falc' in lbl or 'lig' in lbl:
            f_curves.append(scaled)

    # Use gap-bridging threshold = 140.0 px to recover anchors missed by human gaps
    coords_norm, visibility, coords_px = extract_gt_junctions(data, orig_w, orig_h, canvas_size=1024, threshold_px=140.0)

    # 1. Draw boundary curves
    overlay = canvas.copy()
    for c in r_curves:
        cv2.polylines(overlay, [c.astype(np.int32)], False, COLOR_RIDGE_BGR, 8, cv2.LINE_AA)
    for c in s_curves:
        cv2.polylines(overlay, [c.astype(np.int32)], False, COLOR_SIL_BGR, 8, cv2.LINE_AA)
    for c in f_curves:
        cv2.polylines(overlay, [c.astype(np.int32)], False, COLOR_FALC_BGR, 8, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

    arrow_len = 110
    meta = {
        'stem': stem,
        'patient': stem.rsplit('_', 1)[0] if '_' in stem else stem,
        'frame': stem.rsplit('_', 1)[1] if '_' in stem else '',
        'num_visible': int(visibility.sum()),
        'junctions': {},
        'is_flipped': False
    }

    junction_defs = [
        ('J_top', coords_px[0], visibility[0], 0, lambda pt: extract_3way_tangents_denoised(pt, f_curves, s_curves, is_bottom=False)),
        ('J_bottom', coords_px[1], visibility[1], 1, lambda pt: extract_3way_tangents_denoised(pt, f_curves, r_curves, is_bottom=True)),
        ('J_lat_right', coords_px[2], visibility[2], 2, lambda pt: extract_2way_corner_denoised(pt, r_curves, s_curves, is_left_tip=False)),
        ('J_lat_left', coords_px[3], visibility[3], 3, lambda pt: extract_2way_corner_denoised(pt, r_curves, s_curves, is_left_tip=True))
    ]

    for name, pt, vis, k_idx, extractor in junction_defs:
        if vis < 0.5:
            meta['junctions'][name] = {'visible': False}
            continue

        px, py = int(pt[0]), int(pt[1])
        tangents, angle_deg, is_flipped = extractor(pt)

        is_flipped = bool(is_flipped)
        if is_flipped:
            meta['is_flipped'] = True

        j_info = {
            'visible': True,
            'coords_px': [int(px), int(py)],
            'num_branches': int(len(tangents)),
            'angle_deg': round(float(angle_deg), 1) if angle_deg is not None else None,
            'is_flipped': bool(is_flipped)
        }
        meta['junctions'][name] = j_info

        # Draw directional arrows
        for arm_name, t_vec, col in tangents:
            p_end = (int(px + t_vec[0] * arrow_len), int(py + t_vec[1] * arrow_len))
            cv2.arrowedLine(canvas, (px, py), p_end, col, 4, tipLength=0.22, line_type=cv2.LINE_AA)

        # Draw arc if 2 arms
        if len(tangents) == 2 and angle_deg is not None:
            t1 = tangents[0][1]
            t2 = tangents[1][1]
            start_angle = np.degrees(np.arctan2(t1[1], t1[0]))
            end_angle = np.degrees(np.arctan2(t2[1], t2[0]))
            cv2.ellipse(canvas, (px, py), (45, 45), 0, start_angle, end_angle, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw junction center dot
        cv2.circle(canvas, (px, py), 16, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 12, JUNCTION_COLORS[k_idx], -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)

        # Build label text
        if angle_deg is not None:
            status_str = "FLIPPED" if is_flipped else "NORMAL"
            badge_col = (0, 140, 255) if is_flipped else (0, 255, 100)
            label_txt = f"{name}: {angle_deg:.1f}deg [{status_str}]"
        else:
            branch_str = f"{len(tangents)}-Way"
            label_txt = f"{name}: {branch_str}"
            badge_col = (0, 255, 100)

        # Smart label offsets
        if k_idx == 0:
            txt_pos = (max(10, px - 160), max(30, py - 20))
        elif k_idx == 1:
            txt_pos = (max(10, px - 60), min(1010, py + 40))
        elif k_idx == 2:
            txt_pos = (max(10, px - 280), max(30, py - 20))
        else:
            txt_pos = (max(10, px - 20), min(1010, py + 40))

        cv2.putText(canvas, label_txt, txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label_txt, txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.58, badge_col, 2, cv2.LINE_AA)

    # Header HUD
    n_pts = int(visibility.sum())
    flip_tag = " [FLIPPED RETRACTION DETECTED]" if meta['is_flipped'] else ""
    tag_col = (0, 140, 255) if meta['is_flipped'] else (255, 255, 255)
    cv2.rectangle(canvas, (0, 0), (1024, 70), (20, 20, 20), -1)
    title_str = f"{stem} | Visible: {n_pts}/4{flip_tag}"
    cv2.putText(canvas, title_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tag_col, 2, cv2.LINE_AA)
    sub_str = "Red=J_top (Root) | Ylw=J_bot (Notch) | Mag=J_lat_r (Tip) | Cyan=J_lat_l (Tip) | Arrows=Denoised Tangents"
    cv2.putText(canvas, sub_str, (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 210, 240), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return meta


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    train_img_dir = os.path.join(root_dir, 'data/L3D/Train/images')
    train_lbl_dir = os.path.join(root_dir, 'data/L3D/Train/labels')
    out_dir = os.path.join(root_dir, 'data/inspections/train_anchor_overlays')
    os.makedirs(out_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(train_img_dir, '*.jpg')))
    print(f"Found {len(img_files)} training images.")

    tasks = []
    for img_p in img_files:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        json_p = os.path.join(train_lbl_dir, f"{stem}.json")
        if os.path.exists(json_p):
            tasks.append((img_p, json_p, out_dir))

    print(f"Rendering {len(tasks)} overlays with 8 threads (Denoised)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(process_single_frame, tasks))
    dt = time.time() - t0
    print(f"Rendered {len(results)} denoised images in {dt:.1f}s ({len(results)/dt:.1f} fps)!")

    # Save metadata manifest
    valid_metas = [r for r in results if r is not None]
    manifest_path = os.path.join(root_dir, 'data/inspections/manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(valid_metas, f, indent=2)
    print(f"Manifest saved to: {manifest_path}")


if __name__ == '__main__':
    main()
