import os
import sys
import json
import glob
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

def render_single_uv_frame(task):
    img_path, json_path, out_path, is_flipped_from_anchor, target_size = task
    stem = os.path.splitext(os.path.basename(img_path))[0]
    
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    orig_h, orig_w = img_bgr.shape[:2]
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
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

    # Determine x range
    pts_for_x = []
    if has_s: pts_for_x.append(sil_all)
    if has_r: pts_for_x.append(ridge_all)
    if has_f: pts_for_x.append(falc_all)
    
    if len(pts_for_x) == 0:
        return None
        
    all_combined = np.vstack(pts_for_x)
    x_min = max(0, int(np.floor(all_combined[:, 0].min())))
    x_max = min(target_size - 1, int(np.ceil(all_combined[:, 0].max())))
    xs = np.arange(x_min, x_max + 1)

    # 1. Silhouette profile y_sil(x)
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

    # 2. Ridge profile y_ridge(x)
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
        hinge_status = "Detected (Falciform Polyline)"
    else:
        falc_x_mid = float(x_min + x_max) / 2.0
        hinge_status = "Inferred Centerline"

    # Chirality detection:
    # In normal view, ridge is on bottom (larger y), silhouette is on top (smaller y)
    is_flipped = bool(is_flipped_from_anchor)
    mean_y_sil = float(np.mean(y_sil_interp))
    mean_y_ridge = float(np.mean(y_ridge_interp))
    if mean_y_ridge < mean_y_sil - 15.0:
        is_flipped = True

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
    
    # ── LEFT PANEL: 1D Sparse Polylines (Current Baseline) ────────────────────
    p_left = cv2.resize(img_bgr, (target_size, target_size))
    for p in ridge_pts:
        cv2.polylines(p_left, [np.round(p).astype(np.int32)], False, (0, 230, 80), 8, lineType=cv2.LINE_AA) # Green
    for p in sil_pts:
        cv2.polylines(p_left, [np.round(p).astype(np.int32)], False, (0, 60, 255), 8, lineType=cv2.LINE_AA) # Red
    for p in falc_pts:
        cv2.polylines(p_left, [np.round(p).astype(np.int32)], False, (255, 140, 0), 8, lineType=cv2.LINE_AA) # Blue
        
    cv2.rectangle(p_left, (0, 0), (target_size, 46), (15, 18, 24), -1)
    cv2.putText(p_left, "Current Paradigm: Sparse 1D Polylines", (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.rectangle(p_left, (0, target_size - 36), (target_size, target_size), (15, 18, 24), -1)
    cv2.putText(p_left, "Green: Ridge | Red: Silhouette | Blue: Falciform  [Interior = 0 labels]", (15, target_size - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 200, 210), 1, cv2.LINE_AA)

    # ── RIGHT PANEL: 2D Continuous Riemannian Manifold (Proposed) ─────────────
    p_right = cv2.resize(img_bgr, (target_size, target_size))
    uv_bgr = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    uv_bgr[:, :, 0] = np.clip((1.0 - u_coord) * 230.0, 0, 255).astype(np.uint8)  # Blue (Left lobe weight)
    uv_bgr[:, :, 1] = np.clip(v_coord * 255.0, 0, 255).astype(np.uint8)          # Green (Vertical height)
    uv_bgr[:, :, 2] = np.clip(u_coord * 255.0, 0, 255).astype(np.uint8)          # Red (Right lobe weight)
    
    m_bool = mask_liver > 0
    p_right[m_bool] = (p_right[m_bool] * 0.38 + uv_bgr[m_bool] * 0.62).astype(np.uint8)
    
    # Coordinate grid mesh (cyan u-ribs, gold v-rings, magenta hinge)
    grid_img = np.zeros_like(p_right)
    for u_val in np.linspace(0.1, 0.9, 9):
        diff = np.abs(u_coord - u_val)
        iso = (diff < 0.012) & (mask_liver > 0)
        grid_img[iso] = (255, 255, 0) # Cyan ribs
    for v_val in np.linspace(0.1, 0.9, 9):
        diff = np.abs(v_coord - v_val)
        iso = (diff < 0.012) & (mask_liver > 0)
        grid_img[iso] = (0, 215, 255) # Gold rings
    falc_iso = (np.abs(u_coord - 0.5) < 0.015) & (mask_liver > 0)
    grid_img[falc_iso] = (255, 0, 220) # Magenta central hinge
    
    m_grid = (grid_img.sum(axis=-1) > 0)
    p_right[m_grid] = (p_right[m_grid] * 0.25 + grid_img[m_grid] * 0.75).astype(np.uint8)
    
    # Draw sharp white boundary
    cnts_liver, _ = cv2.findContours(mask_liver, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p_right, cnts_liver, -1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Anatomical GPS sample callouts
    coords_pts = np.argwhere(mask_liver > 0)
    if len(coords_pts) > 0:
        y_min_l, y_max_l = coords_pts[:, 0].min(), coords_pts[:, 0].max()
        x_min_l, x_max_l = coords_pts[:, 1].min(), coords_pts[:, 1].max()
        callouts = [
            (int(x_min_l * 0.7 + falc_x_mid * 0.3), int(y_min_l * 0.4 + y_max_l * 0.6)),
            (int(x_min_l * 0.4 + falc_x_mid * 0.6), int(y_min_l * 0.7 + y_max_l * 0.3)),
            (int(falc_x_mid * 0.6 + x_max_l * 0.4), int(y_min_l * 0.4 + y_max_l * 0.6)),
            (int(falc_x_mid * 0.3 + x_max_l * 0.7), int(y_min_l * 0.7 + y_max_l * 0.3)),
        ]
        for cx, cy in callouts:
            if 0 <= cy < target_size and 0 <= cx < target_size and mask_liver[cy, cx] > 0:
                cv2.circle(p_right, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(p_right, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)
                txt = f"({u_coord[cy, cx]:.2f}, {v_coord[cy, cx]:.2f})"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                cv2.rectangle(p_right, (cx + 6, cy - th - 3), (cx + 10 + tw, cy + 4), (10, 12, 16), -1)
                cv2.putText(p_right, txt, (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # Header Right
    status_str = "[FLIPPED RETRACTION]" if is_flipped else "[NORMAL ANATOMY]"
    status_col = (0, 140, 255) if is_flipped else (0, 255, 120)
    cv2.rectangle(p_right, (0, 0), (target_size, 46), (15, 18, 24), -1)
    cv2.putText(p_right, f"Proposed: 2D UV Manifold  {status_str}", (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.68, status_col, 2, cv2.LINE_AA)
    
    # Footer Right
    cv2.rectangle(p_right, (0, target_size - 36), (target_size, target_size), (15, 18, 24), -1)
    cv2.putText(p_right, "u in [0,1] (Left-Right) | v in [0,1] (Ridge-Silhouette) | Magenta=Hinge (u=0.5)", (15, target_size - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 230, 255), 1, cv2.LINE_AA)

    # Stitch into 1536 x 768
    composite = np.hstack([p_left, p_right])
    
    # Write JPEG with high fidelity and compact size
    cv2.imwrite(out_path, composite, [cv2.IMWRITE_JPEG_QUALITY, 86])
    
    # Extract patient and frame index
    parts = stem.split('_')
    patient = f"{parts[0]}_{parts[1]}"
    frame_num = parts[2] if len(parts) > 2 else "000000"

    meta = {
        'stem': stem,
        'patient': patient,
        'frame': frame_num,
        'is_flipped': is_flipped,
        'landmarks': {
            'has_ridge': bool(has_r),
            'has_sil': bool(has_s),
            'has_falc': bool(has_f),
            'count': int(has_r) + int(has_s) + int(has_f)
        },
        'liver_area_px': int(mask_liver.sum()),
        'coverage_pct': round(float(mask_liver.sum()) / (target_size * target_size) * 100.0, 2),
        'falc_x_mid': round(float(falc_x_mid), 1),
        'hinge_status': hinge_status,
        'u_span': [round(float(u_coord[mask_liver > 0].min()), 2), round(float(u_coord[mask_liver > 0].max()), 2)] if len(coords_pts) > 0 else [0.0, 0.0],
        'v_span': [round(float(v_coord[mask_liver > 0].min()), 2), round(float(v_coord[mask_liver > 0].max()), 2)] if len(coords_pts) > 0 else [0.0, 0.0],
        'orig_w': orig_w,
        'orig_h': orig_h
    }
    return meta

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    train_img_dir = os.path.join(root_dir, 'data/L3D/Train/images')
    train_lbl_dir = os.path.join(root_dir, 'data/L3D/Train/labels')
    out_dir = os.path.join(root_dir, 'data/inspections/train_uv_overlays')
    os.makedirs(out_dir, exist_ok=True)

    # Load existing manifest to inherit verified corner chirality
    orig_manifest_p = os.path.join(root_dir, 'data/inspections/manifest.json')
    flipped_lookup = {}
    if os.path.exists(orig_manifest_p):
        with open(orig_manifest_p, 'r') as f:
            for item in json.load(f):
                flipped_lookup[item['stem']] = item.get('is_flipped', False)

    img_files = sorted(glob.glob(os.path.join(train_img_dir, '*.jpg')))
    print(f"Found {len(img_files)} training images to render.")

    tasks = []
    for img_p in img_files:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        json_p = os.path.join(train_lbl_dir, f"{stem}.json")
        out_p = os.path.join(out_dir, f"{stem}.jpg")
        if os.path.exists(json_p):
            is_flip = flipped_lookup.get(stem, False)
            tasks.append((img_p, json_p, out_p, is_flip, 768))

    print(f"🚀 Launching multi-threaded batch rendering of {len(tasks)} frames (8 threads)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(render_single_uv_frame, tasks))
    dt = time.time() - t0
    
    valid_metas = [r for r in results if r is not None]
    fps = len(valid_metas) / max(dt, 0.01)
    print(f"✅ Rendered {len(valid_metas)} frames in {dt:.1f}s ({fps:.1f} fps)!")

    # Save enriched manifest
    manifest_out = os.path.join(root_dir, 'data/inspections/uv_manifest.json')
    with open(manifest_out, 'w') as f:
        json.dump(valid_metas, f, indent=2)
    print(f"📄 Saved UV manifest to: {manifest_out}")

    # Summary Statistics
    count_3 = sum(1 for m in valid_metas if m['landmarks']['count'] == 3)
    count_2 = sum(1 for m in valid_metas if m['landmarks']['count'] == 2)
    count_1 = sum(1 for m in valid_metas if m['landmarks']['count'] == 1)
    count_flipped = sum(1 for m in valid_metas if m['is_flipped'])
    
    print("\n" + "="*60)
    print("SURGICAL UV MANIFOLD BATCH RENDERING SUMMARY:")
    print(f"  Total Valid Frames:       {len(valid_metas)} / {len(img_files)}")
    print(f"  3 Landmarks Present:      {count_3} ({count_3/len(valid_metas)*100:.1f}%)")
    print(f"  2 Landmarks Present:      {count_2} ({count_2/len(valid_metas)*100:.1f}%)")
    print(f"  1 Landmark Present:       {count_1} ({count_1/len(valid_metas)*100:.1f}%)")
    print(f"  Flipped / Inverted Views: {count_flipped} ({count_flipped/len(valid_metas)*100:.1f}%)")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
