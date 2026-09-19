#!/usr/bin/env python3
"""
Center of Mass & Landmark Geometry Visualizer for Surgical AI
Visualizes the raw surgical frame + ground truth landmark polylines + Center of Mass points and vectors.
Generates both high-res overlay images and an interactive HTML viewer for easy inspection.
"""
import os
import sys
import glob
import json
import cv2
import numpy as np
from pathlib import Path

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from experiments.EXPERIMENT_4.models.bezier_utils import resample_polyline

COLORS = {
    1: {'bgr': (0, 255, 0),    'hex': '#22c55e', 'name': 'Ridge'},       # Green (Class 1)
    2: {'bgr': (0, 0, 255),    'hex': '#ef4444', 'name': 'Silhouette'},  # Red (Class 2)
    3: {'bgr': (255, 140, 0),  'hex': '#3b82f6', 'name': 'Falciform'}    # Blue (Class 3)
}

def render_frame_overlay(img_path, json_path, out_path, canvas_size=1024):
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    orig_h, orig_w = img.shape[:2]
    
    # Resize image to canvas_size x canvas_size
    img_canvas = cv2.resize(img, (canvas_size, canvas_size), interpolation=cv2.INTER_LINEAR)
    scale_x = canvas_size / float(orig_w)
    scale_y = canvas_size / float(orig_h)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Extract polylines by class
    curves_by_class = {1: [], 2: [], 3: []}
    all_raw_pts_by_class = {1: [], 2: [], 3: []}
    all_res_pts_by_class = {1: [], 2: [], 3: []}
    
    for shape in data.get('shapes', []):
        label = str(shape.get('label', '')).lower().strip()
        if label.startswith('r') or 'ridge' in label or 'rigde' in label:
            class_id = 1
        elif label.startswith('s') or 'sil' in label or 'margin' in label:
            class_id = 2
        elif label.startswith('f') or 'falc' in label or 'lig' in label or 'ligament' in label:
            class_id = 3
        else:
            continue
            
        pts = shape.get('points', [])
        if len(pts) < 2:
            continue
            
        scaled_line = np.array([[p[0] * scale_x, p[1] * scale_y] for p in pts], dtype=np.float32)
        curves_by_class[class_id].append(scaled_line)
        
        # Normalized raw coords
        norm_raw = np.array([[p[0] / orig_w, p[1] / orig_h] for p in pts], dtype=np.float32)
        all_raw_pts_by_class[class_id].append(norm_raw)
        
        resampled = resample_polyline(scaled_line, spacing=5.0)
        if len(resampled) >= 2:
            all_res_pts_by_class[class_id].append(resampled / float(canvas_size))
            
    # Draw GT polylines on overlay mask
    mask_layer = np.zeros_like(img_canvas)
    for cid in [1, 2, 3]:
        color = COLORS[cid]['bgr']
        for curve in curves_by_class[cid]:
            pts_int = np.round(curve).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(mask_layer, [pts_int], isClosed=False, color=color, thickness=35, lineType=cv2.LINE_AA)
            
    # Alpha blend GT polylines over original image
    has_mask = (mask_layer > 0)
    alpha = 0.55
    composite = img_canvas.copy()
    composite[has_mask] = (img_canvas[has_mask].astype(np.float32) * (1.0 - alpha) +
                           mask_layer[has_mask].astype(np.float32) * alpha).astype(np.uint8)
                           
    # Compute Center of Mass
    com_data = {}
    for cid in [1, 2, 3]:
        name = COLORS[cid]['name']
        if len(all_raw_pts_by_class[cid]) > 0:
            cat_raw = np.concatenate(all_raw_pts_by_class[cid], axis=0)
            raw_com = cat_raw.mean(axis=0) # [cx, cy] in [0, 1]
            
            cat_res = np.concatenate(all_res_pts_by_class[cid], axis=0) if len(all_res_pts_by_class[cid]) > 0 else cat_raw
            res_com = cat_res.mean(axis=0)
            
            com_data[name] = {
                'present': True,
                'raw_com': raw_com.tolist(),
                'res_com': res_com.tolist(),
                'pixel_raw': (int(raw_com[0] * canvas_size), int(raw_com[1] * canvas_size)),
                'pixel_res': (int(res_com[0] * canvas_size), int(res_com[1] * canvas_size)),
                'color': COLORS[cid]['bgr']
            }
        else:
            com_data[name] = {'present': False}
            
    # Draw Center of Mass points and markers
    for name, cinfo in com_data.items():
        if not cinfo['present']:
            continue
            
        px, py = cinfo['pixel_raw']
        col = cinfo['color']
        
        # 1. Outer black ring for strong contrast against any background
        cv2.circle(composite, (px, py), 22, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        # 2. White halo
        cv2.circle(composite, (px, py), 19, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        # 3. Solid colored center
        cv2.circle(composite, (px, py), 15, col, -1, lineType=cv2.LINE_AA)
        # 4. Central crosshair
        cv2.drawMarker(composite, (px, py), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2, line_type=cv2.LINE_AA)
        cv2.drawMarker(composite, (px, py), (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1, line_type=cv2.LINE_AA)
        
        # Label tag badge
        label_text = f"{name} ({cinfo['raw_com'][0]:.2f}, {cinfo['raw_com'][1]:.2f})"
        text_origin = (min(max(px + 25, 20), canvas_size - 220), min(max(py + 5, 30), canvas_size - 20))
        
        # Badge background
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bx, by = text_origin[0] - 4, text_origin[1] - th - 4
        cv2.rectangle(composite, (bx, by), (bx + tw + 8, by + th + baseline + 8), (0, 0, 0), -1)
        cv2.rectangle(composite, (bx, by), (bx + tw + 8, by + th + baseline + 8), col, 2)
        cv2.putText(composite, label_text, (text_origin[0], text_origin[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw Relative Vector Arrow: Silhouette -> Ridge
    vector_info = {}
    if com_data['Silhouette']['present'] and com_data['Ridge']['present']:
        sil_pt = com_data['Silhouette']['pixel_raw']
        rid_pt = com_data['Ridge']['pixel_raw']
        
        dx = com_data['Ridge']['raw_com'][0] - com_data['Silhouette']['raw_com'][0]
        dy = com_data['Ridge']['raw_com'][1] - com_data['Silhouette']['raw_com'][1]
        
        # Draw dotted / prominent vector line
        cv2.arrowedLine(composite, sil_pt, rid_pt, (0, 0, 0), 6, line_type=cv2.LINE_AA, tipLength=0.08)
        cv2.arrowedLine(composite, sil_pt, rid_pt, (255, 255, 255), 3, line_type=cv2.LINE_AA, tipLength=0.08)
        
        # Midpoint badge for vector
        mid_x = (sil_pt[0] + rid_pt[0]) // 2
        mid_y = (sil_pt[1] + rid_pt[1]) // 2
        v_text = f"v_rel: dx={dx:+.2f}, dy={dy:+.2f}"
        (vtw, vth), _ = cv2.getTextSize(v_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(composite, (mid_x - 4, mid_y - vth - 4), (mid_x + vtw + 4, mid_y + 4), (20, 20, 20), -1)
        cv2.rectangle(composite, (mid_x - 4, mid_y - vth - 4), (mid_x + vtw + 4, mid_y + 4), (255, 255, 255), 1)
        cv2.putText(composite, v_text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        pose_tag = "NORMAL"
        if dx < -0.20:
            pose_tag = "FLIPPED (Horizontal Inversion)"
        elif dy < -0.10:
            pose_tag = "RETRACTED (Vertical Inversion)"
            
        vector_info = {'dx': float(dx), 'dy': float(dy), 'pose': pose_tag}
    else:
        vector_info = {'dx': None, 'dy': None, 'pose': 'SINGLE/MISSING LANDMARK'}

    # HUD Banner at top
    fname = os.path.basename(img_path)
    banner_h = 52
    overlay_hud = composite.copy()
    cv2.rectangle(overlay_hud, (0, 0), (canvas_size, banner_h), (15, 15, 15), -1)
    composite = cv2.addWeighted(composite, 0.2, overlay_hud, 0.8, 0)
    
    cv2.putText(composite, f"FRAME: {fname}", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    pose_color = (0, 0, 255) if 'FLIPPED' in vector_info['pose'] or 'RETRACTED' in vector_info['pose'] else (0, 255, 100)
    cv2.putText(composite, f"POSE: {vector_info['pose']}", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, pose_color, 1, cv2.LINE_AA)
    
    # Legend at bottom
    cv2.rectangle(composite, (0, canvas_size - 30), (canvas_size, canvas_size), (15, 15, 15), -1)
    cv2.putText(composite, "Legend: Green = Ridge (1) | Red = Silhouette (2) | Blue = Falciform (3) | Target Circles = Center of Mass",
                (15, canvas_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
                
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, composite, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    return {
        'filename': fname,
        'image_path': os.path.relpath(out_path, os.path.dirname(os.path.dirname(out_path))),
        'com_data': {k: {'present': v['present'], 'raw_com': v.get('raw_com')} for k, v in com_data.items()},
        'vector_info': vector_info
    }

def generate_html_gallery(records, out_html_path, title="Center of Mass Geometry Visualizer"):
    records_json = json.dumps(records, indent=2)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #121214;
    color: #e5e5e7;
    display: flex;
    height: 100vh;
    overflow: hidden;
  }}
  #sidebar {{
    width: 360px;
    background: #1c1c1e;
    border-right: 1px solid #2c2c2e;
    display: flex;
    flex-direction: column;
    height: 100%;
  }}
  #sidebar-header {{
    padding: 16px;
    border-bottom: 1px solid #2c2c2e;
  }}
  #sidebar-header h2 {{
    margin: 0 0 8px 0;
    font-size: 1.1rem;
    color: #fff;
  }}
  .filter-btn-group {{
    display: flex;
    gap: 6px;
    margin-top: 10px;
    flex-wrap: wrap;
  }}
  .filter-btn {{
    background: #2c2c2e;
    border: none;
    color: #aaa;
    padding: 5px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    cursor: pointer;
  }}
  .filter-btn.active {{
    background: #007aff;
    color: #fff;
    font-weight: 600;
  }}
  #search-box {{
    width: 100%;
    box-sizing: border-box;
    padding: 8px 12px;
    background: #2c2c2e;
    border: 1px solid #3a3a3c;
    border-radius: 6px;
    color: #fff;
    font-size: 0.85rem;
    margin-top: 8px;
  }}
  #frame-list {{
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }}
  .frame-item {{
    padding: 10px 12px;
    border-radius: 6px;
    cursor: pointer;
    margin-bottom: 4px;
    border-left: 4px solid transparent;
    transition: background 0.15s;
    font-size: 0.85rem;
  }}
  .frame-item:hover {{
    background: #2c2c2e;
  }}
  .frame-item.selected {{
    background: #2c2c2e;
    border-left-color: #007aff;
  }}
  .frame-item .pose-badge {{
    display: inline-block;
    font-size: 0.7rem;
    font-weight: bold;
    padding: 2px 6px;
    border-radius: 4px;
    margin-top: 4px;
  }}
  .pose-normal {{ background: rgba(52, 199, 89, 0.2); color: #34c759; }}
  .pose-flipped {{ background: rgba(255, 59, 48, 0.25); color: #ff3b30; }}
  .pose-retracted {{ background: rgba(255, 149, 0, 0.25); color: #ff9500; }}
  .pose-single {{ background: rgba(142, 142, 147, 0.2); color: #8e8e93; }}

  #main-viewport {{
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
    background: #0d0d0f;
    position: relative;
  }}
  #image-container {{
    max-width: 90vh;
    max-height: 85vh;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
    border-radius: 8px;
    overflow: hidden;
    background: #000;
  }}
  #active-image {{
    display: block;
    width: 100%;
    height: 100%;
    object-fit: contain;
  }}
  #nav-bar {{
    position: absolute;
    bottom: 20px;
    display: flex;
    gap: 12px;
    align-items: center;
    background: rgba(28, 28, 30, 0.85);
    backdrop-filter: blur(10px);
    padding: 8px 18px;
    border-radius: 30px;
    border: 1px solid #3a3a3c;
  }}
  .nav-btn {{
    background: #3a3a3c;
    color: #fff;
    border: none;
    padding: 6px 14px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 0.85rem;
  }}
  .nav-btn:hover {{ background: #007aff; }}
  #counter-badge {{
    font-size: 0.85rem;
    color: #8e8e93;
    min-width: 80px;
    text-align: center;
  }}
</style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <h2>🎯 CoM Landmark Geometry</h2>
    <div style="font-size: 0.75rem; color: #888;">Inspect whether Center of Mass reflects true liver landmark pose.</div>
    <input type="text" id="search-box" placeholder="Search frame (e.g. 08940)..." oninput="applyFilters()">
    <div class="filter-btn-group">
      <button class="filter-btn active" onclick="setFilter('all')">All</button>
      <button class="filter-btn" onclick="setFilter('Patient_40')">Patient 40</button>
      <button class="filter-btn" onclick="setFilter('FLIPPED')">Flipped</button>
      <button class="filter-btn" onclick="setFilter('RETRACTED')">Retracted</button>
      <button class="filter-btn" onclick="setFilter('NORMAL')">Normal</button>
    </div>
  </div>
  <div id="frame-list"></div>
</div>

<div id="main-viewport">
  <div id="image-container">
    <img id="active-image" src="" alt="Select a frame">
  </div>
  <div id="nav-bar">
    <button class="nav-btn" onclick="prevFrame()">◀ Prev [A]</button>
    <div id="counter-badge">1 / 100</div>
    <button class="nav-btn" onclick="nextFrame()">Next [D] ▶</button>
  </div>
</div>

<script>
const RECORDS = {records_json};
let filteredRecords = [...RECORDS];
let currentIndex = 0;
let currentFilter = 'all';

function getPoseClass(pose) {{
  if (pose.includes('FLIPPED')) return 'pose-flipped';
  if (pose.includes('RETRACTED')) return 'pose-retracted';
  if (pose.includes('NORMAL')) return 'pose-normal';
  return 'pose-single';
}}

function renderList() {{
  const listEl = document.getElementById('frame-list');
  listEl.innerHTML = '';
  filteredRecords.forEach((r, idx) => {{
    const item = document.createElement('div');
    item.className = 'frame-item' + (idx === currentIndex ? ' selected' : '');
    item.onclick = () => selectIndex(idx);
    
    const pose = r.vector_info.pose;
    const poseClass = getPoseClass(pose);
    
    item.innerHTML = `
      <div style="font-weight: 600;">${{r.filename}}</div>
      <div><span class="pose-badge ${{poseClass}}">${{pose}}</span></div>
    `;
    listEl.appendChild(item);
  }});
  updateViewer();
}}

function selectIndex(idx) {{
  if (idx < 0 || idx >= filteredRecords.length) return;
  currentIndex = idx;
  const items = document.querySelectorAll('.frame-item');
  items.forEach((it, i) => it.classList.toggle('selected', i === currentIndex));
  if (items[currentIndex]) {{
    items[currentIndex].scrollIntoView({{ block: 'nearest' }});
  }}
  updateViewer();
}}

function updateViewer() {{
  if (filteredRecords.length === 0) return;
  const cur = filteredRecords[currentIndex];
  document.getElementById('active-image').src = cur.image_path;
  document.getElementById('counter-badge').innerText = `${{currentIndex + 1}} / ${{filteredRecords.length}}`;
}}

function prevFrame() {{
  if (currentIndex > 0) selectIndex(currentIndex - 1);
}}

function nextFrame() {{
  if (currentIndex < filteredRecords.length - 1) selectIndex(currentIndex + 1);
}}

function setFilter(f) {{
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(btn => {{
    btn.classList.toggle('active', btn.innerText.toLowerCase().includes(f.toLowerCase()) || (f==='all' && btn.innerText==='All'));
  }});
  applyFilters();
}}

function applyFilters() {{
  const query = document.getElementById('search-box').value.toLowerCase();
  filteredRecords = RECORDS.filter(r => {{
    const matchQuery = r.filename.toLowerCase().includes(query);
    let matchFilter = true;
    if (currentFilter === 'Patient_40') matchFilter = r.filename.includes('Patient_40');
    else if (currentFilter === 'FLIPPED') matchFilter = r.vector_info.pose.includes('FLIPPED');
    else if (currentFilter === 'RETRACTED') matchFilter = r.vector_info.pose.includes('RETRACTED');
    else if (currentFilter === 'NORMAL') matchFilter = r.vector_info.pose.includes('NORMAL');
    return matchQuery && matchFilter;
  }});
  currentIndex = 0;
  renderList();
}}

document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') prevFrame();
  if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') nextFrame();
}});

// Initialize
renderList();
</script>

</body>
</html>
"""
    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    with open(out_html_path, 'w') as f:
        f.write(html_content)
    print(f"Generated Interactive HTML Gallery: {out_html_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit_train', type=int, default=150,
                        help="Number of training samples to render (default 150 for fast generation, or -1 for all 921)")
    args = parser.parse_args()

    out_base = os.path.join(WORKSPACE_ROOT, 'scratch/com_visualizations')
    p40_out_dir = os.path.join(out_base, 'val_patient40')
    train_out_dir = os.path.join(out_base, 'train')
    os.makedirs(p40_out_dir, exist_ok=True)
    os.makedirs(train_out_dir, exist_ok=True)

    records = []

    # 1. Patient 40 Validation Frames (Render ALL 101 frames)
    val_p40_images = sorted(glob.glob(os.path.join(WORKSPACE_ROOT, 'data/L3D/Val/images/Patient_40_*.jpg')))
    print(f"Found {len(val_p40_images)} Patient 40 validation frames.")
    for img_path in val_p40_images:
        stem = Path(img_path).stem
        json_path = os.path.join(WORKSPACE_ROOT, f'data/L3D/Val/labels/{stem}.json')
        if not os.path.exists(json_path):
            continue
        out_img = os.path.join(p40_out_dir, f"{stem}_com.jpg")
        rec = render_frame_overlay(img_path, json_path, out_img)
        if rec:
            rec['image_path'] = os.path.relpath(out_img, out_base)
            records.append(rec)

    # 2. Training Set Frames (Render sample or all)
    train_images = sorted(glob.glob(os.path.join(WORKSPACE_ROOT, 'data/L3D/Train/images/*.jpg')))
    total_train = len(train_images)
    limit = args.limit_train if args.limit_train > 0 else total_train
    print(f"Rendering {min(limit, total_train)} of {total_train} training frames...")
    
    # Stratified selection: ensure flipped and normal frames are included
    for i, img_path in enumerate(train_images[:limit]):
        stem = Path(img_path).stem
        json_path = os.path.join(WORKSPACE_ROOT, f'data/L3D/Train/labels/{stem}.json')
        if not os.path.exists(json_path):
            continue
        out_img = os.path.join(train_out_dir, f"{stem}_com.jpg")
        rec = render_frame_overlay(img_path, json_path, out_img)
        if rec:
            rec['image_path'] = os.path.relpath(out_img, out_base)
            records.append(rec)

    # 3. Generate HTML Gallery
    html_out = os.path.join(out_base, 'index.html')
    generate_html_gallery(records, html_out)
    print(f"\n🎉 Successfully rendered {len(records)} frames!")
    print(f"👉 Open in your browser: file://{html_out}")

if __name__ == '__main__':
    main()
