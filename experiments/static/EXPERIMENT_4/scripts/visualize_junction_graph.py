#!/usr/bin/env python3
"""
Anatomical Junction Keypoints & Topology Graph Visualizer
Automatically extracts and visualizes the 3 biological junction keypoints:
  1. J_top (Gold Diamond): Falciform-Diaphragm/Silhouette Root
  2. J_bottom (Cyan Diamond): Falciform-Ridge Umbilical Notch
  3. J_lateral (Magenta Diamond): Silhouette-Ridge Lateral Extremity Corner
Also draws the structural anatomical triangle graph connecting these junctions.
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

JUNCTION_STYLES = {
    'J_top':       {'name': 'Falc-Sil Root',      'bgr': (0, 215, 255),   'hex': '#ffd700'}, # Gold/Yellow
    'J_bottom':    {'name': 'Umbilical Notch',    'bgr': (255, 255, 0),   'hex': '#00ffff'}, # Cyan
    'J_lat_right': {'name': 'Right Lateral Tip',  'bgr': (255, 0, 255),   'hex': '#ff00ff'}, # Magenta
    'J_lat_left':  {'name': 'Left Lateral Tip',   'bgr': (0, 140, 255),   'hex': '#ff8c00'}  # Orange
}

CURVE_COLORS = {
    'ridge':      {'bgr': (0, 255, 0),   'name': 'Ridge'},       # Green (Class 1)
    'silhouette': {'bgr': (0, 0, 255),   'name': 'Silhouette'},  # Red (Class 2)
    'falciform':  {'bgr': (255, 140, 0), 'name': 'Falciform'}    # Blue (Class 3)
}

def extract_anatomical_junctions(data, orig_w, orig_h, canvas_size=1024, threshold_px=100.0):
    sx = canvas_size / float(orig_w)
    sy = canvas_size / float(orig_h)
    
    r_curves, s_curves, f_curves = [], [], []
    for shape in data.get('shapes', []):
        lbl = str(shape.get('label', '')).lower().strip()
        pts = np.array(shape.get('points', []), dtype=np.float32)
        if len(pts) < 2: continue
        scaled = np.column_stack([pts[:, 0] * sx, pts[:, 1] * sy])
        if 'rig' in lbl: r_curves.append(scaled)
        elif 'sil' in lbl: s_curves.append(scaled)
        elif 'falc' in lbl or 'lig' in lbl: f_curves.append(scaled)
        
    junctions = {}
    
    # 1. J_top (Falciform meets Silhouette)
    best_d = 1e9
    j_top_pt = None
    for f in f_curves:
        for s in s_curves:
            for f_pt in [f[0], f[-1]]:
                dists = np.linalg.norm(s - f_pt, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < best_d:
                    best_d = dists[min_idx]
                    j_top_pt = (f_pt + s[min_idx]) / 2.0
    visible_top = (best_d <= threshold_px and j_top_pt is not None)
    junctions['J_top'] = {
        'pt': j_top_pt if visible_top else None,
        'visible': visible_top,
        'dist_px': float(best_d) if j_top_pt is not None else None,
        'name': JUNCTION_STYLES['J_top']['name']
    }
    
    # 2. J_bottom (Falciform meets Ridge - Umbilical Notch)
    best_d = 1e9
    j_bot_pt = None
    for f in f_curves:
        for r in r_curves:
            for f_pt in [f[0], f[-1]]:
                dists = np.linalg.norm(r - f_pt, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < best_d:
                    best_d = dists[min_idx]
                    j_bot_pt = (f_pt + r[min_idx]) / 2.0
    visible_bot = (best_d <= threshold_px and j_bot_pt is not None)
    junctions['J_bottom'] = {
        'pt': j_bot_pt if visible_bot else None,
        'visible': visible_bot,
        'dist_px': float(best_d) if j_bot_pt is not None else None,
        'name': JUNCTION_STYLES['J_bottom']['name']
    }
    
    # 3. J_lat_right and J_lat_left (Ridge meets Silhouette at both extremities)
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

    vis_right = (best_dr <= threshold_px and pt_r is not None)
    vis_left  = (best_dl <= threshold_px and pt_l is not None)
    
    junctions['J_lat_right'] = {
        'pt': pt_r if vis_right else None,
        'visible': vis_right,
        'dist_px': float(best_dr) if pt_r is not None else None,
        'name': JUNCTION_STYLES['J_lat_right']['name']
    }
    junctions['J_lat_left'] = {
        'pt': pt_l if vis_left else None,
        'visible': vis_left,
        'dist_px': float(best_dl) if pt_l is not None else None,
        'name': JUNCTION_STYLES['J_lat_left']['name']
    }
    
    return junctions, r_curves, s_curves, f_curves

def draw_diamond(img, center, size, color, thickness=-1):
    cx, cy = int(center[0]), int(center[1])
    pts = np.array([
        [cx, cy - size],
        [cx + size, cy],
        [cx, cy + size],
        [cx - size, cy]
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA) if thickness == -1 else cv2.polylines(img, [pts], True, color, thickness, lineType=cv2.LINE_AA)

def render_junction_overlay(img_path, json_path, out_path, canvas_size=1024):
    img = cv2.imread(img_path)
    if img is None:
        return None
    orig_h, orig_w = img.shape[:2]
    img_canvas = cv2.resize(img, (canvas_size, canvas_size), interpolation=cv2.INTER_LINEAR)
    
    with open(json_path) as f:
        data = json.load(f)
        
    junctions, r_curves, s_curves, f_curves = extract_anatomical_junctions(data, orig_w, orig_h, canvas_size=canvas_size)
    
    # 1. Draw semi-transparent GT curves
    mask_layer = np.zeros_like(img_canvas)
    for c in r_curves:
        pts = np.round(c).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(mask_layer, [pts], False, CURVE_COLORS['ridge']['bgr'], 35, lineType=cv2.LINE_AA)
    for c in s_curves:
        pts = np.round(c).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(mask_layer, [pts], False, CURVE_COLORS['silhouette']['bgr'], 35, lineType=cv2.LINE_AA)
    for c in f_curves:
        pts = np.round(c).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(mask_layer, [pts], False, CURVE_COLORS['falciform']['bgr'], 35, lineType=cv2.LINE_AA)
        
    has_mask = (mask_layer > 0)
    composite = img_canvas.copy()
    composite[has_mask] = (img_canvas[has_mask].astype(np.float32) * 0.45 +
                           mask_layer[has_mask].astype(np.float32) * 0.55).astype(np.uint8)
                           
    # 2. Draw Structural Topology Graph (Dashed Polygon Edges)
    graph_pts = {}
    for k in ['J_top', 'J_bottom', 'J_lat_right', 'J_lat_left']:
        if junctions[k]['visible']:
            graph_pts[k] = np.round(junctions[k]['pt']).astype(int)
            
    # Connect available pairs with high-contrast dashed lines
    def draw_dashed_line(img, pt1, pt2, color, thickness=3, gap=15):
        dist = np.linalg.norm(pt2 - pt1)
        if dist < 1e-3: return
        num_steps = int(dist / gap)
        for i in range(0, num_steps, 2):
            t1 = i / float(num_steps)
            t2 = min((i + 1) / float(num_steps), 1.0)
            p_a = (int(pt1[0] + t1 * (pt2[0] - pt1[0])), int(pt1[1] + t1 * (pt2[1] - pt1[1])))
            p_b = (int(pt1[0] + t2 * (pt2[0] - pt1[0])), int(pt1[1] + t2 * (pt2[1] - pt1[1])))
            cv2.line(img, p_a, p_b, (0, 0, 0), thickness + 3, lineType=cv2.LINE_AA)
            cv2.line(img, p_a, p_b, color, thickness, lineType=cv2.LINE_AA)

    # Edge 1: Falciform ligament vertical span (J_top <-> J_bottom)
    if 'J_top' in graph_pts and 'J_bottom' in graph_pts:
        draw_dashed_line(composite, graph_pts['J_top'], graph_pts['J_bottom'], (255, 255, 255), thickness=3)
    # Edge 2: Right Ridge span (J_bottom <-> J_lat_right)
    if 'J_bottom' in graph_pts and 'J_lat_right' in graph_pts:
        draw_dashed_line(composite, graph_pts['J_bottom'], graph_pts['J_lat_right'], (0, 255, 255), thickness=3)
    # Edge 3: Left Ridge span (J_bottom <-> J_lat_left)
    if 'J_bottom' in graph_pts and 'J_lat_left' in graph_pts:
        draw_dashed_line(composite, graph_pts['J_bottom'], graph_pts['J_lat_left'], (0, 165, 255), thickness=3)
    # Edge 4: Right Silhouette dome (J_top <-> J_lat_right)
    if 'J_top' in graph_pts and 'J_lat_right' in graph_pts:
        draw_dashed_line(composite, graph_pts['J_top'], graph_pts['J_lat_right'], (255, 0, 255), thickness=3)
    # Edge 5: Left Silhouette triangular ligament (J_top <-> J_lat_left)
    if 'J_top' in graph_pts and 'J_lat_left' in graph_pts:
        draw_dashed_line(composite, graph_pts['J_top'], graph_pts['J_lat_left'], (100, 255, 100), thickness=3)
        
    # 3. Draw Junction Keypoint Markers (Diamond with double halo)
    for k, info in junctions.items():
        if not info['visible']:
            continue
        pt = np.round(info['pt']).astype(int)
        col = JUNCTION_STYLES[k]['bgr']
        
        # Halo + Diamond
        draw_diamond(composite, pt, 24, (0, 0, 0), thickness=-1)       # outer black
        draw_diamond(composite, pt, 20, (255, 255, 255), thickness=-1) # white ring
        draw_diamond(composite, pt, 15, col, thickness=-1)             # solid color
        draw_diamond(composite, pt, 6, (0, 0, 0), thickness=-1)        # central pin
        
        # Text label badge
        lbl = f"{k}: {info['name']}"
        coords_lbl = f"({pt[0]}, {pt[1]})"
        
        # Position label cleanly
        lx = min(max(pt[0] + 28, 20), canvas_size - 260)
        ly = min(max(pt[1] - 10, 50), canvas_size - 40)
        
        (tw1, th1), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        (tw2, th2), _ = cv2.getTextSize(coords_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bw = max(tw1, tw2) + 12
        bh = th1 + th2 + 16
        
        cv2.rectangle(composite, (lx - 6, ly - th1 - 4), (lx + bw, ly + th2 + 8), (0, 0, 0), -1)
        cv2.rectangle(composite, (lx - 6, ly - th1 - 4), (lx + bw, ly + th2 + 8), col, 2)
        cv2.putText(composite, lbl, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(composite, coords_lbl, (lx, ly + th2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # 4. Top HUD Banner
    fname = os.path.basename(img_path)
    banner_h = 58
    hud = composite.copy()
    cv2.rectangle(hud, (0, 0), (canvas_size, banner_h), (15, 15, 15), -1)
    composite = cv2.addWeighted(composite, 0.2, hud, 0.8, 0)
    
    vis_count = sum(1 for v in junctions.values() if v['visible'])
    status_str = f"Visible Junctions: {vis_count}/4"
    
    cv2.putText(composite, f"FRAME: {fname}", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    top_status = f"STRUCTURAL GRAPH: {status_str} | Top={'✓' if junctions['J_top']['visible'] else '✗'}, Bot={'✓' if junctions['J_bottom']['visible'] else '✗'}, LatR={'✓' if junctions['J_lat_right']['visible'] else '✗'}, LatL={'✓' if junctions['J_lat_left']['visible'] else '✗'}"
    cv2.putText(composite, top_status, (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1, cv2.LINE_AA)
                
    # Bottom Legend
    cv2.rectangle(composite, (0, canvas_size - 32), (canvas_size, canvas_size), (15, 15, 15), -1)
    cv2.putText(composite, "Green=Ridge (1) | Red=Silhouette (2) | Blue=Falciform (3) | Gold=J_top | Cyan=J_bot | Mag=J_lat_right | Org=J_lat_left",
                (15, canvas_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, composite, [cv2.IMWRITE_JPEG_QUALITY, 92])
    
    return {
        'filename': fname,
        'image_path': os.path.relpath(out_path, os.path.dirname(os.path.dirname(out_path))),
        'junctions': {k: {'visible': bool(v['visible']), 'pt': [int(p) for p in v['pt']] if v['pt'] is not None else None} for k, v in junctions.items()},
        'vis_count': int(vis_count)
    }

def generate_html_junction_gallery(records, out_html_path):
    records_json = json.dumps(records, indent=2)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Anatomical Junctions & Structural Graph Visualizer</title>
<style>
  body {{
    margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #111113; color: #f0f0f3;
    display: flex; height: 100vh; overflow: hidden;
  }}
  #sidebar {{
    width: 380px; background: #1a1a1d; border-right: 1px solid #2a2a2e;
    display: flex; flex-direction: column; height: 100%;
  }}
  #sidebar-header {{
    padding: 16px; border-bottom: 1px solid #2a2a2e;
  }}
  #sidebar-header h2 {{
    margin: 0 0 6px 0; font-size: 1.15rem; color: #ffd700;
  }}
  .legend-box {{
    background: #242428; border-radius: 6px; padding: 8px 10px; margin-top: 8px; font-size: 0.78rem; line-height: 1.5;
  }}
  .color-dot {{
    display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle;
  }}
  .filter-btns {{
    display: flex; gap: 6px; margin-top: 10px; flex-wrap: wrap;
  }}
  .filter-btn {{
    background: #2a2a2e; border: none; color: #aaa; padding: 6px 12px;
    border-radius: 6px; font-size: 0.75rem; cursor: pointer;
  }}
  .filter-btn.active {{
    background: #007aff; color: #fff; font-weight: 600;
  }}
  #search {{
    width: 100%; box-sizing: border-box; padding: 8px 12px;
    background: #2a2a2e; border: 1px solid #3a3a3e; border-radius: 6px;
    color: #fff; font-size: 0.85rem; margin-top: 10px;
  }}
  #list {{
    flex: 1; overflow-y: auto; padding: 8px;
  }}
  .item {{
    padding: 10px 12px; border-radius: 6px; cursor: pointer;
    margin-bottom: 4px; border-left: 4px solid transparent;
    transition: background 0.15s; font-size: 0.85rem;
  }}
  .item:hover {{ background: #26262a; }}
  .item.selected {{ background: #26262a; border-left-color: #ffd700; }}
  .badge {{
    display: inline-block; font-size: 0.7rem; font-weight: bold;
    padding: 2px 6px; border-radius: 4px; margin-top: 4px;
  }}
  .badge-4vis {{ background: rgba(52, 199, 89, 0.25); color: #34c759; }}
  .badge-3vis {{ background: rgba(0, 122, 255, 0.25); color: #007aff; }}
  .badge-2vis {{ background: rgba(255, 215, 0, 0.25); color: #ffd700; }}
  .badge-1vis {{ background: rgba(255, 149, 0, 0.25); color: #ff9500; }}
  .badge-0vis {{ background: rgba(255, 59, 48, 0.25); color: #ff3b30; }}

  #viewport {{
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 20px; background: #0c0c0e; position: relative;
  }}
  #img-box {{
    max-width: 90vh; max-height: 85vh;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.8);
    border-radius: 8px; overflow: hidden; background: #000;
  }}
  #main-img {{
    display: block; width: 100%; height: 100%; object-fit: contain;
  }}
  #navbar {{
    position: absolute; bottom: 20px; display: flex; gap: 12px;
    align-items: center; background: rgba(26, 26, 29, 0.9);
    backdrop-filter: blur(10px); padding: 8px 20px;
    border-radius: 30px; border: 1px solid #3a3a3e;
  }}
  .nav-btn {{
    background: #3a3a3e; color: #fff; border: none;
    padding: 6px 14px; border-radius: 20px; cursor: pointer; font-size: 0.85rem;
  }}
  .nav-btn:hover {{ background: #ffd700; color: #000; }}
  #counter {{
    font-size: 0.85rem; color: #aaa; min-width: 80px; text-align: center;
  }}
</style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <h2>📐 Anatomical Junction Graph</h2>
    <div class="legend-box">
      <div><span class="color-dot" style="background:#22c55e;"></span> <b>Ridge (Class 1)</b></div>
      <div><span class="color-dot" style="background:#ef4444;"></span> <b>Silhouette (Class 2)</b></div>
      <div><span class="color-dot" style="background:#3b82f6;"></span> <b>Falciform (Class 3)</b></div>
      <div style="margin-top:6px; border-top:1px solid #3a3a3e; padding-top:4px;">
        <span style="color:#ffd700;">◆</span> <b>J_top</b>: Falc-Sil Root<br>
        <span style="color:#00ffff;">◆</span> <b>J_bottom</b>: Umbilical Notch<br>
        <span style="color:#ff00ff;">◆</span> <b>J_lat_right</b>: Right Lateral Tip<br>
        <span style="color:#ff8c00;">◆</span> <b>J_lat_left</b>: Left Lateral Tip
      </div>
    </div>
    <input type="text" id="search" placeholder="Search frame (e.g. 03870, 08730)..." oninput="applyFilters()">
    <div class="filter-btns">
      <button class="filter-btn active" onclick="setFilter('all')">All</button>
      <button class="filter-btn" onclick="setFilter('Patient_40')">Patient 40</button>
      <button class="filter-btn" onclick="setFilter('both_lat')">Both Lateral Tips</button>
      <button class="filter-btn" onclick="setFilter('3_plus')">3+ Junctions</button>
      <button class="filter-btn" onclick="setFilter('hard_cases')">Key Hard Cases</button>
    </div>
  </div>
  <div id="list"></div>
</div>

<div id="viewport">
  <div id="img-box">
    <img id="main-img" src="" alt="Select a frame">
  </div>
  <div id="navbar">
    <button class="nav-btn" onclick="prevFrame()">◀ Prev [A]</button>
    <div id="counter">1 / 50</div>
    <button class="nav-btn" onclick="nextFrame()">Next [D] ▶</button>
  </div>
</div>

<script>
const RECORDS = {records_json};
let filtered = [...RECORDS];
let curIdx = 0;
let curFilter = 'all';

function renderList() {{
  const el = document.getElementById('list');
  el.innerHTML = '';
  filtered.forEach((r, i) => {{
    const item = document.createElement('div');
    item.className = 'item' + (i === curIdx ? ' selected' : '');
    item.onclick = () => selectIdx(i);
    
    let bClass = 'badge-1vis';
    if (r.vis_count === 4) bClass = 'badge-4vis';
    else if (r.vis_count === 3) bClass = 'badge-3vis';
    else if (r.vis_count === 2) bClass = 'badge-2vis';
    else if (r.vis_count === 0) bClass = 'badge-0vis';
    
    let subInfo = [];
    if (r.junctions.J_top && r.junctions.J_top.visible) subInfo.push('Top');
    if (r.junctions.J_bottom && r.junctions.J_bottom.visible) subInfo.push('Bot');
    if (r.junctions.J_lat_right && r.junctions.J_lat_right.visible) subInfo.push('LatR');
    if (r.junctions.J_lat_left && r.junctions.J_lat_left.visible) subInfo.push('LatL');
    let subStr = subInfo.length > 0 ? ('(' + subInfo.join(', ') + ')') : '(None)';

    item.innerHTML = `
      <div style="font-weight: 600;">${{r.filename}}</div>
      <span class="badge ${{bClass}}">${{r.vis_count}}/4 Junctions ${{subStr}}</span>
    `;
    el.appendChild(item);
  }});
  updateView();
}}

function selectIdx(i) {{
  if (i < 0 || i >= filtered.length) return;
  curIdx = i;
  document.querySelectorAll('.item').forEach((it, idx) => it.classList.toggle('selected', idx === curIdx));
  const curItem = document.querySelectorAll('.item')[curIdx];
  if (curItem) curItem.scrollIntoView({{ block: 'nearest' }});
  updateView();
}}

function updateView() {{
  if (filtered.length === 0) return;
  const cur = filtered[curIdx];
  document.getElementById('main-img').src = cur.image_path;
  document.getElementById('counter').innerText = `${{curIdx + 1}} / ${{filtered.length}}`;
}}

function prevFrame() {{ if (curIdx > 0) selectIdx(curIdx - 1); }}
function nextFrame() {{ if (curIdx < filtered.length - 1) selectIdx(curIdx + 1); }}

function setFilter(f) {{
  curFilter = f;
  document.querySelectorAll('.filter-btn').forEach(btn => {{
    btn.classList.toggle('active', btn.innerText.toLowerCase().includes(f.toLowerCase()) || (f==='all' && btn.innerText==='All'));
  }});
  applyFilters();
}}

function applyFilters() {{
  const q = document.getElementById('search').value.toLowerCase();
  filtered = RECORDS.filter(r => {{
    const matchQ = r.filename.toLowerCase().includes(q);
    let matchF = true;
    if (curFilter === 'Patient_40') matchF = r.filename.includes('Patient_40');
    else if (curFilter === 'both_lat') matchF = (r.junctions.J_lat_right && r.junctions.J_lat_right.visible && r.junctions.J_lat_left && r.junctions.J_lat_left.visible);
    else if (curFilter === '3_plus') matchF = (r.vis_count >= 3);
    else if (curFilter === 'hard_cases') {{
      const hc = ['03870', '08730', '08790', '08940', '09000', '09150', '03360', '03660'];
      matchF = hc.some(k => r.filename.includes(k));
    }}
    return matchQ && matchF;
  }});
  curIdx = 0;
  renderList();
}}

document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') prevFrame();
  if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') nextFrame();
}});

renderList();
</script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    with open(out_html_path, 'w') as f:
        f.write(html)
    print(f"Generated Interactive Junction HTML Gallery: {out_html_path}")

def main():
    out_dir = os.path.join(WORKSPACE_ROOT, 'scratch/junction_visualizations')
    img_out_dir = os.path.join(out_dir, 'overlays')
    os.makedirs(img_out_dir, exist_ok=True)
    
    # 1. All Patient 40 validation frames (101 frames)
    p40_json = sorted(glob.glob(os.path.join(WORKSPACE_ROOT, 'data/L3D/Val/labels/Patient_40_*.json')))
    
    # 2. Add known dual-lateral-tip frames from Train & Val so user can inspect both tips
    dual_lat_names = [
        'Patient_41_08700', 'Patient_41_06600', 'Patient_41_07470', 'Patient_41_07350',
        'Patient_41_07530', 'Patient_45_22530', 'Patient_35_0014040', 'Patient_50_0175800',
        'Patient_45_22500', 'Patient_35_0013920', 'Patient_35_0171600', 'Patient_35_0014160',
        'Patient_35_0014520', 'Patient_22_0142200', 'Patient_35_0014280', 'Patient_35_0013800'
    ]
    dual_targets = []
    for name in dual_lat_names:
        matches = glob.glob(os.path.join(WORKSPACE_ROOT, f'data/L3D/*/labels/{name}.json'))
        dual_targets.extend(matches)
        
    # 3. Additional diverse training frames
    train_sample = sorted(glob.glob(os.path.join(WORKSPACE_ROOT, 'data/L3D/Train/labels/*.json')))[:30]
    
    all_targets = sorted(list(set(p40_json + dual_targets + train_sample)))
    print(f"Rendering anatomical junction overlays for {len(all_targets)} frames...")
    
    records = []
    for jp in all_targets:
        stem = Path(jp).stem
        # find matching image
        is_val = 'Val' in jp
        img_dir = os.path.join(WORKSPACE_ROOT, f'data/L3D/{"Val" if is_val else "Train"}/images')
        img_path = os.path.join(img_dir, f"{stem}.jpg")
        if not os.path.exists(img_path):
            continue
            
        out_path = os.path.join(img_out_dir, f"{stem}_junction.jpg")
        rec = render_junction_overlay(img_path, jp, out_path)
        if rec:
            rec['image_path'] = os.path.relpath(out_path, out_dir)
            records.append(rec)
            
    html_path = os.path.join(out_dir, 'index.html')
    generate_html_junction_gallery(records, html_path)
    print(f"\n🎉 Successfully rendered {len(records)} junction frames!")
    print(f"👉 Open in browser: file://{html_path}")

if __name__ == '__main__':
    main()
