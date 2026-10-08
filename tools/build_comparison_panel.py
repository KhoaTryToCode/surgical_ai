#!/usr/bin/env python3
"""
Builds an interactive Side-by-Side Comparison Panel and high-res Montage
comparing Flipped / Inverted cases in TRAIN vs. VAL.

Inputs:
  - User audit results CSV: data/user_audit_results.csv
  - L3D images & labels: data/L3D/Train, data/L3D/Val

Outputs:
  - Artifact PNG: train_vs_val_flipped_montage.png
  - Artifact HTML: train_vs_val_flipped_panel.html (and tools/train_vs_val_flipped_panel.html)
"""

import os
import glob
import json
import base64
import cv2
import pandas as pd
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ARTIFACT_DIR = "/Users/khoale/.gemini/antigravity/brain/5c26d00f-97c9-4c34-987d-61488255419c"
CSV_PATH = os.path.join(WORKSPACE_ROOT, "data/user_audit_results.csv")
DATA_DIR = os.path.join(WORKSPACE_ROOT, "data/L3D")


def render_overlay_bgr(img_path, json_path, target_size=(640, 360)):
    bgr = cv2.imread(img_path)
    if bgr is None:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    orig_h, orig_w = bgr.shape[:2]
    overlay = bgr.copy()
    
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Colors: Ridge=Green, Silhouette=Red, Falciform=Blue
            for shape in data.get('shapes', []):
                lbl = shape.get('label', '').lower()
                pts = np.array(shape.get('points', []), dtype=np.int32)
                if len(pts) < 2:
                    continue
                
                if 'ridg' in lbl or 'rigd' in lbl or lbl.startswith('r'):
                    color = (94, 197, 34)  # Green
                    thickness = 5
                elif 'sil' in lbl or lbl.startswith('s'):
                    color = (68, 68, 239)  # Red
                    thickness = 5
                elif 'falc' in lbl or 'lig' in lbl or lbl.startswith('f'):
                    color = (246, 130, 59) # Blue
                    thickness = 5
                else:
                    color = (200, 200, 200)
                    thickness = 3
                    
                cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.75, bgr, 0.25, 0, overlay)
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            
    if target_size:
        return cv2.resize(overlay, target_size, interpolation=cv2.INTER_AREA)
    return overlay


def img_to_base64_jpeg(img_bgr, quality=80):
    _, enc = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(enc).decode('utf-8')


def build_montage_png(train_cases, val_cases, out_path):
    """
    Creates a 4-row x 4-col high-res comparison grid:
    Cols 1 & 2: Train Flipped (Raw & Overlay)
    Cols 3 & 4: Val Flipped (Raw & Overlay)
    """
    card_w, card_h = 480, 270
    num_rows = min(len(train_cases), len(val_cases), 4)
    grid_img = np.zeros((num_rows * card_h + 100, 4 * card_w, 3), dtype=np.uint8)
    
    # Header
    grid_img[:100, :] = (20, 24, 33)
    cv2.putText(grid_img, "TRAIN SET FLIPPED (74 frames audited)", (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 230, 100), 2, cv2.LINE_AA)
    cv2.putText(grid_img, "VAL SET FLIPPED (19 frames audited: Patient 40 & 32)", (2 * card_w + 40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 150, 255), 2, cv2.LINE_AA)
    cv2.putText(grid_img, "Raw Frame | Landmark Overlay (Green=Ridge, Red=Sil, Blue=Falc)", (40, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(grid_img, "Raw Frame | Landmark Overlay (Green=Ridge, Red=Sil, Blue=Falc)", (2 * card_w + 40, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
    
    for r in range(num_rows):
        y_offset = 100 + r * card_h
        
        # Train frame
        tc = train_cases[r]
        raw_t = cv2.resize(cv2.imread(tc['img_path']), (card_w, card_h))
        over_t = render_overlay_bgr(tc['img_path'], tc['json_path'], (card_w, card_h))
        cv2.putText(raw_t, f"Train: {tc['patient']} ({tc['id']})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(raw_t, f"Train: {tc['patient']} ({tc['id']})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(over_t, f"Ridge Peak Y: {tc['r_peak_y']:.3f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(over_t, f"Ridge Peak Y: {tc['r_peak_y']:.3f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 100), 1)
        
        grid_img[y_offset:y_offset+card_h, 0:card_w] = raw_t
        grid_img[y_offset:y_offset+card_h, card_w:2*card_w] = over_t
        
        # Val frame
        vc = val_cases[r]
        raw_v = cv2.resize(cv2.imread(vc['img_path']), (card_w, card_h))
        over_v = render_overlay_bgr(vc['img_path'], vc['json_path'], (card_w, card_h))
        cv2.putText(raw_v, f"Val: {vc['patient']} ({vc['id']})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(raw_v, f"Val: {vc['patient']} ({vc['id']})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(over_v, f"Ridge Peak Y: {vc['r_peak_y']:.3f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(over_v, f"Ridge Peak Y: {vc['r_peak_y']:.3f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 200, 255), 1)
        
        grid_img[y_offset:y_offset+card_h, 2*card_w:3*card_w] = raw_v
        grid_img[y_offset:y_offset+card_h, 3*card_w:4*card_w] = over_v
        
        # Dividing lines
        cv2.line(grid_img, (2*card_w, y_offset), (2*card_w, y_offset+card_h), (80, 80, 80), 3)
        cv2.line(grid_img, (0, y_offset), (4*card_w, y_offset), (50, 50, 50), 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, grid_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"✅ Montage saved to: {out_path} ({grid_img.shape[1]}x{grid_img.shape[0]} px)")


def main():
    print("🚀 Reading audit CSV...")
    df = pd.read_csv(CSV_PATH)
    
    # Extract Train Flipped (74 frames)
    train_flipped_df = df[(df["split"] == "Train") & (df["status"] == "flipped")].sort_values("r_peak_y")
    print(f"Found {len(train_flipped_df)} Train Flipped frames.")
    
    # Extract Val Flipped (19 frames)
    val_flipped_df = df[(df["split"] == "Val") & (df["status"] == "flipped")].sort_values("r_peak_y")
    print(f"Found {len(val_flipped_df)} Val Flipped frames (Patient 40: 15, Patient 32: 4).")
    
    def df_to_items(sub_df):
        items = []
        for _, row in sub_df.iterrows():
            stem = row['id']
            split = row['split']
            img_path = os.path.join(DATA_DIR, split, "images", f"{stem}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(DATA_DIR, split, "images", f"{stem}.png")
            json_path = os.path.join(DATA_DIR, split, "labels", f"{stem}.json")
            if os.path.exists(img_path):
                items.append({
                    'id': stem,
                    'patient': row['patient'],
                    'split': split,
                    'r_peak_y': float(row['r_peak_y']),
                    'img_path': img_path,
                    'json_path': json_path
                })
        return items

    train_cases = df_to_items(train_flipped_df)
    val_cases = df_to_items(val_flipped_df)
    
    # 1. Generate PNG Montage (comparing top 4 Train vs top 4 Val)
    montage_png_path = os.path.join(ARTIFACT_DIR, "train_vs_val_flipped_montage.png")
    build_montage_png(train_cases, val_cases, montage_png_path)
    
    # 2. Build Interactive HTML Comparison Panel
    print("🎨 Pre-rendering base64 thumbnails for interactive HTML panel...")
    
    def prepare_card_data(cases_list):
        res = []
        for c in cases_list:
            raw_bgr = cv2.imread(c['img_path'])
            raw_thumb = cv2.resize(raw_bgr, (400, 225), interpolation=cv2.INTER_AREA)
            over_thumb = render_overlay_bgr(c['img_path'], c['json_path'], (400, 225))
            res.append({
                'id': c['id'],
                'patient': c['patient'],
                'split': c['split'],
                'r_peak_y': round(c['r_peak_y'], 3),
                'raw_b64': img_to_base64_jpeg(raw_thumb, 75),
                'over_b64': img_to_base64_jpeg(over_thumb, 75)
            })
        return res
        
    train_cards = prepare_card_data(train_cases)
    val_cards = prepare_card_data(val_cases)
    
    html_out_path = os.path.join(ARTIFACT_DIR, "train_vs_val_flipped_panel.html")
    tools_html_path = os.path.join(WORKSPACE_ROOT, "tools/train_vs_val_flipped_panel.html")
    
    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Train vs. Val Flipped Liver Anatomical Comparison Panel</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    .scrollbar-thin::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    .scrollbar-thin::-webkit-scrollbar-track {{ background: #0f172a; }}
    .scrollbar-thin::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 3px; }}
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen p-6 antialiased flex flex-col gap-6">

  <!-- Header & Executive Summary -->
  <header class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-wrap items-center justify-between gap-4">
    <div>
      <div class="flex items-center gap-2">
        <span class="text-2xl">⚖️</span>
        <h1 class="text-2xl font-bold tracking-tight text-white">Train vs. Val Flipped Anatomical Comparison Panel</h1>
      </div>
      <p class="text-sm text-slate-400 mt-1">
        Side-by-side analysis of your audited surgical inversion frames: <span class="text-emerald-400 font-semibold">TRAIN ({len(train_cards)} frames)</span> vs. <span class="text-rose-400 font-semibold">VAL ({len(val_cards)} frames)</span>.
      </p>
    </div>

    <!-- Quick Stats Summary -->
    <div class="flex items-center gap-3 text-xs">
      <div class="bg-slate-800 border border-slate-700 px-4 py-2 rounded-xl text-center">
        <div class="text-slate-400 font-medium">Train Flipped</div>
        <div class="text-emerald-400 font-bold text-base">{len(train_cards)} frames (8.0%)</div>
        <div class="text-[11px] text-slate-500">Patient 20, 12, 38, 53, 39</div>
      </div>
      <div class="bg-slate-800 border border-slate-700 px-4 py-2 rounded-xl text-center">
        <div class="text-slate-400 font-medium">Val Flipped</div>
        <div class="text-rose-400 font-bold text-base">{len(val_cards)} frames (15.6%)</div>
        <div class="text-[11px] text-slate-500">Patient 40 (15) & 32 (4)</div>
      </div>
      <div class="bg-slate-800 border border-slate-700 px-4 py-2 rounded-xl text-center">
        <div class="text-slate-400 font-medium">Imbalance Factor</div>
        <div class="text-amber-400 font-bold text-base">{(len(val_cards)/122) / (len(train_cards)/921):.1f}x</div>
        <div class="text-[11px] text-slate-500">Val has 2x higher flipped rate</div>
      </div>
    </div>
  </header>

  <!-- Side-by-Side Dual Synchronized Inspection Workbench -->
  <section class="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">

    <!-- LEFT COLUMN: TRAIN FLIPPED -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col gap-4 shadow-xl">
      <div class="flex items-center justify-between pb-3 border-b border-slate-800">
        <div class="flex items-center gap-2">
          <span class="w-3 h-3 rounded-full bg-emerald-500"></span>
          <h2 class="text-lg font-bold text-emerald-400">TRAIN Flipped Frames ({len(train_cards)})</h2>
        </div>
        <span id="train-counter" class="text-xs text-slate-400 font-mono">Frame 1 of {len(train_cards)}</span>
      </div>

      <!-- Main Train Inspection Display -->
      <div class="flex flex-col gap-2 bg-slate-950 p-3 rounded-xl border border-slate-800">
        <div class="flex items-center justify-between text-xs">
          <span id="train-card-meta" class="font-bold text-emerald-300">Patient 00</span>
          <span id="train-card-ridge" class="text-slate-400 font-mono">Ridge Peak Y: 0.000</span>
        </div>

        <div class="grid grid-cols-2 gap-2 aspect-video overflow-hidden rounded-lg bg-black">
          <div class="relative flex flex-col">
            <span class="absolute top-2 left-2 bg-black/70 px-2 py-0.5 rounded text-[10px] text-slate-300 z-10">Raw Frame</span>
            <img id="train-img-raw" src="" alt="Train raw" class="w-full h-full object-contain">
          </div>
          <div class="relative flex flex-col">
            <span class="absolute top-2 left-2 bg-black/70 px-2 py-0.5 rounded text-[10px] text-emerald-300 z-10">Overlay (Green=Ridge)</span>
            <img id="train-img-over" src="" alt="Train overlay" class="w-full h-full object-contain">
          </div>
        </div>
      </div>

      <!-- Train Thumbnail Selector Carousel -->
      <div class="flex flex-col gap-1.5">
        <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Select Train Frame:</div>
        <div id="train-thumbs" class="flex gap-2.5 overflow-x-auto scrollbar-thin pb-2 pt-1">
          <!-- Populated by JS -->
        </div>
      </div>
    </div>

    <!-- RIGHT COLUMN: VAL FLIPPED -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col gap-4 shadow-xl">
      <div class="flex items-center justify-between pb-3 border-b border-slate-800">
        <div class="flex items-center gap-2">
          <span class="w-3 h-3 rounded-full bg-rose-500"></span>
          <h2 class="text-lg font-bold text-rose-400">VAL Flipped Frames ({len(val_cards)})</h2>
        </div>
        <span id="val-counter" class="text-xs text-slate-400 font-mono">Frame 1 of {len(val_cards)}</span>
      </div>

      <!-- Main Val Inspection Display -->
      <div class="flex flex-col gap-2 bg-slate-950 p-3 rounded-xl border border-slate-800">
        <div class="flex items-center justify-between text-xs">
          <span id="val-card-meta" class="font-bold text-rose-300">Patient 40</span>
          <span id="val-card-ridge" class="text-slate-400 font-mono">Ridge Peak Y: 0.005</span>
        </div>

        <div class="grid grid-cols-2 gap-2 aspect-video overflow-hidden rounded-lg bg-black">
          <div class="relative flex flex-col">
            <span class="absolute top-2 left-2 bg-black/70 px-2 py-0.5 rounded text-[10px] text-slate-300 z-10">Raw Frame</span>
            <img id="val-img-raw" src="" alt="Val raw" class="w-full h-full object-contain">
          </div>
          <div class="relative flex flex-col">
            <span class="absolute top-2 left-2 bg-black/70 px-2 py-0.5 rounded text-[10px] text-rose-300 z-10">Overlay (Green=Ridge)</span>
            <img id="val-img-over" src="" alt="Val overlay" class="w-full h-full object-contain">
          </div>
        </div>
      </div>

      <!-- Val Thumbnail Selector Carousel -->
      <div class="flex flex-col gap-1.5">
        <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Select Val Frame:</div>
        <div id="val-thumbs" class="flex gap-2.5 overflow-x-auto scrollbar-thin pb-2 pt-1">
          <!-- Populated by JS -->
        </div>
      </div>
    </div>

  </section>

  <!-- Key Clinical Findings Box -->
  <section class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
    <h3 class="text-base font-bold text-white flex items-center gap-2 mb-3">
      <span>💡</span> Key Findings from Your Train vs. Val Audit
    </h3>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-300 leading-relaxed">
      <div class="bg-slate-950/60 border border-slate-800 p-4 rounded-xl">
        <h4 class="font-bold text-emerald-400 mb-1">1. Train Flipped Anatomy</h4>
        <p>In Train (74 frames), inverted frames come primarily from <strong>Patient 20 (29 frames)</strong>, <strong>Patient 12 (20 frames)</strong>, and <strong>Patient 53 (5 frames)</strong>. Surgical graspers clamp the visceral flap and gallbladder, hoisting the anterior ridge up to $Y \in [0.000, 0.170]$.</p>
      </div>
      <div class="bg-slate-950/60 border border-slate-800 p-4 rounded-xl">
        <h4 class="font-bold text-rose-400 mb-1">2. Val Flipped Anatomy</h4>
        <p>In Val (19 frames), <strong>Patient 40 (15 frames)</strong> exhibits identical severe dual-grasper retraction with the ridge elevated to $Y = 0.005 - 0.015$, accompanied by <strong>Patient 32 (4 frames)</strong> with lateral elevation.</p>
      </div>
      <div class="bg-slate-950/60 border border-slate-800 p-4 rounded-xl">
        <h4 class="font-bold text-amber-400 mb-1">3. Why Vanilla Models Failed</h4>
        <p>Because $92\%$ of Train is flat canonical dome views, standard models overfit to "Ridge is at the bottom". Your <strong>Junction Steering mechanism</strong> is what allowed Experiment 5 to achieve <strong>70.91% on Patient 40</strong> by anchoring predictions to the biological junction graph.</p>
      </div>
    </div>
  </section>

  <script>
    const trainCards = {json.dumps(train_cards)};
    const valCards = {json.dumps(val_cards)};

    let currentTrainIdx = 0;
    let currentValIdx = 0;

    function renderTrainThumbs() {{
      const cont = document.getElementById('train-thumbs');
      cont.innerHTML = '';
      trainCards.forEach((c, idx) => {{
        const btn = document.createElement('button');
        btn.className = `flex-shrink-0 w-28 rounded-lg overflow-hidden border-2 transition ${{idx === currentTrainIdx ? 'border-emerald-500 scale-105' : 'border-slate-800 opacity-60 hover:opacity-100'}}`;
        btn.onclick = () => selectTrain(idx);
        btn.innerHTML = `
          <img src="${{c.over_b64}}" class="w-full h-16 object-cover">
          <div class="bg-slate-950 p-1 text-[10px] text-center font-mono text-slate-300 truncate">${{c.patient}}</div>
        `;
        cont.appendChild(btn);
      }});
    }}

    function selectTrain(idx) {{
      currentTrainIdx = idx;
      const c = trainCards[idx];
      document.getElementById('train-counter').textContent = `Frame ${{idx + 1}} of ${{trainCards.length}}`;
      document.getElementById('train-card-meta').textContent = `${{c.patient}} — ${{c.id}}`;
      document.getElementById('train-card-ridge').textContent = `Ridge Peak Y: ${{c.r_peak_y}}`;
      document.getElementById('train-img-raw').src = c.raw_b64;
      document.getElementById('train-img-over').src = c.over_b64;
      renderTrainThumbs();
    }}

    function renderValThumbs() {{
      const cont = document.getElementById('val-thumbs');
      cont.innerHTML = '';
      valCards.forEach((c, idx) => {{
        const btn = document.createElement('button');
        btn.className = `flex-shrink-0 w-28 rounded-lg overflow-hidden border-2 transition ${{idx === currentValIdx ? 'border-rose-500 scale-105' : 'border-slate-800 opacity-60 hover:opacity-100'}}`;
        btn.onclick = () => selectVal(idx);
        btn.innerHTML = `
          <img src="${{c.over_b64}}" class="w-full h-16 object-cover">
          <div class="bg-slate-950 p-1 text-[10px] text-center font-mono text-slate-300 truncate">${{c.patient}}</div>
        `;
        cont.appendChild(btn);
      }});
    }}

    function selectVal(idx) {{
      currentValIdx = idx;
      const c = valCards[idx];
      document.getElementById('val-counter').textContent = `Frame ${{idx + 1}} of ${{valCards.length}}`;
      document.getElementById('val-card-meta').textContent = `${{c.patient}} — ${{c.id}}`;
      document.getElementById('val-card-ridge').textContent = `Ridge Peak Y: ${{c.r_peak_y}}`;
      document.getElementById('val-img-raw').src = c.raw_b64;
      document.getElementById('val-img-over').src = c.over_b64;
      renderValThumbs();
    }}

    selectTrain(0);
    selectVal(0);
  </script>
</body>
</html>
"""

    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(html_code)
    with open(tools_html_path, "w", encoding="utf-8") as f:
        f.write(html_code)
    print(f"✅ HTML Panel saved to: {html_out_path} and {tools_html_path}")


if __name__ == "__main__":
    main()
