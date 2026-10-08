#!/usr/bin/env python3
"""
Interactive Laparoscopic Liver Orientation & Traction Auditor.
A standalone local web app to audit Train and Val datasets and compute the true anatomical distribution.

Features:
  - Side-by-side view: Raw Laparoscopic Frame vs. Landmark Polyline Overlay
  - Rapid keyboard shortcuts:
      [F] or [1] -> Flipped / Retracted Undersurface (auto-advances)
      [N] or [0] -> Normal / Canonical Dome (auto-advances)
      [U]        -> Unsure / Ambiguous
      [Left/Right] -> Navigate
  - Live statistical dashboard: Train vs. Val Flipped % and imbalance ratio
  - Instant auto-save to 'data/orientation_audit_results.json'
  - Export to CSV / JSON
"""

import os
import sys
import glob
import json
import re
import cv2
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import webbrowser

# Workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(WORKSPACE_ROOT, 'data/L3D')
SAVE_FILE = os.path.join(WORKSPACE_ROOT, 'data/orientation_audit_results.json')

# Cache for overlays
OVERLAY_CACHE = {}


def extract_patient_id(filename):
    m = re.search(r"patient[_\s-]*(\d+)", filename, re.IGNORECASE)
    if m:
        return f"Patient_{int(m.group(1)):02d}"
    return "Unknown"


def scan_dataset():
    """Scans L3D Train, Val, and Test frames, loading existing labels and computing initial heuristic."""
    # Load existing annotations if present
    existing_labels = {}
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, 'r', encoding='utf-8') as f:
                existing_labels = json.load(f)
        except Exception as e:
            print(f"Warning: could not load existing {SAVE_FILE}: {e}")

    frames = []
    splits = ['Train', 'Val', 'Test']

    for split in splits:
        img_dir = os.path.join(DATA_DIR, split, 'images')
        lbl_dir = os.path.join(DATA_DIR, split, 'labels')
        if not os.path.exists(img_dir):
            continue

        for img_path in sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(lbl_dir, f"{stem}.json")
            patient = extract_patient_id(stem)

            # Analyze geometric pose heuristic
            r_peak_y = 1.0
            s_mean_y = 0.5
            has_ridge = False
            has_sil = False
            has_falc = False

            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as jf:
                        data = json.load(jf)
                    w = data.get('imageWidth', 1920)
                    h = data.get('imageHeight', 1080)
                    r_pts, s_pts = [], []
                    for shape in data.get('shapes', []):
                        lbl = shape.get('label', '').lower()
                        pts = np.array(shape.get('points', []))
                        if len(pts) < 2:
                            continue
                        scaled = pts / [w, h]
                        if 'ridg' in lbl or 'rigd' in lbl or lbl.startswith('r'):
                            r_pts.extend(scaled)
                            has_ridge = True
                        elif 'sil' in lbl or lbl.startswith('s'):
                            s_pts.extend(scaled)
                            has_sil = True
                        elif 'falc' in lbl or 'lig' in lbl or lbl.startswith('f'):
                            has_falc = True
                    if len(r_pts) > 0:
                        r_pts = np.array(r_pts)
                        r_peak_y = float(np.min(r_pts[:, 1]))
                    if len(s_pts) > 0:
                        s_pts = np.array(s_pts)
                        s_mean_y = float(np.mean(s_pts[:, 1]))
                except Exception:
                    pass

            # Heuristic guess: if Ridge is pulled into upper half of camera (peak Y < 0.35)
            # or peak Ridge is above mean Silhouette, frame is likely inverted/retracted
            is_heuristic_flipped = bool((r_peak_y < 0.35 or r_peak_y < s_mean_y) and has_ridge)

            current_label = existing_labels.get(stem, {}).get('status', 'unreviewed')

            frames.append({
                'id': stem,
                'filename': os.path.basename(img_path),
                'split': split,
                'patient': patient,
                'img_path': img_path,
                'json_path': json_path if os.path.exists(json_path) else None,
                'r_peak_y': round(r_peak_y, 3),
                'has_ridge': has_ridge,
                'has_sil': has_sil,
                'has_falc': has_falc,
                'heuristic_flipped': is_heuristic_flipped,
                'status': current_label  # 'flipped', 'normal', 'unsure', 'unreviewed'
            })

    return frames


def generate_overlay(img_path, json_path):
    """Draws colored landmark contours onto the image."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None
    H, W = bgr.shape[:2]

    overlay = bgr.copy()
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Colors (BGR)
            # Ridge: Green (#22c55e) -> (94, 197, 34)
            # Silhouette: Red (#ef4444) -> (68, 68, 239)
            # Falciform: Blue (#3b82f6) -> (246, 130, 59)
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
                    color = (246, 130, 59)  # Blue
                    thickness = 5
                else:
                    color = (200, 200, 200)
                    thickness = 3

                cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
                for pt in pts[::max(1, len(pts)//10)]:
                    cv2.circle(overlay, tuple(pt), 4, (255, 255, 255), -1)

            # Blend with 65% opacity
            cv2.addWeighted(overlay, 0.75, bgr, 0.25, 0, overlay)
        except Exception as e:
            print(f"Error rendering overlay: {e}")

    # Add watermark key legend in top-left
    cv2.putText(overlay, "Ridge (Green) | Silhouette (Red) | Falciform (Blue)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, "Ridge (Green) | Silhouette (Red) | Falciform (Blue)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # Encode to JPEG
    _, encoded = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return encoded.tobytes()


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Laparoscopic Liver Orientation Auditor</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .flash-green { animation: flashG 0.25s ease-out; }
    .flash-red { animation: flashR 0.25s ease-out; }
    @keyframes flashG { 0% { outline: 6px solid #22c55e; } 100% { outline: 0px solid transparent; } }
    @keyframes flashR { 0% { outline: 6px solid #ef4444; } 100% { outline: 0px solid transparent; } }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col antialiased select-none">

  <!-- Top Navigation & Statistics Bar -->
  <header class="bg-slate-900 border-b border-slate-800 px-6 py-4 sticky top-0 z-30 shadow-md">
    <div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
      <div>
        <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
          <span>🩺</span> Laparoscopic Liver Anatomical Orientation Auditor
        </h1>
        <p class="text-xs text-slate-400">Audit Train & Val datasets to establish empirical anatomical distribution of liver inversion/retraction.</p>
      </div>

      <!-- Quick Metrics Summary Pills -->
      <div class="flex items-center gap-3">
        <div class="bg-slate-800/80 border border-slate-700/80 px-3 py-1.5 rounded-lg text-center">
          <div class="text-[11px] uppercase tracking-wider text-slate-400 font-semibold">Total Audited</div>
          <div id="stat-audited" class="text-sm font-bold text-sky-400">0 / 0 (0%)</div>
        </div>
        <div class="bg-slate-800/80 border border-slate-700/80 px-3 py-1.5 rounded-lg text-center">
          <div class="text-[11px] uppercase tracking-wider text-slate-400 font-semibold">Train Flipped</div>
          <div id="stat-train-flipped" class="text-sm font-bold text-emerald-400">0% (0 / 921)</div>
        </div>
        <div class="bg-slate-800/80 border border-slate-700/80 px-3 py-1.5 rounded-lg text-center">
          <div class="text-[11px] uppercase tracking-wider text-slate-400 font-semibold">Val Flipped</div>
          <div id="stat-val-flipped" class="text-sm font-bold text-rose-400">0% (0 / 122)</div>
        </div>
        <div class="bg-slate-800/80 border border-slate-700/80 px-3 py-1.5 rounded-lg text-center">
          <div class="text-[11px] uppercase tracking-wider text-slate-400 font-semibold">Imbalance Ratio</div>
          <div id="stat-ratio" class="text-sm font-bold text-amber-400">1.0x</div>
        </div>
        <button onclick="exportCSV()" class="bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 border border-slate-600 px-3 py-2 rounded-lg font-medium transition flex items-center gap-1.5">
          <span>📥</span> Export CSV
        </button>
      </div>
    </div>
  </header>

  <!-- Filter & Controls Toolbar -->
  <section class="bg-slate-900/60 border-b border-slate-800/80 px-6 py-2.5">
    <div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3 text-xs">
      <div class="flex items-center gap-3">
        <label class="flex items-center gap-1.5 text-slate-400">
          Split:
          <select id="filter-split" onchange="applyFilters()" class="bg-slate-800 border border-slate-700 text-slate-200 rounded px-2.5 py-1 focus:outline-none">
            <option value="all">All Splits (1,043)</option>
            <option value="Train" selected>Train Only (921)</option>
            <option value="Val">Val Only (122)</option>
            <option value="Test">Test Only (109)</option>
          </select>
        </label>

        <label class="flex items-center gap-1.5 text-slate-400">
          Status:
          <select id="filter-status" onchange="applyFilters()" class="bg-slate-800 border border-slate-700 text-slate-200 rounded px-2.5 py-1 focus:outline-none">
            <option value="all">All Statuses</option>
            <option value="unreviewed">Unreviewed Only</option>
            <option value="flipped">Flipped Only</option>
            <option value="normal">Normal Only</option>
          </select>
        </label>

        <label class="flex items-center gap-1.5 text-slate-400">
          Patient:
          <select id="filter-patient" onchange="applyFilters()" class="bg-slate-800 border border-slate-700 text-slate-200 rounded px-2.5 py-1 focus:outline-none">
            <option value="all">All Patients</option>
          </select>
        </label>
      </div>

      <!-- Quick Action Utilities -->
      <div class="flex items-center gap-2">
        <button onclick="acceptHeuristicsForVisible()" class="bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-300 border border-indigo-500/40 px-2.5 py-1 rounded transition text-xs">
          ⚡ Apply Heuristic to Unreviewed
        </button>
        <span id="counter-label" class="text-slate-400 font-mono text-xs">Frame 1 of 0</span>
      </div>
    </div>
  </section>

  <!-- Main Dual-View Inspection Arena -->
  <main class="flex-1 max-w-7xl w-full mx-auto p-6 flex flex-col gap-6">

    <!-- View Mode Switcher & Metadata -->
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-3">
        <span id="badge-patient" class="bg-sky-500/10 text-sky-400 border border-sky-500/20 text-xs px-2.5 py-1 rounded font-semibold">Patient 00</span>
        <span id="badge-split" class="bg-slate-800 text-slate-300 border border-slate-700 text-xs px-2.5 py-1 rounded">Train</span>
        <span id="badge-status" class="bg-amber-500/10 text-amber-400 border border-amber-500/20 text-xs px-2.5 py-1 rounded">Unreviewed</span>
        <span id="badge-filename" class="text-slate-400 font-mono text-xs">filename.jpg</span>
      </div>

      <div class="flex items-center gap-2 text-xs">
        <span class="text-slate-400">View:</span>
        <button id="btn-view-split" onclick="setViewMode('split')" class="bg-sky-600 text-white px-3 py-1 rounded font-medium">Side-by-Side</button>
        <button id="btn-view-overlay" onclick="setViewMode('overlay')" class="bg-slate-800 text-slate-300 px-3 py-1 rounded font-medium hover:bg-slate-700">Overlay Only</button>
        <button id="btn-view-raw" onclick="setViewMode('raw')" class="bg-slate-800 text-slate-300 px-3 py-1 rounded font-medium hover:bg-slate-700">Raw Only</button>
      </div>
    </div>

    <!-- Image Viewing Grid -->
    <div id="image-container" class="grid grid-cols-1 md:grid-cols-2 gap-4 bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl relative min-h-[500px] items-center justify-center">
      
      <!-- Panel 1: Raw Image -->
      <div id="panel-raw" class="flex flex-col gap-2">
        <div class="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center justify-between">
          <span>1. Raw Laparoscopic Frame</span>
          <span class="text-[11px] text-slate-500">Unprocessed</span>
        </div>
        <div class="relative bg-black rounded-lg overflow-hidden border border-slate-800 aspect-video flex items-center justify-center">
          <img id="img-raw" src="" alt="Raw frame" class="w-full h-full object-contain">
        </div>
      </div>

      <!-- Panel 2: GT Landmark Overlay -->
      <div id="panel-overlay" class="flex flex-col gap-2">
        <div class="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center justify-between">
          <span>2. Landmark Overlay</span>
          <span class="text-[11px] text-emerald-400">Green: Ridge | Red: Sil | Blue: Falc</span>
        </div>
        <div class="relative bg-black rounded-lg overflow-hidden border border-slate-800 aspect-video flex items-center justify-center">
          <img id="img-overlay" src="" alt="Overlay frame" class="w-full h-full object-contain">
        </div>
      </div>

    </div>

    <!-- Ticking Decision Action Bar -->
    <div class="bg-slate-900/90 border border-slate-800 p-4 rounded-xl flex flex-wrap items-center justify-between gap-4 sticky bottom-4 z-20 shadow-2xl backdrop-blur-md">
      
      <!-- Navigation Buttons -->
      <div class="flex items-center gap-2">
        <button onclick="prevFrame()" class="bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 px-4 py-3 rounded-lg font-semibold text-sm transition flex items-center gap-2">
          <span>←</span> Previous (Left)
        </button>
        <button onclick="nextFrame()" class="bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 px-4 py-3 rounded-lg font-semibold text-sm transition flex items-center gap-2">
          Next (Right) <span>→</span>
        </button>
      </div>

      <!-- Core Decision Buttons (Large Clickable / Shortcut Key) -->
      <div class="flex items-center gap-3">
        <button id="btn-normal" onclick="setLabel('normal')" class="bg-emerald-600/20 hover:bg-emerald-600 text-emerald-300 hover:text-white border border-emerald-500/40 px-6 py-3 rounded-xl font-bold text-sm transition flex items-center gap-2 shadow-lg">
          <span class="bg-emerald-500/30 text-emerald-200 px-1.5 py-0.5 rounded text-xs">N</span>
          <span>Normal / Canonical Dome [0]</span>
        </button>

        <button id="btn-flipped" onclick="setLabel('flipped')" class="bg-rose-600/20 hover:bg-rose-600 text-rose-300 hover:text-white border border-rose-500/40 px-6 py-3 rounded-xl font-bold text-sm transition flex items-center gap-2 shadow-lg">
          <span class="bg-rose-500/30 text-rose-200 px-1.5 py-0.5 rounded text-xs">F</span>
          <span>Flipped / Retracted Flap [1]</span>
        </button>

        <button id="btn-unsure" onclick="setLabel('unsure')" class="bg-amber-600/20 hover:bg-amber-600 text-amber-300 hover:text-white border border-amber-500/40 px-4 py-3 rounded-xl font-semibold text-sm transition flex items-center gap-2">
          <span class="bg-amber-500/30 text-amber-200 px-1.5 py-0.5 rounded text-xs">U</span>
          <span>Unsure</span>
        </button>
      </div>

      <!-- Quick Tips -->
      <div class="text-slate-400 text-xs hidden lg:block text-right">
        <div><strong>Hotkeys:</strong> <kbd class="bg-slate-800 px-1.5 py-0.5 rounded">F</kbd> or <kbd class="bg-slate-800 px-1.5 py-0.5 rounded">1</kbd> for Flipped</div>
        <div><kbd class="bg-slate-800 px-1.5 py-0.5 rounded">N</kbd> or <kbd class="bg-slate-800 px-1.5 py-0.5 rounded">0</kbd> for Normal</div>
      </div>
    </div>

  </main>

  <script>
    let allFrames = [];
    let filteredIndices = [];
    let currentIndex = 0;
    let viewMode = 'split';

    async function loadData() {
      try {
        const res = await fetch('/api/frames');
        allFrames = await res.json();
        
        // Populate patient dropdown
        const patients = Array.from(new Set(allFrames.map(f => f.patient))).sort();
        const ptSelect = document.getElementById('filter-patient');
        patients.forEach(pt => {
          const opt = document.createElement('option');
          opt.value = pt;
          opt.textContent = pt;
          ptSelect.appendChild(opt);
        });

        applyFilters();
      } catch (e) {
        console.error("Failed to load frames:", e);
      }
    }

    function applyFilters() {
      const splitFilter = document.getElementById('filter-split').value;
      const statusFilter = document.getElementById('filter-status').value;
      const ptFilter = document.getElementById('filter-patient').value;

      filteredIndices = [];
      allFrames.forEach((frame, idx) => {
        if (splitFilter !== 'all' && frame.split !== splitFilter) return;
        if (statusFilter !== 'all' && frame.status !== statusFilter) return;
        if (ptFilter !== 'all' && frame.patient !== ptFilter) return;
        filteredIndices.push(idx);
      });

      if (filteredIndices.length === 0) {
        currentIndex = 0;
        showNoFrames();
      } else {
        currentIndex = 0;
        renderCurrentFrame();
      }
      updateStats();
    }

    function renderCurrentFrame() {
      if (filteredIndices.length === 0) return;
      const frameIdx = filteredIndices[currentIndex];
      const frame = allFrames[frameIdx];

      document.getElementById('counter-label').textContent = `Frame ${currentIndex + 1} of ${filteredIndices.length}`;
      document.getElementById('badge-patient').textContent = frame.patient;
      document.getElementById('badge-split').textContent = frame.split;
      document.getElementById('badge-filename').textContent = frame.filename;

      const badgeStatus = document.getElementById('badge-status');
      badgeStatus.textContent = frame.status.toUpperCase();
      badgeStatus.className = 'text-xs px-2.5 py-1 rounded font-semibold ';
      if (frame.status === 'flipped') {
        badgeStatus.className += 'bg-rose-500/20 text-rose-300 border border-rose-500/30';
      } else if (frame.status === 'normal') {
        badgeStatus.className += 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
      } else if (frame.status === 'unsure') {
        badgeStatus.className += 'bg-amber-500/20 text-amber-300 border border-amber-500/30';
      } else {
        badgeStatus.className += 'bg-slate-800 text-slate-400 border border-slate-700';
      }

      // Load images via API
      document.getElementById('img-raw').src = `/api/image?split=${frame.split}&file=${frame.filename}`;
      document.getElementById('img-overlay').src = `/api/overlay?split=${frame.split}&file=${frame.filename}`;
    }

    async function setLabel(label) {
      if (filteredIndices.length === 0) return;
      const frameIdx = filteredIndices[currentIndex];
      const frame = allFrames[frameIdx];
      frame.status = label;

      // Visual feedback
      const cont = document.getElementById('image-container');
      cont.classList.remove('flash-green', 'flash-red');
      void cont.offsetWidth; // trigger reflow
      cont.classList.add(label === 'flipped' ? 'flash-red' : 'flash-green');

      // Post to backend
      fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: frame.id, status: label, split: frame.split, patient: frame.patient })
      });

      updateStats();

      // Auto-advance
      if (currentIndex < filteredIndices.length - 1) {
        currentIndex++;
        renderCurrentFrame();
      }
    }

    function prevFrame() {
      if (currentIndex > 0) {
        currentIndex--;
        renderCurrentFrame();
      }
    }

    function nextFrame() {
      if (currentIndex < filteredIndices.length - 1) {
        currentIndex++;
        renderCurrentFrame();
      }
    }

    function setViewMode(mode) {
      viewMode = mode;
      const pRaw = document.getElementById('panel-raw');
      const pOverlay = document.getElementById('panel-overlay');
      const cont = document.getElementById('image-container');

      document.getElementById('btn-view-split').className = 'px-3 py-1 rounded font-medium ' + (mode === 'split' ? 'bg-sky-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700');
      document.getElementById('btn-view-overlay').className = 'px-3 py-1 rounded font-medium ' + (mode === 'overlay' ? 'bg-sky-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700');
      document.getElementById('btn-view-raw').className = 'px-3 py-1 rounded font-medium ' + (mode === 'raw' ? 'bg-sky-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700');

      if (mode === 'split') {
        cont.className = 'grid grid-cols-1 md:grid-cols-2 gap-4 bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl';
        pRaw.style.display = 'flex';
        pOverlay.style.display = 'flex';
      } else if (mode === 'overlay') {
        cont.className = 'grid grid-cols-1 gap-4 bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl';
        pRaw.style.display = 'none';
        pOverlay.style.display = 'flex';
      } else if (mode === 'raw') {
        cont.className = 'grid grid-cols-1 gap-4 bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl';
        pRaw.style.display = 'flex';
        pOverlay.style.display = 'none';
      }
    }

    function updateStats() {
      const audited = allFrames.filter(f => f.status !== 'unreviewed');
      const total = allFrames.length;
      document.getElementById('stat-audited').textContent = `${audited.length} / ${total} (${(audited.length/total*100).toFixed(1)}%)`;

      const trainFrames = allFrames.filter(f => f.split === 'Train');
      const trainFlipped = trainFrames.filter(f => f.status === 'flipped');
      const trainRatio = trainFrames.length ? (trainFlipped.length / trainFrames.length * 100) : 0;
      document.getElementById('stat-train-flipped').textContent = `${trainRatio.toFixed(1)}% (${trainFlipped.length} / ${trainFrames.length})`;

      const valFrames = allFrames.filter(f => f.split === 'Val');
      const valFlipped = valFrames.filter(f => f.status === 'flipped');
      const valRatio = valFrames.length ? (valFlipped.length / valFrames.length * 100) : 0;
      document.getElementById('stat-val-flipped').textContent = `${valRatio.toFixed(1)}% (${valFlipped.length} / ${valFrames.length})`;

      const disparity = (trainRatio > 0) ? (valRatio / trainRatio).toFixed(1) + 'x' : 'N/A';
      document.getElementById('stat-ratio').textContent = disparity;
    }

    function acceptHeuristicsForVisible() {
      if (!confirm("Apply automated geometric heuristic (Ridge Y < 0.35 -> Flipped) to all unreviewed frames in the current view?")) return;
      filteredIndices.forEach(idx => {
        const frame = allFrames[idx];
        if (frame.status === 'unreviewed') {
          frame.status = frame.heuristic_flipped ? 'flipped' : 'normal';
          fetch('/api/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: frame.id, status: frame.status, split: frame.split, patient: frame.patient })
          });
        }
      });
      renderCurrentFrame();
      updateStats();
    }

    function exportCSV() {
      window.open('/api/export_csv', '_blank');
    }

    // Keyboard Hotkeys
    window.addEventListener('keydown', (e) => {
      if (e.key === 'f' || e.key === 'F' || e.key === '1') {
        setLabel('flipped');
      } else if (e.key === 'n' || e.key === 'N' || e.key === '0') {
        setLabel('normal');
      } else if (e.key === 'u' || e.key === 'U') {
        setLabel('unsure');
      } else if (e.key === 'ArrowLeft') {
        prevFrame();
      } else if (e.key === 'ArrowRight') {
        nextFrame();
      }
    });

    loadData();
  </script>
</body>
</html>
"""


class AuditorHandler(BaseHTTPRequestHandler):
    frames_data = []

    def do_GET(self):
        url = urlparse(self.path)
        params = parse_qs(url.query)

        if url.path == '/' or url.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))

        elif url.path == '/api/frames':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(AuditorHandler.frames_data).encode('utf-8'))

        elif url.path == '/api/image':
            split = params.get('split', ['Train'])[0]
            filename = params.get('file', [''])[0]
            img_path = os.path.join(DATA_DIR, split, 'images', filename)
            if os.path.exists(img_path):
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Cache-Control', 'public, max-age=3600')
                self.end_headers()
                with open(img_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Image not found")

        elif url.path == '/api/overlay':
            split = params.get('split', ['Train'])[0]
            filename = params.get('file', [''])[0]
            stem = os.path.splitext(filename)[0]
            img_path = os.path.join(DATA_DIR, split, 'images', filename)
            json_path = os.path.join(DATA_DIR, split, 'labels', f"{stem}.json")

            if stem in OVERLAY_CACHE:
                overlay_bytes = OVERLAY_CACHE[stem]
            else:
                overlay_bytes = generate_overlay(img_path, json_path)
                if overlay_bytes:
                    OVERLAY_CACHE[stem] = overlay_bytes

            if overlay_bytes:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Cache-Control', 'public, max-age=3600')
                self.end_headers()
                self.wfile.write(overlay_bytes)
            else:
                self.send_error(404, "Overlay could not be rendered")

        elif url.path == '/api/export_csv':
            self.send_response(200)
            self.send_header('Content-Type', 'text/csv')
            self.send_header('Content-Disposition', 'attachment; filename="orientation_audit_results.csv"')
            self.end_headers()
            csv_lines = ["id,filename,patient,split,status,r_peak_y,heuristic_flipped\n"]
            for f in AuditorHandler.frames_data:
                csv_lines.append(f"{f['id']},{f['filename']},{f['patient']},{f['split']},{f['status']},{f['r_peak_y']},{f['heuristic_flipped']}\n")
            self.wfile.write("".join(csv_lines).encode('utf-8'))

        else:
            self.send_error(404)

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == '/api/save':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body)

            # Update memory
            frame_id = data.get('id')
            status = data.get('status')
            for f in AuditorHandler.frames_data:
                if f['id'] == frame_id:
                    f['status'] = status
                    break

            # Persist to disk
            save_dict = {}
            if os.path.exists(SAVE_FILE):
                try:
                    with open(SAVE_FILE, 'r') as sf:
                        save_dict = json.load(sf)
                except Exception:
                    pass
            save_dict[frame_id] = {
                'status': status,
                'patient': data.get('patient'),
                'split': data.get('split')
            }
            with open(SAVE_FILE, 'w') as sf:
                json.dump(save_dict, sf, indent=2)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Silence verbose request logging
        return


def run_server(port=8080):
    print("=" * 65)
    print("🚀 Initializing Laparoscopic Liver Orientation Auditor...")
    AuditorHandler.frames_data = scan_dataset()
    print(f"📦 Loaded {len(AuditorHandler.frames_data)} total frames from L3D dataset.")
    print(f"💾 Annotation save target: {SAVE_FILE}")
    print("=" * 65)
    print(f"🌐 Server running at: http://localhost:{port}")
    print("   Open this URL in your web browser to start auditing!")
    print("   Press Ctrl+C to terminate.")
    print("=" * 65)

    server = HTTPServer(('localhost', port), AuditorHandler)
    server.serve_forever()


if __name__ == '__main__':
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
