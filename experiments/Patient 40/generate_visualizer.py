import os
import json
import pandas as pd

frames = [
    'Patient_40_08460', 'Patient_40_08490', 'Patient_40_08550', 'Patient_40_08610',
    'Patient_40_08670', 'Patient_40_08730', 'Patient_40_08790', 'Patient_40_08820',
    'Patient_40_08880', 'Patient_40_08940', 'Patient_40_09000', 'Patient_40_09060',
    'Patient_40_09120', 'Patient_40_09150', 'Patient_40_09210', 'Patient_40_09270',
    'Patient_40_09330', 'Patient_40_09390'
]

models = [
    {
        'id': 'm2f_baseline',
        'name': 'M2F: Baseline Control',
        'suite': 'EXP_2 (Mask2Former)',
        'suite_short': 'EXP_2',
        'badge_class': 'badge-m2f',
        'color': '#3b82f6',
        'csv': 'experiments/EXPERIMENT_2/results/EXPERIMENT_2_RESULTS_BASELINE/val_per_frame_results.csv',
        'img_dir_rel': 'EXPERIMENT_2/results/EXPERIMENT_2_RESULTS_BASELINE/patient_40_diagnostics',
        'desc': 'Full Mask2Former: Masked Cross-Attn + MSDeformAttn MultiScale + Query Self-Attn'
    },
    {
        'id': 'm2f_wo_masked_attn',
        'name': 'M2F: w/o Masked Attn',
        'suite': 'EXP_2 (Mask2Former)',
        'suite_short': 'EXP_2',
        'badge_class': 'badge-m2f',
        'color': '#6366f1',
        'csv': 'experiments/EXPERIMENT_2/results/EXPERIMENT_2_RESULTS_WO_MASKED_ATTN/val_per_frame_results.csv',
        'img_dir_rel': 'EXPERIMENT_2/results/EXPERIMENT_2_RESULTS_WO_MASKED_ATTN/patient_40_diagnostics',
        'desc': 'Spatial Gating Ablation: Full Global Cross-Attention (attn_mask = None)'
    },
    {
        'id': 'm2f_wo_multiscale',
        'name': 'M2F: w/o Multi-Scale',
        'suite': 'EXP_2 (Mask2Former)',
        'suite_short': 'EXP_2',
        'badge_class': 'badge-m2f',
        'color': '#ec4899',
        'csv': 'experiments/EXPERIMENT_2/results/EXPERIMENT_2_RESULTS_WO_MULTISCALE/val_per_frame_results.csv',
        'img_dir_rel': 'EXPERIMENT_2/results/EXPERIMENT_2_RESULTS_WO_MULTISCALE/patient_40_diagnostics',
        'desc': 'Scale Engine Ablation: Single-Scale Stride-16 Pixel Decoding'
    },
    {
        'id': 'm2f_wo_self_attn',
        'name': 'M2F: w/o Query Self-Attn',
        'suite': 'EXP_2 (Mask2Former)',
        'suite_short': 'EXP_2',
        'badge_class': 'badge-m2f',
        'color': '#8b5cf6',
        'csv': 'experiments/EXPERIMENT_2/results/EXPERIMENT_2_RESULTS_WO_SELF_ATTN/val_per_frame_results.csv',
        'img_dir_rel': 'EXPERIMENT_2/results/EXPERIMENT_2_RESULTS_WO_SELF_ATTN/patient_40_diagnostics',
        'desc': 'Query Interaction Ablation: Parallel Independent Decoding (No Q*Q^T)'
    },
    {
        'id': 'toponet_baseline',
        'name': 'TopoNet: Baseline (Dice Only)',
        'suite': 'EXP_1 (TopoNet)',
        'suite_short': 'EXP_1',
        'badge_class': 'badge-toponet',
        'color': '#10b981',
        'csv': 'experiments/EXPERIMENT_1/results/EXPERIMENT_1_RESULTS_BASELINE/working/results/run_baseline/validation_per_frame_results.csv',
        'img_dir_rel': 'EXPERIMENT_1/results/EXPERIMENT_1_RESULTS_BASELINE/working/results/run_baseline/patient_40_diagnostics',
        'desc': 'Standard CNN + Concat RGB-D Depth Fusion + Soft Dice loss only'
    },
    {
        'id': 'toponet_wo_btf',
        'name': 'TopoNet: w/o BTF',
        'suite': 'EXP_1 (TopoNet)',
        'suite_short': 'EXP_1',
        'badge_class': 'badge-toponet',
        'color': '#14b8a6',
        'csv': 'experiments/EXPERIMENT_1/results/EXPERIMENT_1_RESULTS_WO_BTF/run_wo_btf/validation_per_frame_results.csv',
        'img_dir_rel': 'EXPERIMENT_1/results/EXPERIMENT_1_RESULTS_WO_BTF/run_wo_btf/patient_40_diagnostics',
        'desc': 'Concat Fusion + Snake DSCNet + L_dice + L_cl + L_per'
    },
    {
        'id': 'toponet_wo_lcl',
        'name': 'TopoNet: w/o L_cl',
        'suite': 'EXP_1 (TopoNet)',
        'suite_short': 'EXP_1',
        'badge_class': 'badge-toponet',
        'color': '#06b6d4',
        'csv': 'experiments/EXPERIMENT_1/results/EXPERIMENT_1_RESULTS_WO_LCL/run_wo_lcl/validation_per_frame_results.csv',
        'img_dir_rel': 'EXPERIMENT_1/results/EXPERIMENT_1_RESULTS_WO_LCL/run_wo_lcl/patient_40_diagnostics',
        'desc': 'BTF + Snake DSCNet + L_dice + L_per (Betti matching persistence)'
    },
    {
        'id': 'toponet_wo_lper',
        'name': 'TopoNet: w/o L_per',
        'suite': 'EXP_1 (TopoNet)',
        'suite_short': 'EXP_1',
        'badge_class': 'badge-toponet',
        'color': '#f59e0b',
        'csv': 'experiments/EXPERIMENT_1/results/EXPERIMENT_1_RESULTS_WO_LPER/run_wo_lper/validation_per_frame_results.csv',
        'img_dir_rel': 'EXPERIMENT_1/results/EXPERIMENT_1_RESULTS_WO_LPER/run_wo_lper/patient_40_diagnostics',
        'desc': 'BTF + Snake DSCNet + L_dice + L_cl (clDice soft centerline skeletonization)'
    },
    {
        'id': 'toponet_wo_lper_lcl',
        'name': 'TopoNet: w/o L_per & L_cl',
        'suite': 'EXP_1 (TopoNet)',
        'suite_short': 'EXP_1',
        'badge_class': 'badge-toponet',
        'color': '#f97316',
        'csv': 'experiments/EXPERIMENT_1/results/EXPERIMENT_1_RESULTS_WO_LPER_LCL/run_wo_lper_lcl/validation_per_frame_results.csv',
        'img_dir_rel': 'EXPERIMENT_1/results/EXPERIMENT_1_RESULTS_WO_LPER_LCL/run_wo_lper_lcl/patient_40_diagnostics',
        'desc': 'BTF + Snake DSCNet + L_dice only (Dual topological losses removed)'
    }
]

# Collect per-frame and summary data
frame_filenames = [f + '.jpg' for f in frames]
model_metrics = {}
model_summary = []

for m in models:
    df = pd.read_csv(m['csv'])
    # filter to the 18 frames
    seg_df = df[df['filename'].isin(frame_filenames)].copy()
    seg_df.set_index('filename', inplace=True)
    
    per_frame = {}
    for f in frames:
        fn = f + '.jpg'
        if fn in seg_df.index:
            row = seg_df.loc[fn]
            per_frame[f] = {
                'macro_dice': round(float(row['macro_dice']) * 100, 2),
                'fg_dice': round(float(row['fg_dice']) * 100, 2),
                'macro_assd': round(float(row['macro_assd']), 2),
                'ridge_dice': round(float(row['ridge_dice']) * 100, 2),
                'sil_dice': round(float(row['sil_dice']) * 100, 2),
                'falc_dice': round(float(row['falc_dice']) * 100, 2),
                'macro_iou': round(float(row.get('macro_iou', 0)) * 100, 2),
                'fg_iou': round(float(row.get('fg_iou', 0)) * 100, 2)
            }
        else:
            per_frame[f] = {'macro_dice': 0, 'fg_dice': 0, 'macro_assd': 80, 'ridge_dice': 0, 'sil_dice': 0, 'falc_dice': 0}
            
    model_metrics[m['id']] = per_frame
    
    # Sequence summary
    avg_macro = seg_df['macro_dice'].mean() * 100
    avg_fg = seg_df['fg_dice'].mean() * 100
    avg_assd = seg_df['macro_assd'].mean()
    avg_ridge = seg_df['ridge_dice'].mean() * 100
    avg_sil = seg_df['sil_dice'].mean() * 100
    avg_falc = seg_df['falc_dice'].mean() * 100
    
    model_summary.append({
        'id': m['id'],
        'name': m['name'],
        'suite': m['suite_short'],
        'color': m['color'],
        'macro_dice': round(avg_macro, 2),
        'fg_dice': round(avg_fg, 2),
        'assd': round(avg_assd, 2),
        'ridge_dice': round(avg_ridge, 2),
        'sil_dice': round(avg_sil, 2),
        'falc_dice': round(avg_falc, 2)
    })

print("Metrics extracted successfully for 18 frames across 9 models.")

# Generate HTML for experiments/
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Patient 40 Multi-Model Benchmark Visualizer | Surgical AI</title>
<style>
  :root {{
    --bg-primary: #0a0f1d;
    --bg-secondary: #111827;
    --bg-card: #1f2937;
    --bg-card-hover: #263344;
    --text-primary: #f9fafb;
    --text-secondary: #9ca3af;
    --text-muted: #6b7280;
    --accent-blue: #3b82f6;
    --accent-indigo: #6366f1;
    --accent-emerald: #10b981;
    --accent-pink: #ec4899;
    --accent-amber: #f59e0b;
    --border-subtle: #374151;
    --border-active: #60a5fa;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.5;
    padding-bottom: 80px;
  }}
  header {{
    background: linear-gradient(180deg, #131d35 0%, var(--bg-secondary) 100%);
    border-bottom: 1px solid var(--border-subtle);
    padding: 24px 32px 16px 32px;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
  }}
  .header-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 16px;
  }}
  .title-group h1 {{
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .title-group p {{
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 4px;
  }}
  .legend-bar {{
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
    font-size: 0.8rem;
    background: rgba(0,0,0,0.3);
    padding: 8px 14px;
    border-radius: 8px;
    border: 1px solid var(--border-subtle);
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }}
  .controls-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,0.06);
  }}
  .scrubber-group {{
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .btn {{
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border-subtle);
    padding: 7px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 500;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  .btn:hover {{
    background: var(--bg-card-hover);
    border-color: var(--border-active);
  }}
  .btn.active {{
    background: var(--accent-blue);
    border-color: var(--accent-blue);
    color: #fff;
    box-shadow: 0 0 12px rgba(59, 130, 246, 0.4);
  }}
  .frame-indicator {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.92rem;
    font-weight: 600;
    background: #000;
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid var(--border-subtle);
    color: #60a5fa;
  }}
  .filter-group, .view-group {{
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,0,0,0.2);
    padding: 4px;
    border-radius: 8px;
    border: 1px solid var(--border-subtle);
  }}
  .filter-btn, .view-btn {{
    background: transparent;
    border: none;
    color: var(--text-secondary);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .filter-btn:hover, .view-btn:hover {{
    color: var(--text-primary);
  }}
  .filter-btn.active, .view-btn.active {{
    background: var(--bg-card);
    color: var(--text-primary);
    font-weight: 600;
    box-shadow: 0 1px 3px rgba(0,0,0,0.4);
  }}
  .filmstrip {{
    display: flex;
    gap: 8px;
    overflow-x: auto;
    padding: 12px 32px;
    background: #0c1222;
    border-bottom: 1px solid var(--border-subtle);
    scrollbar-width: thin;
  }}
  .filmstrip::-webkit-scrollbar {{
    height: 6px;
  }}
  .filmstrip::-webkit-scrollbar-thumb {{
    background: var(--border-subtle);
    border-radius: 3px;
  }}
  .thumb-pill {{
    flex: 0 0 auto;
    padding: 6px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    font-size: 0.75rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    cursor: pointer;
    color: var(--text-secondary);
    transition: all 0.15s ease;
  }}
  .thumb-pill:hover {{
    border-color: var(--border-active);
    color: var(--text-primary);
  }}
  .thumb-pill.active {{
    background: #1e3a8a;
    border-color: #3b82f6;
    color: #93c5fd;
    font-weight: 700;
  }}
  main {{
    padding: 24px 32px;
    max-width: 1920px;
    margin: 0 auto;
  }}
  .section-summary {{
    margin-bottom: 24px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px 20px;
  }}
  .summary-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    user-select: none;
  }}
  .summary-header h2 {{
    font-size: 1rem;
    font-weight: 600;
  }}
  .summary-content {{
    margin-top: 14px;
    display: block;
    overflow-x: auto;
  }}
  .summary-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    text-align: left;
  }}
  .summary-table th, .summary-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-subtle);
  }}
  .summary-table th {{
    color: var(--text-secondary);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    background: rgba(0,0,0,0.2);
  }}
  .summary-table tr:hover td {{
    background: rgba(255,255,255,0.03);
  }}
  .badge {{
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    display: inline-block;
  }}
  .badge-m2f {{ background: #1e3a8a; color: #93c5fd; }}
  .badge-toponet {{ background: #064e3b; color: #6ee7b7; }}
  
  .current-frame-stats {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
  }}
  .stat-card {{
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .stat-card-title {{
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.03em;
    display: flex;
    justify-content: space-between;
  }}
  .stat-card-value {{
    font-size: 1.3rem;
    font-weight: 700;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }}
  
  .cards-grid {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 24px;
  }}
  .model-card {{
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    overflow: hidden;
    transition: border-color 0.2s ease;
  }}
  .model-card:hover {{
    border-color: #4b5563;
  }}
  .card-header {{
    padding: 14px 20px;
    background: rgba(255,255,255,0.02);
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
  }}
  .card-title-wrap {{
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .card-title {{
    font-size: 1rem;
    font-weight: 600;
  }}
  .card-desc {{
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 2px;
  }}
  .metrics-strip {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }}
  .metric-badge {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.78rem;
    padding: 4px 9px;
    border-radius: 5px;
    background: #000;
    border: 1px solid var(--border-subtle);
  }}
  .metric-val-high {{ color: #4ade80; }}
  .metric-val-med {{ color: #fbbf24; }}
  .metric-val-low {{ color: #f87171; }}
  
  .image-stage {{
    padding: 16px 20px;
    background: #000;
    position: relative;
  }}
  .panel-labels {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    padding: 8px 12px;
    background: #0a0f1d;
    font-size: 0.75rem;
    color: var(--text-secondary);
    font-weight: 600;
    border-bottom: 1px solid var(--border-subtle);
    text-align: center;
  }}
  .image-wrapper {{
    position: relative;
    width: 100%;
    background: #000;
    border-radius: 6px;
    overflow: hidden;
  }}
  .full-montage-img {{
    width: 100%;
    height: auto;
    display: block;
    cursor: zoom-in;
    transition: opacity 0.2s ease;
  }}
  
  /* Cropped Focus Views */
  .crop-container {{
    position: relative;
    width: 100%;
    max-width: 700px;
    margin: 0 auto;
    aspect-ratio: 1/1;
    overflow: hidden;
    border-radius: 6px;
    border: 1px solid var(--border-subtle);
  }}
  .crop-container img {{
    position: absolute;
    width: 400%;
    height: 100%;
    top: 0;
    cursor: zoom-in;
  }}
  .crop-pred img {{
    left: -200%;
  }}
  .crop-error img {{
    left: -300%;
  }}
  .crop-gt img {{
    left: -100%;
  }}
  
  /* Side by Side Mode */
  .side-by-side-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 16px;
    max-width: 1200px;
    margin: 0 auto;
  }}
  .side-box {{
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    overflow: hidden;
    background: #0a0f1d;
  }}
  .side-box-title {{
    padding: 6px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-align: center;
    border-bottom: 1px solid var(--border-subtle);
  }}
  .side-box .crop-container {{
    max-width: 100%;
    border: none;
  }}
  
  /* Modal Lightbox */
  .lightbox {{
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0,0,0,0.92);
    z-index: 1000;
    justify-content: center;
    align-items: center;
    padding: 20px;
  }}
  .lightbox.active {{
    display: flex;
  }}
  .lightbox img {{
    max-width: 96vw;
    max-height: 92vh;
    border-radius: 6px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.8);
  }}
  .lightbox-close {{
    position: absolute;
    top: 20px;
    right: 30px;
    color: #fff;
    font-size: 32px;
    font-weight: bold;
    cursor: pointer;
  }}
</style>
</head>
<body>

<header>
  <div class="header-top">
    <div class="title-group">
      <h1>🩺 Surgical AI: Patient 40 Multi-Model Benchmark Visualizer</h1>
      <p>Continuous Anatomical Sequence: Frame <code>08460</code> to <code>09390</code> (18 Validation Keyframes) across Mask2Former (EXP_2) and TopoNet (EXP_1)</p>
    </div>
    <div class="legend-bar">
      <span style="font-weight:600; color:var(--text-secondary);">Landmarks:</span>
      <div class="legend-item"><span class="legend-dot" style="background:#22c55e;"></span> Ridge (Class 1)</div>
      <div class="legend-item"><span class="legend-dot" style="background:#ef4444;"></span> Silhouette (Class 2)</div>
      <div class="legend-item"><span class="legend-dot" style="background:#3b82f6;"></span> Falciform (Class 3)</div>
      <span style="border-left:1px solid var(--border-subtle); height:14px; margin:0 4px;"></span>
      <span style="font-weight:600; color:var(--text-secondary);">Error Map:</span>
      <div class="legend-item"><span class="legend-dot" style="background:#22c55e;"></span> True Positive (TP)</div>
      <div class="legend-item"><span class="legend-dot" style="background:#ef4444;"></span> False Positive (FP)</div>
      <div class="legend-item"><span class="legend-dot" style="background:#3b82f6;"></span> False Negative (FN)</div>
    </div>
  </div>

  <div class="controls-bar">
    <div class="scrubber-group">
      <button class="btn" id="btn-prev" title="Previous Frame (Left Arrow)">◀ Prev</button>
      <button class="btn" id="btn-play" title="Auto-play sequence (Spacebar)">▶ Play</button>
      <button class="btn" id="btn-next" title="Next Frame (Right Arrow)">Next ▶</button>
      <div class="frame-indicator" id="frame-indicator">Patient_40_08460.jpg (1/18)</div>
    </div>

    <div class="filter-group">
      <span style="font-size:0.75rem; color:var(--text-muted); padding-left:8px;">Filter:</span>
      <button class="filter-btn active" data-filter="all">All Models (9)</button>
      <button class="filter-btn" data-filter="exp2">Mask2Former Suite (4)</button>
      <button class="filter-btn" data-filter="exp1">TopoNet Suite (5)</button>
      <button class="filter-btn" data-filter="head2head">Baseline Head-to-Head (2)</button>
    </div>

    <div class="view-group">
      <span style="font-size:0.75rem; color:var(--text-muted); padding-left:8px;">View:</span>
      <button class="view-btn active" data-view="full">Full Montage (4-Panel)</button>
      <button class="view-btn" data-view="pred">Prediction Only</button>
      <button class="view-btn" data-view="gt_pred">GT vs Pred</button>
      <button class="view-btn" data-view="error">Error Map Only</button>
    </div>
  </div>
</header>

<div class="filmstrip" id="filmstrip">
  <!-- Filled dynamically by JavaScript -->
</div>

<main>
  <!-- Sequence Summary Accordion -->
  <div class="section-summary">
    <div class="summary-header" id="summary-toggle">
      <h2>📊 18-Frame Segment Quantitative Benchmark Summary (08460 → 09390)</h2>
      <span id="summary-icon" style="color:var(--text-secondary); font-size:0.85rem;">▼ Click to expand / collapse</span>
    </div>
    <div class="summary-content" id="summary-content">
      <table class="summary-table">
        <thead>
          <tr>
            <th>Model Architecture</th>
            <th>Suite</th>
            <th>Macro Dice</th>
            <th>FG Union Dice</th>
            <th>ASSD (px)</th>
            <th>Ridge DSC</th>
            <th>Silhouette DSC</th>
            <th>Falciform DSC</th>
          </tr>
        </thead>
        <tbody>
"""

# Append summary rows
for s in sorted(model_summary, key=lambda x: x['macro_dice'], reverse=True):
    badge = f"<span class='badge {'badge-m2f' if 'EXP_2' in s['suite'] else 'badge-toponet'}'>{s['suite']}</span>"
    html_content += f"""
          <tr>
            <td style="font-weight:600; color:{s['color']};">{s['name']}</td>
            <td>{badge}</td>
            <td style="font-weight:700;">{s['macro_dice']:.2f}%</td>
            <td>{s['fg_dice']:.2f}%</td>
            <td>{s['assd']:.2f} px</td>
            <td>{s['ridge_dice']:.2f}%</td>
            <td>{s['sil_dice']:.2f}%</td>
            <td>{s['falc_dice']:.2f}%</td>
          </tr>
    """

html_content += f"""
        </tbody>
      </table>
    </div>
  </div>

  <!-- Cards Grid -->
  <div class="cards-grid" id="cards-grid">
    <!-- Populated dynamically by JS for current frame -->
  </div>
</main>

<div class="lightbox" id="lightbox">
  <span class="lightbox-close" id="lightbox-close">&times;</span>
  <img id="lightbox-img" src="" alt="Zoomed view">
</div>

<script>
const FRAMES = {json.dumps(frames)};
const MODELS = {json.dumps(models)};
const MODEL_METRICS = {json.dumps(model_metrics)};

let currentFrameIdx = 0;
let currentFilter = 'all';
let currentView = 'full';
let isPlaying = false;
let playInterval = null;

// Initialize Filmstrip
const filmstripEl = document.getElementById('filmstrip');
FRAMES.forEach((f, idx) => {{
  const pill = document.createElement('div');
  pill.className = 'thumb-pill' + (idx === 0 ? ' active' : '');
  pill.textContent = f.replace('Patient_40_', '');
  pill.addEventListener('click', () => {{
    setFrame(idx);
  }});
  filmstripEl.appendChild(pill);
}});

function getValClass(val) {{
  if (val >= 65.0) return 'metric-val-high';
  if (val >= 45.0) return 'metric-val-med';
  return 'metric-val-low';
}}

function renderCards() {{
  const frameName = FRAMES[currentFrameIdx];
  const grid = document.getElementById('cards-grid');
  grid.innerHTML = '';

  document.getElementById('frame-indicator').textContent = `${{frameName}}.jpg (${{currentFrameIdx + 1}}/${{FRAMES.length}})`;

  // Update filmstrip active pill
  document.querySelectorAll('.thumb-pill').forEach((pill, idx) => {{
    pill.classList.toggle('active', idx === currentFrameIdx);
    if (idx === currentFrameIdx) {{
      pill.scrollIntoView({{ behavior: 'smooth', block: 'nearest', inline: 'center' }});
    }}
  }});

  let visibleModels = MODELS;
  if (currentFilter === 'exp2') {{
    visibleModels = MODELS.filter(m => m.suite_short === 'EXP_2');
  }} else if (currentFilter === 'exp1') {{
    visibleModels = MODELS.filter(m => m.suite_short === 'EXP_1');
  }} else if (currentFilter === 'head2head') {{
    visibleModels = MODELS.filter(m => m.id === 'm2f_baseline' || m.id === 'toponet_baseline');
  }}

  visibleModels.forEach(m => {{
    const metrics = MODEL_METRICS[m.id][frameName] || {{}};
    const card = document.createElement('div');
    card.className = 'model-card';

    const mDice = metrics.macro_dice || 0;
    const fgDice = metrics.fg_dice || 0;
    const assd = metrics.macro_assd || 0;
    const rDice = metrics.ridge_dice || 0;
    const sDice = metrics.sil_dice || 0;
    const fDice = metrics.falc_dice || 0;

    const imgRel = `${{m.img_dir_rel}}/${{frameName}}_diag.png`;

    let stageContent = '';
    if (currentView === 'full') {{
      stageContent = `
        <div class="panel-labels">
          <div>1. Input RGB Keyframe</div>
          <div>2. Ground Truth Landmarks</div>
          <div>3. Model Prediction</div>
          <div>4. Topological Error Map</div>
        </div>
        <div class="image-wrapper">
          <img src="${{imgRel}}" class="full-montage-img" alt="${{m.name}} ${{frameName}}" onclick="openLightbox('${{imgRel}}')">
        </div>
      `;
    }} else if (currentView === 'pred') {{
      stageContent = `
        <div class="crop-container crop-pred">
          <img src="${{imgRel}}" alt="Prediction" onclick="openLightbox('${{imgRel}}')">
        </div>
      `;
    }} else if (currentView === 'error') {{
      stageContent = `
        <div class="crop-container crop-error">
          <img src="${{imgRel}}" alt="Error Map" onclick="openLightbox('${{imgRel}}')">
        </div>
      `;
    }} else if (currentView === 'gt_pred') {{
      stageContent = `
        <div class="side-by-side-grid">
          <div class="side-box">
            <div class="side-box-title">Ground Truth Reference</div>
            <div class="crop-container crop-gt">
              <img src="${{imgRel}}" alt="Ground Truth" onclick="openLightbox('${{imgRel}}')">
            </div>
          </div>
          <div class="side-box">
            <div class="side-box-title">Model Prediction</div>
            <div class="crop-container crop-pred">
              <img src="${{imgRel}}" alt="Prediction" onclick="openLightbox('${{imgRel}}')">
            </div>
          </div>
        </div>
      `;
    }}

    card.innerHTML = `
      <div class="card-header">
        <div class="card-title-wrap">
          <span class="badge ${{m.badge_class}}">${{m.suite_short}}</span>
          <div>
            <div class="card-title" style="color:${{m.color}};">${{m.name}}</div>
            <div class="card-desc">${{m.desc}}</div>
          </div>
        </div>
        <div class="metrics-strip">
          <div class="metric-badge">Macro DSC: <span class="${{getValClass(mDice)}}">${{mDice.toFixed(1)}}%</span></div>
          <div class="metric-badge">FG DSC: <span class="${{getValClass(fgDice)}}">${{fgDice.toFixed(1)}}%</span></div>
          <div class="metric-badge">ASSD: <span>${{assd.toFixed(1)}} px</span></div>
          <div class="metric-badge" style="color:var(--text-muted);">
            R: <span style="color:#22c55e;">${{rDice.toFixed(0)}}%</span> |
            S: <span style="color:#ef4444;">${{sDice.toFixed(0)}}%</span> |
            F: <span style="color:#60a5fa;">${{fDice.toFixed(0)}}%</span>
          </div>
        </div>
      </div>
      <div class="image-stage">
        ${{stageContent}}
      </div>
    `;

    grid.appendChild(card);
  }});
}}

function setFrame(idx) {{
  currentFrameIdx = (idx + FRAMES.length) % FRAMES.length;
  renderCards();
}}

// Controls
document.getElementById('btn-prev').addEventListener('click', () => setFrame(currentFrameIdx - 1));
document.getElementById('btn-next').addEventListener('click', () => setFrame(currentFrameIdx + 1));

const playBtn = document.getElementById('btn-play');
playBtn.addEventListener('click', () => {{
  isPlaying = !isPlaying;
  if (isPlaying) {{
    playBtn.textContent = '⏸ Pause';
    playBtn.classList.add('active');
    playInterval = setInterval(() => setFrame(currentFrameIdx + 1), 1200);
  }} else {{
    playBtn.textContent = '▶ Play';
    playBtn.classList.remove('active');
    clearInterval(playInterval);
  }}
}});

// Filter Buttons
document.querySelectorAll('.filter-btn').forEach(btn => {{
  btn.addEventListener('click', (e) => {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    e.target.classList.add('active');
    currentFilter = e.target.dataset.filter;
    renderCards();
  }});
}});

// View Buttons
document.querySelectorAll('.view-btn').forEach(btn => {{
  btn.addEventListener('click', (e) => {{
    document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
    e.target.classList.add('active');
    currentView = e.target.dataset.view;
    renderCards();
  }});
}});

// Accordion toggle
const summaryToggle = document.getElementById('summary-toggle');
const summaryContent = document.getElementById('summary-content');
const summaryIcon = document.getElementById('summary-icon');
summaryToggle.addEventListener('click', () => {{
  const isHidden = summaryContent.style.display === 'none';
  summaryContent.style.display = isHidden ? 'block' : 'none';
  summaryIcon.textContent = isHidden ? '▼ Click to collapse' : '▲ Click to expand';
}});

// Lightbox
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightbox-img');
const lightboxClose = document.getElementById('lightbox-close');

function openLightbox(src) {{
  lightboxImg.src = src;
  lightbox.classList.add('active');
}}

lightboxClose.addEventListener('click', () => {{
  lightbox.classList.remove('active');
}});

lightbox.addEventListener('click', (e) => {{
  if (e.target === lightbox) lightbox.classList.remove('active');
}});

// Keyboard navigation
window.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {{
    setFrame(currentFrameIdx - 1);
  }} else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {{
    setFrame(currentFrameIdx + 1);
  }} else if (e.key === ' ') {{
    e.preventDefault();
    playBtn.click();
  }} else if (e.key === 'Escape') {{
    lightbox.classList.remove('active');
  }} else if (e.key === '1') {{
    document.querySelector('.view-btn[data-view="full"]').click();
  }} else if (e.key === '2') {{
    document.querySelector('.view-btn[data-view="pred"]').click();
  }} else if (e.key === '3') {{
    document.querySelector('.view-btn[data-view="gt_pred"]').click();
  }} else if (e.key === '4') {{
    document.querySelector('.view-btn[data-view="error"]').click();
  }}
}});

// Initial render
renderCards();
</script>
</body>
</html>
"""

# Save to experiments/
out_path_1 = "experiments/patient_40_diagnostics_comparison.html"
with open(out_path_1, "w") as f:
    f.write(html_content)

out_path_2 = "experiments/patient_40_comparison.html"
with open(out_path_2, "w") as f:
    f.write(html_content)

# Also generate a version for experiments/EXPERIMENT_2/ with '../' relative paths
html_content_exp2 = html_content.replace('EXPERIMENT_2/', '../EXPERIMENT_2/').replace('EXPERIMENT_1/', '../EXPERIMENT_1/')
out_path_3 = "experiments/EXPERIMENT_2/patient_40_diagnostics_comparison.html"
with open(out_path_3, "w") as f:
    f.write(html_content_exp2)

print(f"Generated HTML visualizers at:")
print(f"1. {out_path_1}")
print(f"2. {out_path_2}")
print(f"3. {out_path_3}")
