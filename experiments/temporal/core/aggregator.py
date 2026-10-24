"""
Master Leaderboard Aggregator for experiments/temporal.
Scans results/ across all models (MinVIS, Cutie, SAM 2), computes 5-fold Mean +/- Std,
and compiles a unified Master Benchmark Comparison Table (CSV and Markdown).
"""

import os
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compile Global Temporal Video Benchmark Leaderboard")
    parser.add_argument('--results-dir', type=str, default=None, help='Root results directory')
    return parser.parse_args()


def to_md_table(df_table: pd.DataFrame) -> str:
    headers = list(df_table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---" if i == 0 else "---:" for i in range(len(headers))]) + " |"
    ]
    for _, row in df_table.iterrows():
        vals = [str(row[h]) for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def aggregate_model_results(model_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    val_files = sorted(glob.glob(os.path.join(model_dir, "results_fold_*.json")))
    test_files = sorted(glob.glob(os.path.join(model_dir, "test_results_fold_*.json")))

    if not val_files:
        return None, None, None

    metric_keys = ['macro_dice', 'micro_dice', 'ridge_dice', 'silhouette_dice', 'ligament_dice', 'macro_assd', 'macro_hd95']
    val_rows = []

    for vf in val_files:
        with open(vf, 'r') as f:
            d = json.load(f)
        f_idx = d.get('fold_idx', Path(vf).stem.replace('results_fold_', ''))
        bm = d.get('best_val_metrics', {})
        row = {'Fold': f"Fold {f_idx}"}
        for k in metric_keys:
            row[k] = bm.get(k, np.nan)
        val_rows.append(row)

    df_val = pd.DataFrame(val_rows)
    val_summary = {'Fold': 'Mean ± Std'}
    summary_stats = {}

    for k in metric_keys:
        vals = df_val[k].dropna().values
        m = float(np.mean(vals)) if len(vals) > 0 else 0.0
        s = float(np.std(vals)) if len(vals) > 0 else 0.0
        val_summary[k] = f"{m:.2f} ± {s:.2f}"
        summary_stats[f'val_{k}_mean'] = m
        summary_stats[f'val_{k}_std'] = s

    df_val_export = pd.concat([df_val, pd.DataFrame([val_summary])], ignore_index=True)

    # Test set processing
    df_test_export = None
    if test_files:
        test_rows = []
        for tf in test_files:
            with open(tf, 'r') as f:
                td = json.load(f)
            t_idx = td.get('fold_idx', Path(tf).stem.replace('test_results_fold_', ''))
            tm = td.get('test_metrics', {})
            t_row = {'Model': f"Fold {t_idx} Weights"}
            for k in metric_keys:
                t_row[k] = tm.get(k, np.nan)
            test_rows.append(t_row)

        df_test = pd.DataFrame(test_rows)
        test_summary = {'Model': 'Frozen Test (Mean ± Std)'}
        for k in metric_keys:
            tvals = df_test[k].dropna().values
            tm_val = float(np.mean(tvals)) if len(tvals) > 0 else 0.0
            ts_val = float(np.std(tvals)) if len(tvals) > 0 else 0.0
            test_summary[k] = f"{tm_val:.2f} ± {ts_val:.2f}"
            summary_stats[f'test_{k}_mean'] = tm_val
            summary_stats[f'test_{k}_std'] = ts_val

        df_test_export = pd.concat([df_test, pd.DataFrame([test_summary])], ignore_index=True)

    return df_val_export, df_test_export, summary_stats


def compile_leaderboard(results_root: str):
    root_path = Path(results_root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    print(f"🔎 Scanning results directory: '{root_path}'...")

    # Look for model subdirectories (e.g. results/minvis/, results/cutie/) or results directly in root
    subdirs = [d for d in root_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    model_targets = {}

    if subdirs and any(glob.glob(str(subdirs[0] / "results_fold_*.json")) for d in subdirs):
        for sd in subdirs:
            if glob.glob(str(sd / "results_fold_*.json")):
                model_targets[sd.name] = str(sd)
    elif glob.glob(str(root_path / "results_fold_*.json")):
        model_targets['minvis'] = str(root_path)

    if not model_targets:
        print(f"⚠️ No completed fold runs found in '{root_path}'. Waiting for experiment outputs.")
        return

    print(f"📊 Discovered {len(model_targets)} model results: {list(model_targets.keys())}")

    leaderboard_rows = []
    detailed_reports = {}

    for model_name, m_dir in model_targets.items():
        df_val, df_test, stats = aggregate_model_results(m_dir)
        if stats is None:
            continue

        detailed_reports[model_name] = (df_val, df_test)

        row = {
            'Model Architecture': model_name.upper(),
            'Val Macro Dice (%)': f"{stats.get('val_macro_dice_mean', 0.0):.2f} ± {stats.get('val_macro_dice_std', 0.0):.2f}",
            'Val Micro Dice (%)': f"{stats.get('val_micro_dice_mean', 0.0):.2f} ± {stats.get('val_micro_dice_std', 0.0):.2f}",
            'Val ASSD (px)': f"{stats.get('val_macro_assd_mean', 0.0):.2f} ± {stats.get('val_macro_assd_std', 0.0):.2f}",
            'Val HD95 (px)': f"{stats.get('val_macro_hd95_mean', 0.0):.2f} ± {stats.get('val_macro_hd95_std', 0.0):.2f}",
            'Test Macro Dice (%)': f"{stats.get('test_macro_dice_mean', 0.0):.2f} ± {stats.get('test_macro_dice_std', 0.0):.2f}",
            'Test Micro Dice (%)': f"{stats.get('test_micro_dice_mean', 0.0):.2f} ± {stats.get('test_micro_dice_std', 0.0):.2f}",
            'Test ASSD (px)': f"{stats.get('test_macro_assd_mean', 0.0):.2f} ± {stats.get('test_macro_assd_std', 0.0):.2f}",
        }
        leaderboard_rows.append(row)

    df_leaderboard = pd.DataFrame(leaderboard_rows)

    # Save Global Leaderboard CSV & Markdown
    csv_out = root_path / "LEADERBOARD.csv"
    md_out = root_path / "LEADERBOARD.md"

    df_leaderboard.to_csv(csv_out, index=False)

    md_content = "# Temporal Video Landmark Detection — Master Benchmark Leaderboard\n\n"
    md_content += "### 5-Fold Cross-Validation & Frozen Test Set (Patients 21, 31, 41, 51)\n\n"
    md_content += to_md_table(df_leaderboard)
    md_content += "\n\n---\n\n"

    # Append per-model breakdowns
    for m_name, (df_v, df_t) in detailed_reports.items():
        md_content += f"## Detailed Breakdown: {m_name.upper()}\n\n"
        if df_v is not None:
            md_content += "#### 5-Fold Validation Splits\n\n"
            md_content += to_md_table(df_v)
            md_content += "\n\n"
        if df_t is not None:
            md_content += "#### Frozen Test Set Evaluation\n\n"
            md_content += to_md_table(df_t)
            md_content += "\n\n"

    with open(md_out, 'w') as f:
        f.write(md_content)

    print("\n" + "=" * 100)
    print("               TEMPORAL VIDEO BENCHMARK LEADERBOARD (CROSS-MODEL COMPARISON)")
    print("=" * 100)
    print(df_leaderboard.to_string(index=False))
    print("=" * 100)
    print(f"\n✅ Leaderboard compiled successfully!\n  - CSV: '{csv_out}'\n  - MD:  '{md_out}'")


if __name__ == '__main__':
    args = parse_args()
    res_dir = args.results_dir if args.results_dir else str(Path(__file__).resolve().parent.parent / "results")
    compile_leaderboard(res_dir)
