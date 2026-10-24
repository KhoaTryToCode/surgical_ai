"""
5-Fold Result Aggregator for MinVIS (Experiment 17).
Combines the 5 downloaded JSON result files from Kaggle / local runs into:
1. EXP17_5Fold_Summary.csv
2. EXP17_5Fold_Summary.md (Markdown table for paper/reports)
3. Console summary with Mean +/- Standard Deviation.
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate 5-Fold MinVIS Results")
    parser.add_argument('--results-dir', type=str, default='results', help='Directory containing results_fold_*.json files')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for CSV/MD (defaults to results-dir)')
    return parser.parse_args()


def load_fold_results(results_dir: str):
    val_files = sorted(glob.glob(os.path.join(results_dir, "results_fold_*.json")))
    test_files = sorted(glob.glob(os.path.join(results_dir, "test_results_fold_*.json")))

    if not val_files:
        raise FileNotFoundError(f"No 'results_fold_*.json' files found in '{results_dir}'")

    print(f"📊 Found {len(val_files)} validation fold files and {len(test_files)} frozen test files.")
    return val_files, test_files


def aggregate():
    args = parse_args()
    results_dir = args.results_dir
    out_dir = args.output_dir if args.output_dir else results_dir
    os.makedirs(out_dir, exist_ok=True)

    val_files, test_files = load_fold_results(results_dir)

    rows = []
    metric_keys = ['macro_dice', 'micro_dice', 'ridge_dice', 'silhouette_dice', 'ligament_dice', 'macro_assd', 'macro_hd95']

    for vf in val_files:
        with open(vf, 'r') as f:
            data = json.load(f)

        fold_idx = data.get('fold_idx', Path(vf).stem.replace('results_fold_', ''))
        best_m = data.get('best_val_metrics', {})
        row = {'Fold': f"Fold {fold_idx}"}
        for k in metric_keys:
            row[k] = best_m.get(k, np.nan)
        rows.append(row)

    df_val = pd.DataFrame(rows)

    # Compute Mean and Std across folds
    mean_series = {'Fold': 'Mean'}
    std_series = {'Fold': 'Std'}
    summary_series = {'Fold': 'Mean ± Std'}

    for k in metric_keys:
        vals = df_val[k].dropna().values
        m = float(np.mean(vals)) if len(vals) > 0 else 0.0
        s = float(np.std(vals)) if len(vals) > 0 else 0.0
        mean_series[k] = m
        std_series[k] = s
        summary_series[k] = f"{m:.2f} ± {s:.2f}"

    df_summary = pd.concat([df_val, pd.DataFrame([summary_series])], ignore_index=True)

    # Clean column names for CSV
    col_mapping = {
        'Fold': 'Split / Fold',
        'macro_dice': 'Macro Dice (%)',
        'micro_dice': 'Micro Dice (%)',
        'ridge_dice': 'Ridge Dice (%)',
        'silhouette_dice': 'Silhouette Dice (%)',
        'ligament_dice': 'Ligament Dice (%)',
        'macro_assd': 'Macro ASSD (px)',
        'macro_hd95': 'Macro HD95 (px)'
    }
    df_summary_export = df_summary.rename(columns=col_mapping)

    csv_path = os.path.join(out_dir, "EXP17_5Fold_Summary.csv")
    df_summary_export.to_csv(csv_path, index=False)

    # Frozen test aggregation if available
    test_rows = []
    if test_files:
        for tf in test_files:
            with open(tf, 'r') as f:
                tdata = json.load(f)
            t_idx = tdata.get('fold_idx', Path(tf).stem.replace('test_results_fold_', ''))
            tm = tdata.get('test_metrics', {})
            t_row = {'Model': f"Model Fold {t_idx}"}
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

        df_test_export = pd.concat([df_test, pd.DataFrame([test_summary])], ignore_index=True).rename(columns=col_mapping)
        test_csv_path = os.path.join(out_dir, "EXP17_FrozenTest_Summary.csv")
        df_test_export.to_csv(test_csv_path, index=False)

    def to_md_table(df_table):
        headers = list(df_table.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join([":---" if i == 0 else "---:" for i in range(len(headers))]) + " |"
        ]
        for _, row in df_table.iterrows():
            vals = [str(row[h]) for h in headers]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    # Build Markdown summary
    md_content = f"# Experiment 17 (MinVIS) — 5-Fold Cross-Validation Summary\n\n"
    md_content += f"### 1. Cross-Validation Results (35 Patients, 1,043 Frames)\n\n"
    md_content += to_md_table(df_summary_export)
    md_content += "\n\n"

    if test_files:
        md_content += f"### 2. Frozen Test Set Results (4 Patients, 109 Frames)\n\n"
        md_content += to_md_table(df_test_export)
        md_content += "\n\n"

    md_path = os.path.join(out_dir, "EXP17_5Fold_Summary.md")
    with open(md_path, 'w') as f:
        f.write(md_content)

    print("\n" + "=" * 90)
    print("                     EXPERIMENT 17: MINVIS 5-FOLD BENCHMARK SUMMARY")
    print("=" * 90)
    print(df_summary_export.to_string(index=False))
    print("=" * 90)

    if test_files:
        print("\n" + "=" * 90)
        print("                     FROZEN TEST SET (PATIENTS 21, 31, 41, 51)")
        print("=" * 90)
        print(df_test_export.to_string(index=False))
        print("=" * 90)

    print(f"\n✅ Aggregation complete! Saved to:\n  - '{csv_path}'\n  - '{md_path}'")


if __name__ == '__main__':
    aggregate()
