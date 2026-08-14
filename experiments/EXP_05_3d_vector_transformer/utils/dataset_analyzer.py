import os
import sys

# Add experiment directory to sys.path
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXP_DIR not in sys.path:
    sys.path.append(EXP_DIR)

import glob
import json
import numpy as np
from scipy.interpolate import CubicSpline

def analyze_dataset_gt_points(dataset_dir: str):
    """
    Scans raw JSON landmark annotations to analyze point count distributions
    and evaluate Spline Arc-Length Resampling fidelity across candidate K values.
    """
    print(f"🔍 Analyzing dataset ground-truth annotations in '{dataset_dir}'...")
    json_files = glob.glob(os.path.join(dataset_dir, "**", "*.json"), recursive=True)
    if not json_files:
        # Fallback search under labels/ directory
        json_files = glob.glob(os.path.join(dataset_dir, "labels", "*.json"))
        
    if not json_files:
        print(f"⚠️ No JSON annotation files found under '{dataset_dir}'. Returning default K=20.")
        return {"recommended_K": 20, "point_stats": {"min": 10, "median": 20, "max": 50}}

    curve_point_counts = []
    sample_curves = []

    for jpath in json_files:
        try:
            with open(jpath, 'r') as f:
                data = json.load(f)
            shapes = data.get('shapes', [])
            for shape in shapes:
                pts = np.array(shape.get('points', []), dtype=np.float32)
                if len(pts) >= 2:
                    curve_point_counts.append(len(pts))
                    if len(sample_curves) < 50:
                        sample_curves.append(pts)
        except Exception as e:
            continue

    if not curve_point_counts:
        print("⚠️ No valid shapes extracted from JSON files. Returning default K=20.")
        return {"recommended_K": 20, "point_stats": {"min": 10, "median": 20, "max": 50}}

    counts = np.array(curve_point_counts)
    min_pts, max_pts = int(np.min(counts)), int(np.max(counts))
    mean_pts, median_pts = float(np.mean(counts)), float(np.median(counts))

    print(f"📊 GT Curve Point Count Statistics across {len(counts)} curves:")
    print(f"   • Min: {min_pts} points | Max: {max_pts} points")
    print(f"   • Mean: {mean_pts:.1f} points | Median: {median_pts:.1f} points")

    # Measure Spline Approximation Fidelity for K in [10, 15, 20, 25, 30]
    candidate_Ks = [10, 15, 20, 25, 30]
    spline_errors = {}

    for K in candidate_Ks:
        errors = []
        for orig_pts in sample_curves:
            if len(orig_pts) < 2:
                continue
            # Calculate cumulative arc-length for original curve
            dists = np.sqrt(np.sum(np.diff(orig_pts, axis=0)**2, axis=1))
            cum_length = np.insert(np.cumsum(dists), 0, 0.0)
            total_len = cum_length[-1]
            if total_len < 1e-4:
                continue
            
            # Parametric cubic spline interpolation
            t_orig = cum_length / total_len
            t_resampled = np.linspace(0.0, 1.0, K)
            
            cs_x = CubicSpline(t_orig, orig_pts[:, 0])
            cs_y = CubicSpline(t_orig, orig_pts[:, 1])
            resampled_pts = np.stack([cs_x(t_resampled), cs_y(t_resampled)], axis=1)

            # Evaluate distance from original points to resampled spline curve
            # Mean Point-to-Curve Euclidean Distance
            diffs = resampled_pts[:, None, :] - orig_pts[None, :, :]
            dists_matrix = np.sqrt(np.sum(diffs**2, axis=-1))
            min_dists = np.min(dists_matrix, axis=1)
            errors.append(np.mean(min_dists))

        mean_error = float(np.mean(errors)) if errors else 0.0
        spline_errors[K] = mean_error
        print(f"   • Candidate K={K:2d}: Mean Spline Approximation Error = {mean_error:.3f} px")

    # Select smallest K where reconstruction error < 1.0 pixel
    recommended_K = 20
    for K in candidate_Ks:
        if spline_errors[K] <= 1.0:
            recommended_K = K
            break

    print(f"✅ Recommended optimal K selected: {recommended_K} points")
    return {
        "recommended_K": recommended_K,
        "point_stats": {"min": min_pts, "max": max_pts, "mean": mean_pts, "median": median_pts},
        "spline_errors": spline_errors
    }

if __name__ == "__main__":
    from configs.exp05_config import config
    analyze_dataset_gt_points(config.dataset_dir)
