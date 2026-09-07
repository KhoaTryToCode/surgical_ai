#!/usr/bin/env python3
"""
Side-by-Side Model Comparison: Mask2Former vs. Patch-Bézier ViT on Validation Split.

Pairs and stitches all 122 validation images from:
  - Model A: Mask2Former (Dense Pixel-Wise Segmentation)
  - Model B: Patch-Bézier ViT (Continuous Vector Curves)

Output:
  - data/comparisons_m2f_vs_vit/compare_sample_XXX_Patient_YY_ZZZZZ.png
  - High-resolution 2x4 visual comparison layout with aligned columns:
      Col 1: RGB Input Surgical Frame
      Col 2: Ground Truth (Dense vs. Vector Curves)
      Col 3: Model Prediction (Dense Mask vs. Bézier Curves)
      Col 4: Error / Alignment Map
"""

import os
import glob
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm


def main():
    m2f_dir = "data/mask2former_122_val/working/results_rgbd/worst_cases"
    vit_dir = "data/vit_base_122_val/val_visualizations_base"
    csv_path = os.path.join(m2f_dir, "validation_worst_to_best_summary.csv")
    out_dir = "data/comparisons_m2f_vs_vit"

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("🔬 GENERATING SIDE-BY-SIDE COMPARISON: MASK2FORMER vs. PATCH-BÉZIER ViT")
    print(f"   Mask2Former Dir: {m2f_dir}")
    print(f"   ViT Dir:         {vit_dir}")
    print(f"   Output Dir:      {out_dir}")
    print("=" * 80)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV metadata not found at '{csv_path}'")

    df = pd.read_csv(csv_path)
    # Sort filenames alphabetically to match vit dataset indexing
    sorted_filenames = sorted(df["Filename"].tolist())

    # Map stems to metadata
    meta_map = {}
    for _, row in df.iterrows():
        stem = row["Filename"].replace(".jpg", "").replace(".png", "")
        meta_map[stem] = row

    # Map stems to m2f visualization files
    m2f_files = glob.glob(os.path.join(m2f_dir, "visual_diagnostics", "rank*.png"))
    m2f_file_map = {}
    for f in m2f_files:
        parts = os.path.basename(f).split("_")
        stem = "_".join(parts[2:]).replace(".png", "")
        m2f_file_map[stem] = f

    print(f"📊 Found {len(sorted_filenames)} validation samples. Processing all 122 pairs...\n")

    for i in tqdm(range(1, 123), desc="Stitching comparisons"):
        stem = sorted_filenames[i - 1].replace(".jpg", "").replace(".png", "")
        vit_path = os.path.join(vit_dir, f"val_prediction_sample_{i:02d}.png")
        m2f_path = m2f_file_map.get(stem, None)

        if not m2f_path or not os.path.exists(m2f_path):
            print(f"⚠️ Warning: M2F file for '{stem}' not found.")
            continue
        if not os.path.exists(vit_path):
            print(f"⚠️ Warning: ViT file '{vit_path}' not found.")
            continue

        row_meta = meta_map[stem]
        m2f_dice = row_meta["Dice"]
        m2f_rank = int(row_meta["Rank"])
        patient_id = row_meta["Patient"]
        diagnosis = str(row_meta.get("Diagnosis", ""))

        m2f_img = cv2.imread(m2f_path)
        vit_img = cv2.imread(vit_path)

        # ── 1. Extract Mask2Former 4 Panels (from 2x2 layout) ──
        # M2F header is top 55px, canvas is 2048x2048
        p1 = cv2.resize(m2f_img[55:1079, 0:1024], (512, 512), interpolation=cv2.INTER_AREA)
        p2 = cv2.resize(m2f_img[55:1079, 1024:2048], (512, 512), interpolation=cv2.INTER_AREA)
        p3 = cv2.resize(m2f_img[1079:2103, 0:1024], (512, 512), interpolation=cv2.INTER_AREA)
        p4 = cv2.resize(m2f_img[1079:2103, 1024:2048], (512, 512), interpolation=cv2.INTER_AREA)
        row_m2f = np.hstack([p1, p2, p3, p4])

        # ── 2. Extract Patch-Bézier ViT 4 Panels (from 1x4 layout) ──
        step = vit_img.shape[1] // 4
        v1 = cv2.resize(vit_img[:, 0 * step:1 * step], (512, 512), interpolation=cv2.INTER_AREA)
        v2 = cv2.resize(vit_img[:, 1 * step:2 * step], (512, 512), interpolation=cv2.INTER_AREA)
        v3 = cv2.resize(vit_img[:, 2 * step:3 * step], (512, 512), interpolation=cv2.INTER_AREA)
        v4 = cv2.resize(vit_img[:, 3 * step:4 * step], (512, 512), interpolation=cv2.INTER_AREA)
        row_vit = np.hstack([v1, v2, v3, v4])

        # ── 3. Construct Section Banners ──
        banner_h = 46
        # M2F banner
        banner_m2f = np.full((banner_h, 2048, 3), (35, 20, 20), dtype=np.uint8)
        cv2.putText(
            banner_m2f,
            f"MODEL A: MASK2FORMER (Dense Pixel-Wise Segmentation) | Dice: {m2f_dice:.4f} (Rank #{m2f_rank}/122)",
            (20, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (120, 210, 255), 2, cv2.LINE_AA
        )
        if diagnosis:
            diag_str = f"Diagnosis: {diagnosis[:45]}"
            cv2.putText(
                banner_m2f, diag_str, (1350, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 255), 1, cv2.LINE_AA
            )

        # ViT banner
        banner_vit = np.full((banner_h, 2048, 3), (20, 35, 20), dtype=np.uint8)
        cv2.putText(
            banner_vit,
            "MODEL B: PATCH-BEZIER ViT (Continuous Vector Spline / Bezier Representation)",
            (20, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (120, 255, 120), 2, cv2.LINE_AA
        )

        # ── 4. Main Title Header ──
        top_h = 56
        top_header = np.full((top_h, 2048, 3), (12, 12, 12), dtype=np.uint8)
        title_text = (
            f"SIDE-BY-SIDE EVALUATION: {stem} | Sample #{i:03d} / 122 | "
            f"Patient: {patient_id} | [Left-to-Right: 1. Input | 2. GT | 3. Prediction | 4. Error/Alignment]"
        )
        cv2.putText(top_header, title_text, (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        # ── 5. Assemble Canvas ──
        canvas = np.vstack([top_header, banner_m2f, row_m2f, banner_vit, row_vit])

        out_name = f"compare_sample_{i:03d}_{stem}.png"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, canvas)

    print(f"\n🎉 Successfully saved all 122 side-by-side comparison images to: '{out_dir}'")


if __name__ == "__main__":
    main()
