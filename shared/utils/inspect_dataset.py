"""
Dataset Diagnostic & Inspection Script for Surgical AI (L3D Dataset).

Inspects raw JSON/XML annotations and surgical images across Train, Val, and Test splits:
1. Detailed per-split breakdown (Image count, Annotation count, Missing annotations)
2. All unique label strings present in each split (validating LABEL_MAP)
3. Exact class distributions (Ridge, Silhouette, Ligament, Unknown)
4. Image resolutions vs annotation header dimensions (H, W)
5. Raw coordinate ranges [x_min, x_max, y_min, y_max] and spatial bounds
6. Polyline counts per image and raw point counts per polyline
"""

import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import cv2
import numpy as np


def inspect_dataset(data_path):
    print("=" * 85)
    print(f"🔍 INSPECTING SURGICAL DATASET AT: {data_path}")
    print("=" * 85)

    splits = ['train', 'val', 'test']
    global_label_counts = Counter()
    global_unmapped_labels = Counter()
    global_class_counts = Counter()

    LABEL_MAP = {
        'ridge': 1, 'rigde': 1,
        'silhouette': 2, 'sil': 2,
        'ligament': 3, 'lig': 3,
    }

    def map_label(lbl_str):
        lbl = lbl_str.strip().lower()
        if lbl in LABEL_MAP:
            return LABEL_MAP[lbl]
        if lbl.startswith('r'):
            return 1
        if lbl.startswith('s'):
            return 2
        if lbl.startswith('l'):
            return 3
        return 0

    for split in splits:
        split_dir = os.path.join(data_path, split)
        img_dir = os.path.join(split_dir, 'images')
        lbl_dir = os.path.join(split_dir, 'labels')

        print("\n" + "-" * 70)
        print(f"📂 SPLIT ANALYSIS: [{split.upper()}]")
        print("-" * 70)

        if not os.path.exists(img_dir):
            print(f"⚠️ Warning: Directory not found: {img_dir}")
            continue

        img_files = sorted(glob.glob(os.path.join(img_dir, '*.[jJ][pP][gG]')) +
                           glob.glob(os.path.join(img_dir, '*.[pP][nN][gG]')))

        print(f"   • Image File Count: {len(img_files)}")

        split_label_counts = Counter()
        split_class_counts = Counter()
        split_img_shapes = Counter()
        split_ann_shapes = Counter()
        missing_ann_count = 0

        split_polylines_per_img = []
        split_pts_per_polyline = []

        split_x_mins, split_x_maxs = [], []
        split_y_mins, split_y_maxs = [], []

        for img_path in img_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(lbl_dir, base_name + '.json')
            xml_path = os.path.join(lbl_dir, base_name + '.xml')

            img = cv2.imread(img_path)
            if img is None:
                print(f"   ❌ Error reading image: {img_path}")
                continue

            orig_h, orig_w = img.shape[:2]
            split_img_shapes[(orig_h, orig_w)] += 1

            num_polylines_in_file = 0

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except Exception:
                        print(f"   ❌ Corrupted JSON: {json_path}")
                        continue

                ann_h = data.get('imageHeight', orig_h)
                ann_w = data.get('imageWidth', orig_w)
                split_ann_shapes[(ann_h, ann_w)] += 1

                for shape in data.get('shapes', []):
                    raw_lbl = str(shape.get('label', ''))
                    split_label_counts[raw_lbl] += 1
                    global_label_counts[raw_lbl] += 1

                    cls = map_label(raw_lbl)
                    if cls == 0:
                        global_unmapped_labels[raw_lbl] += 1

                    split_class_counts[cls] += 1
                    global_class_counts[cls] += 1

                    pts = np.array(shape.get('points', []), dtype=np.float64)
                    if len(pts) > 0:
                        num_polylines_in_file += 1
                        split_pts_per_polyline.append(len(pts))
                        split_x_mins.append(pts[:, 0].min())
                        split_x_maxs.append(pts[:, 0].max())
                        split_y_mins.append(pts[:, 1].min())
                        split_y_maxs.append(pts[:, 1].max())

            elif os.path.exists(xml_path):
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                except Exception:
                    print(f"   ❌ Corrupted XML: {xml_path}")
                    continue

                split_ann_shapes[(1080, 1920)] += 1
                for contour in root.findall('contour'):
                    ctype = contour.find('contourType')
                    raw_lbl = ctype.text.strip() if ctype is not None else ''
                    split_label_counts[raw_lbl] += 1
                    global_label_counts[raw_lbl] += 1

                    cls = map_label(raw_lbl)
                    if cls == 0:
                        global_unmapped_labels[raw_lbl] += 1

                    split_class_counts[cls] += 1
                    global_class_counts[cls] += 1

                    x_elem = contour.find('imagePoints/x')
                    y_elem = contour.find('imagePoints/y')
                    if x_elem is not None and y_elem is not None:
                        xs = [float(x) for x in x_elem.text.split(',')]
                        ys = [float(y) for y in y_elem.text.split(',')]
                        pts = np.array(list(zip(xs, ys)), dtype=np.float64)

                        if len(pts) > 0:
                            num_polylines_in_file += 1
                            split_pts_per_polyline.append(len(pts))
                            split_x_mins.append(pts[:, 0].min())
                            split_x_maxs.append(pts[:, 0].max())
                            split_y_mins.append(pts[:, 1].min())
                            split_y_maxs.append(pts[:, 1].max())
            else:
                missing_ann_count += 1

            split_polylines_per_img.append(num_polylines_in_file)

        print(f"   • Missing Annotation Files: {missing_ann_count}")
        print("   • Image Shapes (H, W):", dict(split_img_shapes))
        print("   • Annotation Shapes (H, W):", dict(split_ann_shapes))
        print("   • Class Distribution:", {
            'Class 1 (Ridge)': split_class_counts[1],
            'Class 2 (Silhouette)': split_class_counts[2],
            'Class 3 (Ligament)': split_class_counts[3],
            'Class 0 (Unmapped)': split_class_counts[0],
        })

        if split_polylines_per_img:
            arr_poly = np.array(split_polylines_per_img)
            print(f"   • Polylines per Image: Min={arr_poly.min()}, Max={arr_poly.max()}, Mean={arr_poly.mean():.2f}")
            print(f"   • Images with 0 Polylines: {(arr_poly == 0).sum()} ({(arr_poly == 0).mean()*100:.1f}%)")

        if split_x_mins:
            print(f"   • X Bounds: [{min(split_x_mins):.1f}, {max(split_x_maxs):.1f}]")
            print(f"   • Y Bounds: [{min(split_y_mins):.1f}, {max(split_y_maxs):.1f}]")

    print("\n" + "=" * 85)
    print("📊 OVERALL GLOBAL DATASET SUMMARY")
    print("=" * 85)
    print("🏷️ All Unique Label Strings Across All Splits:")
    for lbl, count in global_label_counts.most_common():
        print(f"   - '{lbl}': {count} occurrences ──► Mapped to Class {map_label(lbl)}")

    if global_unmapped_labels:
        print("\n⚠️ UNMAPPED LABELS FOUND:")
        for lbl, count in global_unmapped_labels.items():
            print(f"   - '{lbl}': {count} occurrences")
    else:
        print("\n✅ 100% of raw label strings are properly mapped to classes {1, 2, 3}.")

    print("=" * 85)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect Surgical AI Dataset")
    parser.add_argument('--data_path', type=str, default='/kaggle/working/L3D', help='Path to dataset root')
    args = parser.parse_args()
    inspect_dataset(args.data_path)
