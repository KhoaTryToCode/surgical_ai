"""
Dataset Diagnostic & Inspection Script for Surgical AI (L3D Dataset).

Inspects raw JSON/XML annotations and surgical images to verify:
1. All unique label strings present in the dataset (validating LABEL_MAP)
2. Exact class distributions (Ridge, Silhouette, Ligament, Unknown)
3. Image resolutions vs annotation header dimensions (H, W)
4. Raw coordinate ranges [x_min, x_max, y_min, y_max] and spatial bounds
5. Polyline counts per image and raw point counts per polyline
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
    print("=" * 80)
    print(f"🔍 INSPECTING SURGICAL DATASET AT: {data_path}")
    print("=" * 80)

    splits = ['train', 'val', 'test']
    all_label_counts = Counter()
    unmapped_labels = Counter()
    class_counts = Counter()

    img_shapes = Counter()
    ann_shapes = Counter()
    shape_mismatches = 0

    polylines_per_img = []
    points_per_polyline = []

    x_mins, x_maxs = [], []
    y_mins, y_maxs = [], []

    total_images = 0
    total_annotations = 0

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

        if not os.path.exists(img_dir):
            print(f"⚠️ Warning: Directory not found: {img_dir}")
            continue

        img_files = sorted(glob.glob(os.path.join(img_dir, '*.[jJ][pP][gG]')) +
                           glob.glob(os.path.join(img_dir, '*.[pP][nN][gG]')))

        print(f"\n📂 Split [{split.upper()}]: Found {len(img_files)} images")
        total_images += len(img_files)

        for img_path in img_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(lbl_dir, base_name + '.json')
            xml_path = os.path.join(lbl_dir, base_name + '.xml')

            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Error reading image: {img_path}")
                continue

            orig_h, orig_w = img.shape[:2]
            img_shapes[(orig_h, orig_w)] += 1

            num_polylines_in_file = 0

            if os.path.exists(json_path):
                total_annotations += 1
                with open(json_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except Exception as e:
                        print(f"❌ Invalid JSON file: {json_path}")
                        continue

                ann_h = data.get('imageHeight', orig_h)
                ann_w = data.get('imageWidth', orig_w)
                ann_shapes[(ann_h, ann_w)] += 1

                if (ann_h, ann_w) != (orig_h, orig_w):
                    shape_mismatches += 1

                for shape in data.get('shapes', []):
                    raw_lbl = str(shape.get('label', ''))
                    all_label_counts[raw_lbl] += 1

                    cls = map_label(raw_lbl)
                    if cls == 0:
                        unmapped_labels[raw_lbl] += 1

                    class_counts[cls] += 1
                    pts = np.array(shape.get('points', []), dtype=np.float64)

                    if len(pts) > 0:
                        num_polylines_in_file += 1
                        points_per_polyline.append(len(pts))
                        x_mins.append(pts[:, 0].min())
                        x_maxs.append(pts[:, 0].max())
                        y_mins.append(pts[:, 1].min())
                        y_maxs.append(pts[:, 1].max())

            elif os.path.exists(xml_path):
                total_annotations += 1
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                except Exception as e:
                    print(f"❌ Invalid XML file: {xml_path}")
                    continue

                for contour in root.findall('contour'):
                    ctype = contour.find('contourType')
                    raw_lbl = ctype.text.strip() if ctype is not None else ''
                    all_label_counts[raw_lbl] += 1

                    cls = map_label(raw_lbl)
                    if cls == 0:
                        unmapped_labels[raw_lbl] += 1

                    class_counts[cls] += 1

                    x_elem = contour.find('imagePoints/x')
                    y_elem = contour.find('imagePoints/y')
                    if x_elem is not None and y_elem is not None:
                        xs = [float(x) for x in x_elem.text.split(',')]
                        ys = [float(y) for y in y_elem.text.split(',')]
                        pts = np.array(list(zip(xs, ys)), dtype=np.float64)

                        if len(pts) > 0:
                            num_polylines_in_file += 1
                            points_per_polyline.append(len(pts))
                            x_mins.append(pts[:, 0].min())
                            x_maxs.append(pts[:, 0].max())
                            y_mins.append(pts[:, 1].min())
                            y_maxs.append(pts[:, 1].max())

            polylines_per_img.append(num_polylines_in_file)

    print("\n" + "=" * 80)
    print("📊 DATASET DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"Total Images Analyzed: {total_images}")
    print(f"Total Annotation Files Found: {total_annotations}")

    print("\n🖼️ Image Resolutions Found (H, W):")
    for shape, count in img_shapes.items():
        print(f"   - {shape[0]}x{shape[1]}: {count} images")

    print("\nHeader Dimensions in Annotations (ann_h, ann_w):")
    for shape, count in ann_shapes.items():
        print(f"   - {shape[0]}x{shape[1]}: {count} annotation files")

    if shape_mismatches > 0:
        print(f"⚠️ Warning: {shape_mismatches} files have annotation header dimensions that mismatch actual image resolution!")
    else:
        print("✅ 100% Match between actual image resolutions and annotation header dimensions.")

    print("\n🏷️ All Raw Label Strings Found in Annotations:")
    for lbl, count in all_label_counts.most_common():
        cls_mapped = map_label(lbl)
        print(f"   - '{lbl}': {count} occurrences ──► Mapped to Class {cls_mapped}")

    if len(unmapped_labels) > 0:
        print(f"\n⚠️ WARNING: {len(unmapped_labels)} UNMAPPED label strings found:")
        for lbl, count in unmapped_labels.items():
            print(f"   - '{lbl}': {count} occurrences (Mapped to Class 0 - Background)")
    else:
        print("\n✅ 100% of label strings are correctly mapped to Classes {1: Ridge, 2: Silhouette, 3: Ligament}.")

    print("\n📈 Class Breakdown:")
    print(f"   - Class 1 (Ridge): {class_counts[1]} polylines")
    print(f"   - Class 2 (Silhouette): {class_counts[2]} polylines")
    print(f"   - Class 3 (Ligament): {class_counts[3]} polylines")
    print(f"   - Class 0 (Unknown/Unmapped): {class_counts[0]} polylines")

    if len(polylines_per_img) > 0:
        arr_polys = np.array(polylines_per_img)
        print("\n📐 Polyline Statistics per Image:")
        print(f"   - Min polylines per image: {arr_polys.min()}")
        print(f"   - Max polylines per image: {arr_polys.max()}")
        print(f"   - Mean polylines per image: {arr_polys.mean():.2f}")
        print(f"   - Images with 0 polylines: {(arr_polys == 0).sum()} ({((arr_polys == 0).sum() / len(arr_polys)) * 100:.1f}%)")

    if len(points_per_polyline) > 0:
        arr_pts = np.array(points_per_polyline)
        print("\n📌 Raw Point Statistics per Polyline:")
        print(f"   - Min points per polyline: {arr_pts.min()}")
        print(f"   - Max points per polyline: {arr_pts.max()}")
        print(f"   - Mean points per polyline: {arr_pts.mean():.2f}")

    if len(x_mins) > 0:
        print("\n📍 Coordinate Range Bounds (Pixel Space):")
        print(f"   - X Range: [{min(x_mins):.1f}, {max(x_maxs):.1f}]")
        print(f"   - Y Range: [{min(y_mins):.1f}, {max(y_maxs):.1f}]")

    print("\n" + "=" * 80)
    print("🎯 END OF DIAGNOSTIC REPORT")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect Surgical AI Dataset")
    parser.add_argument('--data_path', type=str, default='/kaggle/working/L3D', help='Path to dataset root')
    args = parser.parse_args()
    inspect_dataset(args.data_path)
