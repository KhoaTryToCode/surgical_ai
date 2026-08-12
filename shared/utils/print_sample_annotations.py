"""
Sample Annotation Printer Script for Surgical AI (L3D Dataset).

Prints out the raw formatted JSON and XML contents of sample annotation files
from train/labels/ and val/labels/ to inspect exact key structures and point values.
"""

import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET


def inspect_sample_files(data_path, num_samples=3):
    print("=" * 85)
    print(f"🔍 PRINTING SAMPLE ANNOTATION CONTENTS FROM: {data_path}")
    print("=" * 85)

    splits = ['train', 'val']

    for split in splits:
        split_dir = os.path.join(data_path, split)
        lbl_dir = os.path.join(split_dir, 'labels')

        print("\n" + "#" * 70)
        print(f"📂 SPLIT SAMPLES: [{split.upper()}]")
        print("#" * 70)

        if not os.path.exists(lbl_dir):
            print(f"⚠️ Warning: Labels directory not found: {lbl_dir}")
            continue

        json_files = sorted(glob.glob(os.path.join(lbl_dir, '*.json')))
        xml_files = sorted(glob.glob(os.path.join(lbl_dir, '*.xml')))

        print(f"Found {len(json_files)} JSON files and {len(xml_files)} XML files in {split}.")

        # ── Print Sample JSON Files ──
        if json_files:
            print(f"\n--- 📄 Sample JSON Files ({min(num_samples, len(json_files))} shown) ---")
            for j_path in json_files[:num_samples]:
                print(f"\n▶ File: {os.path.basename(j_path)}")
                try:
                    with open(j_path, 'r') as f:
                        data = json.load(f)

                    # Print top-level keys
                    print(f"   Top-level keys: {list(data.keys())}")
                    if 'imageHeight' in data and 'imageWidth' in data:
                        print(f"   Image Dimensions: Height={data['imageHeight']}, Width={data['imageWidth']}")

                    shapes = data.get('shapes', [])
                    print(f"   Shapes Count: {len(shapes)}")

                    for idx, shape in enumerate(shapes[:3]):
                        label = shape.get('label', '')
                        shape_type = shape.get('shape_type', 'N/A')
                        pts = shape.get('points', [])
                        print(f"   • Shape [{idx+1}]: label='{label}', type='{shape_type}', point_count={len(pts)}")
                        if len(pts) > 0:
                            print(f"     First 3 points: {pts[:3]}")
                            print(f"     Last point:     {pts[-1]}")
                except Exception as e:
                    print(f"   ❌ Error reading JSON: {e}")

        # ── Print Sample XML Files ──
        if xml_files:
            print(f"\n--- 📄 Sample XML Files ({min(num_samples, len(xml_files))} shown) ---")
            for x_path in xml_files[:num_samples]:
                print(f"\n▶ File: {os.path.basename(x_path)}")
                try:
                    tree = ET.parse(x_path)
                    root = tree.getroot()
                    print(f"   Root tag: {root.tag}")

                    contours = root.findall('contour')
                    print(f"   Contours Count: {len(contours)}")

                    for idx, contour in enumerate(contours[:3]):
                        ctype_elem = contour.find('contourType')
                        ctype = ctype_elem.text if ctype_elem is not None else 'N/A'
                        x_elem = contour.find('imagePoints/x')
                        y_elem = contour.find('imagePoints/y')

                        if x_elem is not None and y_elem is not None:
                            xs = x_elem.text.split(',')
                            ys = y_elem.text.split(',')
                            print(f"   • Contour [{idx+1}]: type='{ctype}', point_count={len(xs)}")
                            print(f"     First 3 points: {list(zip(xs[:3], ys[:3]))}")
                except Exception as e:
                    print(f"   ❌ Error reading XML: {e}")

    print("\n" + "=" * 85)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Print Sample Annotations")
    parser.add_argument('--data_path', type=str, default='/kaggle/working/L3D', help='Path to dataset root')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to print per split')
    args = parser.parse_args()
    inspect_sample_files(args.data_path, args.num_samples)
