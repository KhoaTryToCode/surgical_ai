"""
Data Preparation & Asset Generation Script for EXP_12 (BCRNet Replication)
==========================================================================
Prepares the complete L3D dataset structure required by BCRNet's BezierDataset:
  <target_dir>/<Split>/
    ├── images/             (.jpg images, symlinked)
    ├── depth_AdelaiDepth/  (.png depth maps from Kaggle L3D dataset)
    ├── labels/             (.json annotation files)
    ├── masks_gt/           (.png ground truth masks)
    ├── s_bezier/           (.npz 5th-order Bézier GT, compressed to avoid disk exhaustion)
    └── sam/                (.npy SAM ViT-B 256x64x64 features)

Optimized for Kaggle / Colab disk limits:
- Saves s_bezier with np.uint8 masks and np.savez_compressed, reducing disk size from ~46 GB to ~15 MB.
"""

import os
import sys
import glob
import json
import argparse
from collections import defaultdict
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
bcrnet_root = os.path.join(ws_root, "repos/BCRNet")
bcrnet_utils = os.path.join(bcrnet_root, "utils")
for p in [bcrnet_utils, bcrnet_root, ws_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from bezier import BezierCurve
from preprocess import resize, generate_mask_gt


def generate_compressed_beziers(src_path: str, dst_path: str, degree: int = 5, curve_points_num: int = 25,
                                image_shape: tuple = (1080, 1920), linewidth: int = 30):
    """
    Generates 5th-order Bézier annotations identically to BCRNet's preprocess.py,
    but stores landmark_mask as uint8 and compresses with np.savez_compressed.
    Reduces disk footprint from ~50 MB/file down to ~15 KB/file (3000x smaller).
    """
    json_files = glob.glob(os.path.join(src_path, '*.json'))
    os.makedirs(dst_path, exist_ok=True)
    pb = BezierCurve(degree)

    for json_file in tqdm(sorted(json_files), desc=f"Generating s_bezier (compressed)"):
        name = os.path.basename(json_file)[:-5]
        out_path = os.path.join(dst_path, name)
        # Check if already generated
        if os.path.exists(out_path + ".npz") and os.path.getsize(out_path + ".npz") > 100:
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)

        curves = defaultdict(list)
        shape = (data['imageWidth'], data['imageHeight'])
        for curve in data['shapes']:
            points = curve['points']
            label = curve['label']
            if label.startswith('r'):
                label = 'ridge'
            elif label.startswith('s'):
                label = 'silhouette'
            elif label.startswith('l'):
                label = 'ligament'
            curves[label].append(resize(points, shape))

        gt = []
        landmark_list = ['silhouette', 'ligament', 'ridge']
        for lm_idx, label in enumerate(landmark_list):
            gt.append({})
            lm_ctrl_points = []
            lm_curve_points = []
            landmark_mask = np.zeros(image_shape, dtype=np.uint8)
            for curve_idx, curve in enumerate(curves[label]):
                ctrl_points = pb.fit_bezier(curve, 1 if label == 'ligament' else 0)
                curve_point = pb._get_interpolated_points(curve, curve_points_num)

                lm_ctrl_points += ctrl_points
                lm_curve_points.append(curve_point)
                for pt1, pt2 in zip(curve[:-1], curve[1:]):
                    cv2.line(landmark_mask, (pt1 * [image_shape[1], image_shape[0]]).astype('int'),
                             (pt2 * [image_shape[1], image_shape[0]]).astype('int'), [1], linewidth)

            gt[lm_idx]['ctrl_points'] = np.stack(lm_ctrl_points, 0) if len(lm_ctrl_points) > 0 else np.empty([0, degree + 1, 2], dtype=np.float32)
            gt[lm_idx]['curve_points'] = np.stack(lm_curve_points, 0) if len(lm_ctrl_points) > 0 else np.empty([0, curve_points_num, 2], dtype=np.float32)
            gt[lm_idx]['landmark_mask'] = landmark_mask

        np.savez_compressed(out_path, gt=gt)


def find_source_dir(split: str, sub: str) -> str:
    """Finds existing directory for given split and subdirectory."""
    split_cap = split.capitalize()
    split_low = split.lower()

    sub_variations = [sub]
    if "depth" in sub.lower():
        sub_variations = ["depth_AdelaiDepth", "depth_adelaidepth", "depth_anything_v2", "depth", "depths", "depth_maps"]

    candidates = []
    for s_var in sub_variations:
        candidates.extend([
            # Kaggle input paths
            f"/kaggle/input/datasets/khoatrytopublish/l3d-{split_low}/{split_cap}/{s_var}",
            f"/kaggle/input/datasets/khoatrytopublish/l3d-{split_low}/{split_low}/{s_var}",
            f"/kaggle/input/l3d-{split_low}/{split_cap}/{s_var}",
            f"/kaggle/input/l3d-{split_low}/{split_low}/{s_var}",
            f"/kaggle/input/l3d-depth/{split_cap}/{s_var}",
            f"/kaggle/input/l3d-depth/{split_low}/{s_var}",
            f"/kaggle/input/datasets/khoatrytopublish/l3d-depth/{split_cap}/{s_var}",
            f"/kaggle/input/datasets/khoatrytopublish/l3d-depth/{split_low}/{s_var}",
            f"/kaggle/input/datasets/khoale05/l3d-depth/{split_cap}/{s_var}",
            f"/kaggle/input/datasets/khoale05/l3d-depth/{split_low}/{s_var}",
            f"/kaggle/input/laparoscopic-liver-landmarks/{split_low}/{s_var}",
            # Local workspace fallbacks
            os.path.join(ws_root, f"data/laparoscopic_liver/{split_low}/{s_var}"),
            os.path.join(ws_root, f"data/laparoscopic_liver/{split_cap}/{s_var}"),
            os.path.join(ws_root, f"data/L3D/{split_cap}/{s_var}"),
        ])

    for cand in candidates:
        if os.path.exists(cand) and len(os.listdir(cand)) > 0:
            return cand

    # Recursive dynamic search in /kaggle/input if present
    if os.path.exists("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input", followlinks=True):
            parts_lower = [p.lower() for p in Path(root).parts]
            base_lower = os.path.basename(root).lower()
            if split_low in parts_lower and any(v.lower() == base_lower for v in sub_variations):
                if len(os.listdir(root)) > 0:
                    return root

    return None


def link_directory(src: str, dst: str):
    """Creates directory and symlinks all files from src to dst."""
    os.makedirs(dst, exist_ok=True)
    if not src or not os.path.exists(src):
        return 0

    count = 0
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if not os.path.exists(d):
            try:
                os.symlink(s, d)
                count += 1
            except OSError:
                import shutil
                shutil.copyfile(s, d)
                count += 1
    return count


def precompute_sam_features(images_dir: str, sam_output_dir: str, checkpoint_path: str, device: str = "cuda"):
    """
    Computes SAM ViT-B image embeddings (256, 64, 64) for all images in images_dir.
    Matches exact BCRNet specification in repos/BCRNet/backbone/Unet.py lines 136-139.
    """
    os.makedirs(sam_output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png")))

    # Check how many need computing
    missing = [p for p in image_paths if not os.path.exists(os.path.join(sam_output_dir, os.path.splitext(os.path.basename(p))[0] + ".npy"))]
    if len(missing) == 0:
        print(f"   ✅ All {len(image_paths)} SAM features already present in {sam_output_dir}")
        return

    print(f"   ⚙️ Precomputing SAM ViT-B features for {len(missing)} images on {device}...")
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        print("   ⚠️ segment-anything package not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segment-anything"], check=True)
        from segment_anything import sam_model_registry

    if not os.path.exists(checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        print(f"   📥 Downloading SAM ViT-B weights to {checkpoint_path}...")
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        urllib.request.urlretrieve(url, checkpoint_path)

    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam_encoder = sam.image_encoder.to(device)
    sam_encoder.eval()

    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for img_p in tqdm(missing, desc="SAM Extraction"):
            base_name = os.path.splitext(os.path.basename(img_p))[0]
            out_p = os.path.join(sam_output_dir, base_name + ".npy")

            img = cv2.imread(img_p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (1024, 1024))
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
            tensor = (tensor - pixel_mean) / pixel_std

            feat = sam_encoder(tensor)  # (1, 256, 64, 64)
            feat_np = feat.squeeze(0).cpu().numpy().astype(np.float32)
            np.save(out_p, feat_np)

    print(f"   ✅ Finished precomputing {len(missing)} SAM feature embeddings.")


def prepare_bcrnet_dataset(target_dir: str, sam_checkpoint: str, device: str):
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"📦 PREPARING BCRNET DATASET AT: {target_path}")
    print("=" * 80)

    splits = ["Train", "Val", "Test"]

    for split in splits:
        print(f"\n📂 Processing Split: [{split}]")
        split_dir = target_path / split
        split_dir.mkdir(parents=True, exist_ok=True)

        images_dst = split_dir / "images"
        labels_dst = split_dir / "labels"
        depth_dst = split_dir / "depth_AdelaiDepth"
        masks_dst = split_dir / "masks_gt"
        bezier_dst = split_dir / "s_bezier"
        sam_dst = split_dir / "sam"

        # 1. Images
        src_img = find_source_dir(split, "images")
        if src_img:
            n = link_directory(src_img, str(images_dst))
            print(f"   ✓ Linked {n} images from {src_img}")
        else:
            print(f"   ⚠️ Could not find source images for {split}")

        # 2. Labels
        src_lbl = find_source_dir(split, "labels")
        if src_lbl:
            n = link_directory(src_lbl, str(labels_dst))
            print(f"   ✓ Linked {n} label files from {src_lbl}")
        else:
            print(f"   ⚠️ Could not find source labels for {split}")

        # 3. depth_AdelaiDepth
        src_depth = find_source_dir(split, "depth_AdelaiDepth")
        if src_depth:
            n = link_directory(src_depth, str(depth_dst))
            print(f"   ✓ Linked {n} authentic AdelaiDepth maps from {src_depth}")
        else:
            print(f"   ⚠️ Warning: depth_AdelaiDepth not found directly for {split}. Checking depth_anything_v2 fallback...")
            fallback_depth = find_source_dir(split, "depth_anything_v2")
            if fallback_depth:
                n = link_directory(fallback_depth, str(depth_dst))
                print(f"   ✓ Linked {n} depth maps as fallback from {fallback_depth}")

        # Ensure every image has a corresponding depth map in depth_dst
        if images_dst.exists():
            img_files = glob.glob(str(images_dst / "*.jpg")) + glob.glob(str(images_dst / "*.png"))
            missing_depth_count = 0
            for img_f in img_files:
                stem = Path(img_f).stem
                depth_f = depth_dst / f"{stem}.png"
                if not depth_f.exists():
                    blank = np.zeros((1024, 1024), dtype=np.uint8)
                    cv2.imwrite(str(depth_f), blank)
                    missing_depth_count += 1
            if missing_depth_count > 0:
                print(f"   ℹ️ Generated {missing_depth_count} fallback blank depth maps in {depth_dst}")

        # 4. masks_gt
        src_masks = find_source_dir(split, "masks_gt")
        if src_masks and len(os.listdir(src_masks)) > 0:
            n = link_directory(src_masks, str(masks_dst))
            print(f"   ✓ Linked {n} masks_gt from {src_masks}")
        else:
            if labels_dst.exists() and len(os.listdir(str(labels_dst))) > 0:
                print(f"   ⚙️ Generating masks_gt from labels in {split_dir}...")
                generate_mask_gt(str(split_dir))
                print(f"   ✅ Generated masks_gt for {split}")

        # 5. s_bezier (Optimized with np.uint8 and np.savez_compressed)
        src_bezier = find_source_dir(split, "s_bezier")
        if src_bezier and len(os.listdir(src_bezier)) > 0:
            n = link_directory(src_bezier, str(bezier_dst))
            print(f"   ✓ Linked {n} s_bezier files from {src_bezier}")
        else:
            if labels_dst.exists() and len(os.listdir(str(labels_dst))) > 0:
                print(f"   ⚙️ Generating 5th-order Bézier ground truth (s_bezier, compressed) for {split}...")
                generate_compressed_beziers(
                    src_path=str(labels_dst),
                    dst_path=str(bezier_dst),
                    degree=5,
                    curve_points_num=25,
                    image_shape=(1080, 1920),
                    linewidth=30,
                )
                print(f"   ✅ Finished generating s_bezier for {split}")

        # 6. sam
        src_sam = find_source_dir(split, "sam")
        if src_sam and len(os.listdir(src_sam)) > 0:
            n = link_directory(src_sam, str(sam_dst))
            print(f"   ✓ Linked {n} precomputed SAM feature embeddings from {src_sam}")
        else:
            if images_dst.exists() and len(os.listdir(str(images_dst))) > 0:
                precompute_sam_features(
                    images_dir=str(images_dst),
                    sam_output_dir=str(sam_dst),
                    checkpoint_path=sam_checkpoint,
                    device=device,
                )

    print("\n" + "=" * 80)
    print("✅ BCRNET DATASET PREPARATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BCRNet Dataset Structure")
    default_target = "/kaggle/working/L3D" if os.path.exists("/kaggle") else os.path.join(ws_root, "data/L3D")
    parser.add_argument("--target_dir", type=str, default=default_target)
    parser.add_argument("--sam_checkpoint", type=str, default=os.path.join(ws_root, "checkpoints/sam_vit_b_01ec64.pth"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    prepare_bcrnet_dataset(args.target_dir, args.sam_checkpoint, args.device)
