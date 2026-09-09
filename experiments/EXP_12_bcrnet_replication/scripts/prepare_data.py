"""
Data Preparation & Asset Generation Script for EXP_12 (BCRNet Replication)
==========================================================================
Prepares the complete L3D dataset structure required by BCRNet's BezierDataset:
  <target_dir>/<Split>/
    ├── images/             (.jpg images, symlinked)
    ├── depth_AdelaiDepth/  (.png depth maps from Kaggle L3D dataset)
    ├── labels/             (.json annotation files)
    ├── masks_gt/           (.png ground truth masks)
    ├── s_bezier/           (.npz 5th-order Bézier GT generated via BCRNet preprocess)
    └── sam/                (.npy SAM ViT-B 256x64x64 features)

Supports Kaggle CUDA, Colab, and local macOS environments with auto-discovery.
"""

import os
import sys
import glob
import json
import argparse
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
from preprocess import generate_beziers, generate_mask_gt


def find_source_dir(split: str, sub: str) -> str:
    """Finds existing directory for given split and subdirectory."""
    split_cap = split.capitalize()
    split_low = split.lower()

    candidates = [
        # Kaggle input paths
        f"/kaggle/input/datasets/khoatrytopublish/l3d-{split_low}/{split_cap}/{sub}",
        f"/kaggle/input/datasets/khoatrytopublish/l3d-{split_low}/{split_low}/{sub}",
        f"/kaggle/input/l3d-{split_low}/{split_cap}/{sub}",
        f"/kaggle/input/l3d-{split_low}/{split_low}/{sub}",
        f"/kaggle/input/laparoscopic-liver-landmarks/{split_low}/{sub}",
        # Local workspace fallbacks
        os.path.join(ws_root, f"data/laparoscopic_liver/{split_low}/{sub}"),
        os.path.join(ws_root, f"data/laparoscopic_liver/{split_cap}/{sub}"),
        os.path.join(ws_root, f"data/L3D/{split_cap}/{sub}"),
    ]

    for cand in candidates:
        if os.path.exists(cand) and len(os.listdir(cand)) > 0:
            return cand

    # Recursive dynamic search in /kaggle/input if present
    if os.path.exists("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input", followlinks=True):
            parts_lower = [p.lower() for p in Path(root).parts]
            if split_low in parts_lower and os.path.basename(root).lower() == sub.lower():
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

    # Preprocessing constants for SAM
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

        # 5. s_bezier
        src_bezier = find_source_dir(split, "s_bezier")
        if src_bezier and len(os.listdir(src_bezier)) > 0:
            n = link_directory(src_bezier, str(bezier_dst))
            print(f"   ✓ Linked {n} s_bezier files from {src_bezier}")
        else:
            if labels_dst.exists() and len(os.listdir(str(labels_dst))) > 0:
                print(f"   ⚙️ Generating 5th-order Bézier ground truth (s_bezier) for {split}...")
                generate_beziers(
                    src_path=str(labels_dst),
                    dst_path=str(bezier_dst),
                    degree=5,
                    curve_points_num=25,
                    num_class=3,
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
