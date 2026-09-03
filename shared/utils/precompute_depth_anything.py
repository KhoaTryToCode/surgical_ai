#!/usr/bin/env python3
"""
Precompute Depth Anything V2 depth maps for Laparoscopic Liver Landmark Dataset (L3D).

Processes Train, Val, and Test splits:
  [data_root]/[split]/images/*.jpg -> [output_root]/[split]/depth_anything_v2/*.png

Supports:
  1. Hugging Face Hub pre-trained Depth Anything V2 (Base-hf / Small-hf) [Default: Base / ViT-B]
  2. Local TopoNet Depth Anything V2 checkpoint (.pth)
  3. Batched GPU inference with mixed precision for high throughput
  4. Automatic resume (skips already computed depth files)
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(SHARED_DIR)

for p in [SCRIPT_DIR, SHARED_DIR, REPO_ROOT]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from shared.utils.prepare_dataset import get_split
except ImportError:
    from utils.prepare_dataset import get_split


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute Depth Anything V2 maps for L3D dataset")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Root directory of dataset containing train/val/test splits. Auto-detected if None.")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root directory to save depth maps. If None, saves into [data_path]/[split]/depth_anything_v2/ (or /kaggle/working/L3D if input is read-only).")
    parser.add_argument("--dirname", type=str, default="depth_anything_v2",
                        help="Subdirectory name inside each split (default: 'depth_anything_v2')")
    parser.add_argument("--model_id", type=str, default="depth-anything/Depth-Anything-V2-Base-hf",
                        help="Hugging Face model ID (default: 'depth-anything/Depth-Anything-V2-Base-hf' for ViT-B)")
    parser.add_argument("--local_weights", type=str, default=None,
                        help="Path to local depth_anything_v2_vitb.pth checkpoint if not using Hugging Face Hub")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for depth inference (default: 8)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker count (default: 4)")
    parser.add_argument("--target_size", type=int, default=1024,
                        help="Output depth map resolution (default: 1024x1024)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip frames whose depth maps already exist (default: True)")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Force recomputing even if depth files already exist")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Execution device (cuda/cpu/mps)")
    return parser.parse_args()


def resolve_data_path(custom_path=None):
    if custom_path and os.path.exists(custom_path):
        return Path(custom_path).resolve()

    candidates = [
        "/kaggle/working/L3D",
        "/kaggle/input/datasets/khoatrytopublish/l3d-train/Train/..",
        "data/laparoscopic_liver",
        "/content/L3D"
    ]
    for c in candidates:
        p = Path(c).resolve()
        if p.exists():
            return p

    raise FileNotFoundError(f"Could not locate dataset root. Checked candidates: {candidates}")


class SurgicalImageDataset(Dataset):
    def __init__(self, file_paths, target_size=1024):
        self.file_paths = [Path(p) for p in file_paths]
        self.target_size = target_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Failed to read image at: {path}")

        if bgr.shape[0] != self.target_size or bgr.shape[1] != self.target_size:
            bgr = cv2.resize(bgr, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Convert to float tensor [0, 1] normalized with ImageNet stats
        tensor_rgb = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0

        return tensor_rgb, str(path)


def load_depth_model(model_id, local_weights=None, device="cuda"):
    print("=" * 80)
    if local_weights and os.path.exists(local_weights):
        print(f"📦 Loading local Depth Anything V2 weights from: '{local_weights}'...")
        # Import local implementation
        from experiments.EXP_02_surgical_gemap.models.depth_anything_v2.dpt import DepthAnythingV2
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        enc = 'vitb' if 'vitb' in local_weights.lower() else 'vits' if 'vits' in local_weights.lower() else 'vitl'
        model = DepthAnythingV2(**model_configs[enc])
        model.load_state_dict(torch.load(local_weights, map_location=device))
        model.to(device).eval()
        return model, "local"
    else:
        print(f"🌐 Loading Depth Anything V2 from Hugging Face: '{model_id}'...")
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError:
            print("Installing transformers library...")
            os.system("pip install -q transformers")
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
        model.eval()
        return (model, processor), "hf"


def run_hf_depth_batch(model, processor, batch_tensors, device, target_size=1024):
    """
    Runs batched inference using Hugging Face Depth Anything V2
    """
    # batch_tensors is (B, 3, H, W) in [0, 1] range
    # HF processor expects uint8 or numpy/PIL images or pixel_values
    pixel_values = processor(
        images=list(batch_tensors),
        return_tensors="pt",
        do_rescale=False
    )["pixel_values"].to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(pixel_values=pixel_values)
        else:
            outputs = model(pixel_values=pixel_values)

        predicted_depth = outputs.predicted_depth  # (B, H_proc, W_proc)

        # Bilinear resize to target 1024x1024
        depth_interpolated = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)  # (B, 1024, 1024)

    return depth_interpolated.cpu().numpy()


def run_local_depth_batch(model, batch_tensors, device, target_size=1024):
    """
    Runs inference using TopoNet's local DepthAnythingV2
    """
    batch = batch_tensors.to(device)
    # Resize to 518x518 for DINOv2 patch division (518 / 14 = 37)
    resized_input = F.interpolate(batch, size=(518, 518), mode="bilinear", align_corners=False)
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    normalized = (resized_input - mean) / std

    with torch.no_grad():
        depth = model(normalized)  # (B, 1, H, W)
        depth_out = F.interpolate(depth, size=(target_size, target_size), mode="bilinear", align_corners=True)

    return depth_out.squeeze(1).cpu().numpy()


def save_depth_map(depth_array, out_path):
    """
    Normalizes single depth map to 0-255 uint8 and writes as PNG.
    """
    d_min = depth_array.min()
    d_max = depth_array.max()

    if d_max - d_min > 1e-6:
        d_norm = (depth_array - d_min) / (d_max - d_min)
    else:
        d_norm = np.zeros_like(depth_array)

    d_uint8 = (d_norm * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(str(out_path), d_uint8)


def determine_out_path(img_path, data_root, output_root, split_name, dirname="depth_anything_v2"):
    """
    Computes output path maintaining split structure:
      output_root / split_name / dirname / [filename].png
    """
    p = Path(img_path)
    stem = p.stem + ".png"

    if output_root:
        return Path(output_root) / split_name / dirname / stem

    # Check if data_root directory is writable
    target_dir = Path(data_root) / split_name / dirname
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        test_file = target_dir / ".write_test"
        with open(test_file, "w") as f:
            f.write("test")
        test_file.unlink()
        return target_dir / stem
    except (OSError, PermissionError):
        # Fallback to /kaggle/working/L3D if input is read-only (e.g. in /kaggle/input)
        fallback = Path("/kaggle/working/L3D") / split_name / dirname
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback / stem


def process_split(split_name, file_paths, model_pack, model_type, data_root, output_root, dirname, args):
    if not file_paths:
        print(f"⚠️ Split '{split_name}' has 0 images. Skipping.")
        return

    print(f"\n🚀 Processing Split: [{split_name.upper()}] — {len(file_paths)} images")

    # Filter out existing if requested
    items_to_process = []
    for fp in file_paths:
        out_p = determine_out_path(fp, data_root, output_root, split_name, dirname)
        if args.skip_existing and not args.force_recompute and out_p.exists():
            continue
        items_to_process.append((fp, out_p))

    if not items_to_process:
        print(f"   ✅ All {len(file_paths)} depth maps already exist for '{split_name}'. Skipping.")
        return

    print(f"   Computed {len(file_paths) - len(items_to_process)} already. Processing remaining {len(items_to_process)} images...")

    dataset = SurgicalImageDataset([x[0] for x in items_to_process], target_size=args.target_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device.type == "cuda")
    )

    device = torch.device(args.device)
    curr_idx = 0

    pbar = tqdm(loader, desc=f"Depth V2 [{split_name}]")
    for batch_imgs, _ in pbar:
        if model_type == "hf":
            model, processor = model_pack
            depth_maps = run_hf_depth_batch(model, processor, batch_imgs, device, target_size=args.target_size)
        else:
            depth_maps = run_local_depth_batch(model_pack, batch_imgs, device, target_size=args.target_size)

        for i in range(len(depth_maps)):
            out_path = items_to_process[curr_idx][1]
            save_depth_map(depth_maps[i], out_path)
            curr_idx += 1


def main():
    args = parse_args()
    args.device = torch.device(args.device)

    data_root = resolve_data_path(args.data_path)
    print(f"📂 Dataset Root: {data_root}")

    train_files, test_files, val_files = get_split(data_root)
    print(f"📊 Dataset Counts: Train={len(train_files)} | Val={len(val_files)} | Test={len(test_files)}")

    # Load Model
    model_pack, model_type = load_depth_model(
        model_id=args.model_id,
        local_weights=args.local_weights,
        device=args.device
    )

    # Process all 3 splits
    splits = [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]

    for split_name, files in splits:
        process_split(
            split_name=split_name,
            file_paths=files,
            model_pack=model_pack,
            model_type=model_type,
            data_root=data_root,
            output_root=args.output_root,
            dirname=args.dirname,
            args=args
        )

    print("\n" + "=" * 80)
    print(f"🎉 Depth Anything V2 Precomputation Complete across Train, Val, and Test!")
    print(f"   Subdirectory name: '{args.dirname}'")
    print("=" * 80)


if __name__ == "__main__":
    main()
