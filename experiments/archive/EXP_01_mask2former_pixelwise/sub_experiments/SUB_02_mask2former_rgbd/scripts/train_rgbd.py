#!/usr/bin/env python3
"""
SUB_02: Mask2Former RGB-D — Adding Depth Anything V2 as a 4th Channel.

Minimal code change extension of Mask2Former baseline:
  - Input: RGB (3 channels) + Depth Anything V2 (1 channel) = 4 channels
  - Swin-Tiny Patch Embedding: First Conv2d expanded from (3 -> 96) to (4 -> 96)
  - Pretrained ImageNet weights preserved; 4th channel initialized to channel mean
  - Loss: Standard BCE + Dice per Query
"""

import os
import sys
import glob
import cv2
import gc
import argparse
from pathlib import Path
import numpy as np

# NumPy 2.0+ compatibility for surface_distance and legacy libraries
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NAN"):
    np.NAN = np.nan
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

# ── Setup paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUB_DIR = os.path.dirname(SCRIPT_DIR)
EXP_DIR = os.path.dirname(os.path.dirname(SUB_DIR))
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))

for p in [SCRIPT_DIR, os.path.join(EXP_DIR, 'models'), os.path.join(REPO_ROOT, 'shared'), REPO_ROOT]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from shared.utils.prepare_dataset import get_split
    from shared.utils.dataset import load_image, load_depth, load_mask
except ImportError:
    from utils.prepare_dataset import get_split
    from utils.dataset import load_image, load_depth, load_mask

# 0. FREE PREVIOUS GPU MEMORY CACHE
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 1. Install Dependencies if missing
try:
    import transformers
except ImportError:
    os.system("pip install -q transformers")
    import transformers

try:
    import wandb
except ImportError:
    os.system("pip install -q wandb")
    import wandb

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

try:
    import surface_distance
    from surface_distance import metrics as sd_metrics
    HAS_SURFACE_DIST = True
except Exception:
    HAS_SURFACE_DIST = False


def parse_args():
    parser = argparse.ArgumentParser(description="Mask2Former RGB-D Training with Depth Anything V2")
    parser.add_argument("--data_path", type=str, default="/kaggle/working/L3D", help="Dataset path")
    parser.add_argument("--save_dir", type=str, default="/kaggle/working/results_rgbd", help="Save directory")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60)")
    parser.add_argument("--lr", type=float, default=8e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per step")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# Official TopoNet metric evaluation function
def evaluation(pred, gt):
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    dice = (2 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
    iou = dice / (2 - dice)
    return iou, dice


def compute_toponet_metrics(pred_map, gt_2d):
    pred_channels = np.array([pred_map == i for i in range(4)]).astype(np.uint8)
    gt_channels = np.array([gt_2d == i for i in range(4)]).astype(np.uint8)
    iou, dice = evaluation(pred_channels[1:].flatten(), gt_channels[1:].flatten())

    assd = None
    if HAS_SURFACE_DIST:
        if 0 == np.count_nonzero(pred_channels[1:]):
            assd = 80.0
        else:
            temp_assd = []
            for i in range(3):
                sd = sd_metrics.compute_surface_distances(
                    np.array(gt_channels[i + 1], dtype=bool),
                    np.array(pred_channels[i + 1], dtype=bool),
                    (1.0, 1.0)
                )
                avg_sd = surface_distance.compute_average_surface_distance(sd)
                temp_assd.append(avg_sd[1])
            if np.mean(temp_assd) < 500:
                assd = np.mean(temp_assd)
            else:
                assd = 80.0

    return dice, iou, assd


# ==================== ADAPT BACKBONE TO 4-CHANNEL RGB-D ====================
def adapt_model_to_rgbd(model):
    """
    Finds the first Conv2d in the model backbone (in_channels=3)
    and replaces it with an in_channels=4 Conv2d (RGB + Depth).
    Preserves ImageNet pretrained weights and initializes the 4th channel
    as the channel-average of the RGB weights.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            old_conv = module
            new_conv = nn.Conv2d(
                in_channels=4,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=old_conv.groups,
                bias=(old_conv.bias is not None)
            )

            with torch.no_grad():
                # Copy pretrained weights to channels 0, 1, 2
                new_conv.weight[:, :3, :, :] = old_conv.weight
                # Initialize depth channel 3 from the mean of RGB channels
                new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)

            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, new_conv)
            print(f"✅ Successfully converted '{name}' from 3 to 4 channels for RGB-D input.")
            return model

    raise RuntimeError("Could not locate first 3-channel Conv2d in model.")


# ==================== DATASET DEFINITION ====================
class Mask2FormerRGBDDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        rgb_img = load_image(img_path)
        # load_depth automatically checks depth_anything_v2/*.png first
        depth_img = load_depth(img_path)
        gt_masks = load_mask(img_path)
        gt_2d = np.argmax(gt_masks, axis=0).astype(np.int32)
        return rgb_img, depth_img, gt_2d, str(img_path)


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("🚀 RUNNING SUB_02: MASK2FORMER RGB-D (DEPTH ANYTHING V2)")
    print(f"   Epochs: {args.epochs} | LR: {args.lr} | Effective Batch: {args.batch_size * args.accumulation_steps}")
    print(f"   Input Channels: 4 (R, G, B, Depth)")
    print("=" * 70)

    # WandB Setup
    student_id = "10423057"
    api_key = os.environ.get("WANDB_API_KEY", "83f4544a22543e319c6009abceaac90b634c68a3")
    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.init(mode="offline")

    wandb.init(
        project="liver-landmark-segmentation-ablation",
        name="SUB_02-Mask2Former-RGBD",
        id=f"sub02_rgbd_{student_id}",
        resume="allow",
        config={
            "experiment": "SUB_02_mask2former_rgbd",
            "backbone": "Swin-Tiny (4-channel RGB-D)",
            "depth_source": "Depth Anything V2 (ViT-B)",
            "epochs": args.epochs,
            "lr": args.lr,
            "effective_batch_size": args.batch_size * args.accumulation_steps,
            "resolution": "1024x1024"
        }
    )

    os.makedirs(args.save_dir, exist_ok=True)
    BEST_CKPT_PATH = os.path.join(args.save_dir, "best_swin_rgbd.pth")
    LATEST_CKPT_PATH = os.path.join(args.save_dir, "latest_swin_rgbd.pth")

    # Load splits
    train_files, test_files, val_files = get_split(args.data_path)
    print(f"Train samples: {len(train_files)} | Val samples: {len(val_files)}")

    train_dataset = Mask2FormerRGBDDataset(train_files)
    val_dataset = Mask2FormerRGBDDataset(val_files)

    # Initialize Processor and Model
    model_name = "facebook/mask2former-swin-tiny-ade-semantic"
    processor = AutoImageProcessor.from_pretrained(model_name, reduce_labels=False, ignore_index=255)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_name,
        num_labels=4,
        ignore_mismatched_sizes=True
    )
    # Adapt to 4-channel RGB-D
    model = adapt_model_to_rgbd(model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Resume Checkpoint if exists
    start_epoch = 1
    best_val_dice = 0.0

    if os.path.exists(LATEST_CKPT_PATH):
        print(f"\n🔄 Resuming from checkpoint '{LATEST_CKPT_PATH}'...")
        ckpt = torch.load(LATEST_CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_dice = ckpt["best_val_dice"]
        print(f"   Resuming from epoch {start_epoch} | Best Val Dice so far: {best_val_dice:.4f}")

    # ==================== TRAINING LOOP ====================
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        train_dices = []
        train_ious = []
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_dataset), total=len(train_dataset), desc=f"Epoch {epoch}/{args.epochs} [RGB-D]")
        for step, (rgb_img, depth_img, gt_2d, _) in pbar:
            # 1. Process RGB and GT Mask through standard HF Processor
            inputs = processor(
                images=[rgb_img],
                segmentation_maps=[gt_2d],
                return_tensors="pt"
            )

            rgb_pixels = inputs["pixel_values"].to(device)  # (1, 3, 1024, 1024)
            mask_labels = [m.to(device) for m in inputs["mask_labels"]]
            class_labels = [c.to(device) for c in inputs["class_labels"]]

            # 2. Normalize Depth and Concatenate as 4th Channel
            depth_tensor = torch.from_numpy(depth_img).float() / 255.0
            depth_tensor = (depth_tensor - 0.5) / 0.25
            depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0).to(device)

            # Match depth spatial dimensions (H, W) to processor's rgb_pixels
            if depth_tensor.shape[-2:] != rgb_pixels.shape[-2:]:
                depth_tensor = F.interpolate(
                    depth_tensor,
                    size=rgb_pixels.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            # RGB-D 4-Channel Input
            pixel_values_4ch = torch.cat([rgb_pixels, depth_tensor], dim=1)

            # 3. Model Forward
            outputs = model(
                pixel_values=pixel_values_4ch,
                mask_labels=mask_labels,
                class_labels=class_labels
            )

            step_loss = outputs.loss
            loss = step_loss / args.accumulation_steps
            loss.backward()

            total_loss += step_loss.item()

            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(train_dataset):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                pred_map = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[(1024, 1024)]
                )[0].cpu().numpy()
                t_dice, t_iou, _ = compute_toponet_metrics(pred_map, gt_2d)
                train_dices.append(t_dice)
                train_ious.append(t_iou)

            pbar.set_postfix({
                "loss": f"{total_loss / (step + 1):.4f}",
                "tr_dice": f"{np.mean(train_dices):.4f}"
            })

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # ==================== EVALUATION ON VAL SPLIT ====================
        model.eval()
        val_losses = []
        val_dices = []
        val_ious = []
        val_assds = []

        with torch.no_grad():
            for rgb_img, depth_img, gt_2d, _ in tqdm(val_dataset, desc=f"Epoch {epoch}/{args.epochs} [Val RGB-D]"):
                inputs = processor(images=[rgb_img], segmentation_maps=[gt_2d], return_tensors="pt")
                rgb_pixels = inputs["pixel_values"].to(device)
                mask_labels = [m.to(device) for m in inputs["mask_labels"]]
                class_labels = [c.to(device) for c in inputs["class_labels"]]

                depth_tensor = torch.from_numpy(depth_img).float() / 255.0
                depth_tensor = (depth_tensor - 0.5) / 0.25
                depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0).to(device)

                if depth_tensor.shape[-2:] != rgb_pixels.shape[-2:]:
                    depth_tensor = F.interpolate(
                        depth_tensor,
                        size=rgb_pixels.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

                pixel_values_4ch = torch.cat([rgb_pixels, depth_tensor], dim=1)

                outputs = model(
                    pixel_values=pixel_values_4ch,
                    mask_labels=mask_labels,
                    class_labels=class_labels
                )

                val_losses.append(outputs.loss.item())

                pred_map = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[(1024, 1024)]
                )[0].cpu().numpy()

                v_dice, v_iou, v_assd = compute_toponet_metrics(pred_map, gt_2d)
                val_dices.append(v_dice)
                val_ious.append(v_iou)
                if v_assd is not None:
                    val_assds.append(v_assd)

        epoch_train_loss = total_loss / len(train_dataset)
        mean_val_loss = np.mean(val_losses)
        mean_train_dice = np.mean(train_dices)
        mean_val_iou = np.mean(val_ious)
        mean_val_dice = np.mean(val_dices)
        mean_val_assd = np.mean(val_assds) if len(val_assds) > 0 else 0.0

        print(f"👉 Epoch {epoch:03d} | Tr Loss: {epoch_train_loss:.4f} | Val Loss: {mean_val_loss:.4f} | "
              f"Tr Dice: {mean_train_dice:.4f} | Val Dice: {mean_val_dice:.4f} | Val IoU: {mean_val_iou:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_dice": mean_train_dice,
            "val_loss": mean_val_loss,
            "val_dice": mean_val_dice,
            "val_iou": mean_val_iou,
            "learning_rate": current_lr
        })

        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            wandb.run.summary["best_val_dice"] = best_val_dice
            print(f"  🏆 New Best Model! Val Dice: {best_val_dice:.4f} -> '{BEST_CKPT_PATH}'")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_dice': best_val_dice
        }, LATEST_CKPT_PATH)

    wandb.finish()
    print(f"\n✅ SUB_02 (RGB-D) Complete! Best Validation Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
