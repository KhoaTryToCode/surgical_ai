import os
import sys
import time
import json
import zipfile
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_7.utils.dataset import L3DManifoldDataset
from experiments.EXPERIMENT_7.models.manifold_steered_mask2former import ManifoldSteeredMask2Former
from experiments.EXPERIMENT_7.models.losses import ManifoldMultiTaskLoss
from experiments.EXPERIMENT_7.scripts.evaluate import run_evaluation, create_results_zip


def collate_fn_l3d(batch):
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    coords = torch.stack([b["coords"] for b in batch])
    visibilities = torch.stack([b["visibilities"] for b in batch])
    uv_maps = torch.stack([b["uv_map"] for b in batch])
    liver_masks = torch.stack([b["liver_mask"] for b in batch])
    has_falcs = torch.stack([b["has_falc"] for b in batch])
    stems = [b["stem"] for b in batch]

    # Build Hungarian matching target lists
    mask_labels_list = []
    class_labels_list = []

    for b in range(len(batch)):
        m = masks[b]
        unique_classes = torch.unique(m)
        unique_classes = unique_classes[unique_classes > 0]  # Filter background

        if len(unique_classes) == 0:
            mask_labels_list.append(torch.zeros((1, m.shape[0], m.shape[1]), dtype=torch.float32))
            class_labels_list.append(torch.zeros(1, dtype=torch.int64))
        else:
            binary_masks = torch.stack([(m == c).float() for c in unique_classes])
            mask_labels_list.append(binary_masks)
            class_labels_list.append(unique_classes.long())

    return {
        "images": images,
        "masks": masks,
        "mask_labels": mask_labels_list,
        "class_labels": class_labels_list,
        "coords": coords,
        "visibilities": visibilities,
        "uv_maps": uv_maps,
        "liver_masks": liver_masks,
        "has_falcs": has_falcs,
        "stems": stems
    }


def build_optimizer_and_scheduler(model, lr_backbone=1e-5, lr_head=1e-4, weight_decay=1e-4, epochs=60):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "pixel_level_module.encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head}
    ], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="Train Manifold-Steered Mask2Former (EXPERIMENT_7)")
    parser.add_argument("--data_dir", type=str, default=None, help="L3D dataset root")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU step")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr_head", type=float, default=1e-4, help="Learning rate for heads & decoder")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for Swin backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_coord", type=float, default=5.0, help="Weight for coordinate loss")
    parser.add_argument("--lambda_vis", type=float, default=1.0, help="Weight for visibility loss")
    parser.add_argument("--lambda_uv", type=float, default=1.0, help="Weight for dense manifold loss")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--eval_splits", type=str, default="both", choices=["val", "both"])
    parser.add_argument("--smoke_test", action="store_true", help="Run 1 step of train and val for verification")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(_WORKSPACE_ROOT, "experiments/EXPERIMENT_7/results")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"🖥️ Using device: {device}")

    # 1. Datasets & DataLoaders
    train_dataset = L3DManifoldDataset(split="Train", data_dir=args.data_dir)
    val_dataset = L3DManifoldDataset(split="Val", data_dir=args.data_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1 if args.smoke_test else args.batch_size,
        shuffle=not args.smoke_test,
        num_workers=0 if args.smoke_test else args.num_workers,
        collate_fn=collate_fn_l3d,
        pin_memory=(device.type == "cuda")
    )

    # 2. Build Model
    model = ManifoldSteeredMask2Former().to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ Loaded ManifoldSteeredMask2Former ({total_params:.2f}M params)")

    # 3. Criterion & Optimizer
    criterion = ManifoldMultiTaskLoss(
        lambda_vis=args.lambda_vis,
        lambda_coord=args.lambda_coord,
        lambda_uv=args.lambda_uv
    )
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        epochs=1 if args.smoke_test else args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 4. Training Loop
    epochs = 1 if args.smoke_test else args.epochs
    best_val_dice = -1.0
    best_epoch = -1
    best_model_path = os.path.join(args.out_dir, "best_model.pth")
    training_log = []

    print(f"\n🚀 Launching training for {epochs} epochs (Accumulation steps: {args.accum_steps})...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        t_epoch_start = time.time()
        optimizer.zero_grad()

        running_loss = 0.0
        running_m2f = 0.0
        running_vis = 0.0
        running_coord = 0.0
        running_uv = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            imgs = batch["images"].to(device)
            mask_labels = [m.to(device) for m in batch["mask_labels"]]
            class_labels = [c.to(device) for c in batch["class_labels"]]
            gt_coords = batch["coords"].to(device)
            gt_vis = batch["visibilities"].to(device)
            gt_uv = batch["uv_maps"].to(device)
            gt_liver_mask = batch["liver_masks"].to(device)
            has_falcs = batch["has_falcs"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(
                    pixel_values=imgs,
                    mask_labels=mask_labels,
                    class_labels=class_labels
                )
                m2f_loss = outputs["m2f_loss"]
                pred_coords = outputs["pred_coords"]
                pred_vis = outputs["pred_vis"]
                pred_uv = outputs["pred_uv"]

                loss, breakdown = criterion(
                    m2f_loss=m2f_loss,
                    pred_coords=pred_coords,
                    pred_vis=pred_vis,
                    pred_uv=pred_uv,
                    gt_coords=gt_coords,
                    gt_vis=gt_vis,
                    gt_uv=gt_uv,
                    gt_liver_mask=gt_liver_mask,
                    has_falc=has_falcs
                )
                loss_scaled = loss / args.accum_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += breakdown["total_loss"]
            running_m2f += breakdown["m2f_loss"]
            running_vis += breakdown["loss_vis"]
            running_coord += breakdown["loss_coord"]
            running_uv += breakdown["loss_uv"]
            num_batches += 1

            if args.smoke_test:
                print(f"🔥 Smoke test step passed! Loss: {loss.item():.4f}")
                break

        scheduler.step()
        epoch_time = time.time() - t_epoch_start

        avg_loss = running_loss / max(num_batches, 1)
        avg_m2f = running_m2f / max(num_batches, 1)
        avg_vis = running_vis / max(num_batches, 1)
        avg_coord = running_coord / max(num_batches, 1)
        avg_uv = running_uv / max(num_batches, 1)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | Loss: {avg_loss:.4f} (M2F: {avg_m2f:.3f}, Vis: {avg_vis:.3f}, Coord: {avg_coord:.3f}, UV: {avg_uv:.3f})")

        # Validation Check
        if not args.smoke_test:
            val_summary, _ = run_evaluation(model, split="Val", data_dir=args.data_dir, out_dir=None, device=args.device, render_p40=False)
            val_dice = val_summary["macro_dice"]
            val_iou = val_summary["macro_iou"]
            val_assd = val_summary["macro_assd"]
            print(f"  👉 [Validation] Macro Dice: {val_dice*100:.2f}% | IoU: {val_iou*100:.2f}% | ASSD: {val_assd:.2f}px")

            log_entry = {
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_m2f": avg_m2f,
                "train_vis": avg_vis,
                "train_coord": avg_coord,
                "train_uv": avg_uv,
                "val_macro_dice": val_dice,
                "val_macro_iou": val_iou,
                "val_macro_assd": val_assd,
                "val_ridge_dice": val_summary["ridge_dice"],
                "val_sil_dice": val_summary["sil_dice"],
                "val_falc_dice": val_summary["falc_dice"]
            }
            training_log.append(log_entry)
            pd.DataFrame(training_log).to_csv(os.path.join(args.out_dir, "training_log.csv"), index=False)

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_summary": val_summary
                }, best_model_path)
                print(f"  ⭐ New Best Model saved! (Macro Dice: {best_val_dice*100:.2f}%)")
        else:
            # Smoke test dummy save
            torch.save({"epoch": 1, "model_state_dict": model.state_dict()}, best_model_path)
            print("🔥 Smoke test checkpoint saved!")

    # 5. Final Comprehensive Evaluation & Packaging
    if not args.smoke_test:
        print("\n" + "="*70)
        print(f"🏆 Training Complete! Best Epoch: {best_epoch} (Macro Dice: {best_val_dice*100:.2f}%)")
        print("Running full benchmark suite on best model...")
        print("="*70)

        best_model = load_manifold_steered_model(best_model_path, device=args.device)

        # Validation Split
        val_summary, val_df = run_evaluation(best_model, split="Val", data_dir=args.data_dir, out_dir=args.out_dir, device=args.device, render_p40=True)
        val_df.to_csv(os.path.join(args.out_dir, "val_predictions.csv"), index=False)

        summary_combined = {
            "val_summary": val_summary,
            "best_epoch": best_epoch,
            "checkpoint_path": best_model_path
        }

        # Test Split
        if args.eval_splits == "both":
            test_summary, test_df = run_evaluation(best_model, split="Test", data_dir=args.data_dir, out_dir=args.out_dir, device=args.device, render_p40=False)
            test_df.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)
            summary_combined["test_summary"] = test_summary

        # Save metrics summary JSON
        summary_path = os.path.join(args.out_dir, "metrics_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary_combined, f, indent=4)
        print(f"📄 Saved metrics summary to: {summary_path}")

        # Automatically package results.zip
        zip_path = os.path.join(args.out_dir, "results.zip")
        create_results_zip(args.out_dir, zip_path)

if __name__ == "__main__":
    main()
