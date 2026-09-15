"""
EXP_12: SurgicalCurveFormer v2 Training Script
==============================================
Omni-Geometric Master Architecture training pipeline.
Supports:
- Local macOS (MPS/CPU) and Kaggle CUDA GPU
- Differential backbone learning rates
- Hold-15 + Cosine Annealing dynamic weighting
- Mixed Precision (AMP)
- W&B metric logging & checkpoint saving
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="torch.amp.*")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

exp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
workspace_root = os.path.abspath(os.path.join(exp_root, "../.."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
if exp_root not in sys.path:
    sys.path.insert(0, exp_root)

from configs.exp12_config import EXP12Config, resolve_dataset_dir
from models.surgical_curve_former_v2 import SurgicalCurveFormerV2
from models.losses_v2 import SurgicalCurveFormerV2Loss
from utils.dataset import SurgicalCurveFormerDataset




def parse_args():
    cfg = EXP12Config()
    p = argparse.ArgumentParser(description="EXP_12: Train SurgicalCurveFormer v2")
    p.add_argument("--dataset_dir",      type=str,   default=resolve_dataset_dir())
    p.add_argument("--epochs",           type=int,   default=cfg.epochs)
    p.add_argument("--batch_size",       type=int,   default=cfg.batch_size)
    p.add_argument("--lr",               type=float, default=cfg.lr)
    p.add_argument("--backbone_lr_mult", type=float, default=cfg.backbone_lr_mult)
    p.add_argument("--weight_decay",     type=float, default=cfg.weight_decay)
    p.add_argument("--hold_epochs",      type=int,   default=cfg.hold_epochs)
    p.add_argument("--lambda_d_min",     type=float, default=cfg.lambda_d_min)
    p.add_argument("--save_dir",         type=str,   default=cfg.save_dir)
    p.add_argument("--acpi_top_k",       type=int,   default=cfg.acpi_top_k)
    p.add_argument("--amp",              action="store_true", default=cfg.amp)
    p.add_argument("--use_depth",        action="store_true", default=cfg.use_depth)
    p.add_argument("--wandb",            action="store_true", default=cfg.wandb)
    p.add_argument("--wandb_key",        type=str,   default=cfg.wandb_key)
    p.add_argument("--device",           type=str,   default="")

    p.add_argument("--num_workers",      type=int,   default=cfg.num_workers)
    return p.parse_args()


def get_device(forced: str = "") -> torch.device:
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dice_metric(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> float:
    inter = (pred_mask & gt_mask).sum()
    return float((2.0 * inter + eps) / (pred_mask.sum() + gt_mask.sum() + eps))


@torch.no_grad()
def evaluate(model, loader, device, dilate_px: int = 30) -> dict:
    """
    Validation evaluation reporting:
    - Dice (CNN Decoder)
    - Dice (Cauchy Soft Rasterizer)
    - Existence Accuracy
    """
    model.eval()
    all_dice_cnn = []
    all_dice_raster = []
    all_exist_acc = []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))

    for batch in loader:
        x = batch["image"].to(device)
        B = x.shape[0]
        out = model(x)

        seg_list = out.get("seg_logits_list", [])
        soft_ras = out["raster_masks"].cpu().float()
        exist_p = out["exist_probs"].cpu()
        exist_gt = batch["active_mask"].float()
        gt_masks = batch["target_masks"]
        active = batch["active_mask"]

        # Existence accuracy
        pred_exist = (exist_p > 0.5).float()
        exist_acc = (pred_exist == exist_gt).float().mean().item()
        all_exist_acc.append(exist_acc)

        # CNN decoder masks (level 0: 512x512)
        if seg_list:
            cnn_logits = seg_list[0].detach().cpu()
            cnn_512 = torch.sigmoid(cnn_logits)
        else:
            cnn_512 = F.interpolate(soft_ras, size=(512, 512), mode="bilinear", align_corners=False)

        raster_512 = F.interpolate(soft_ras, size=(512, 512), mode="bilinear", align_corners=False)

        for b in range(B):
            for m in range(cnn_512.shape[1]):
                if not active[b, m].item():
                    continue
                gt_np = (gt_masks[b, m].numpy() > 0.5).astype(np.uint8)
                gt_dil = cv2.dilate(gt_np, kernel)

                # CNN decoder Dice
                pred_cnn = (cnn_512[b, m].numpy() > 0.3).astype(np.uint8)
                pred_cnn_dil = cv2.dilate(pred_cnn, kernel)
                d_cnn = dice_metric(pred_cnn_dil > 0, gt_dil > 0)
                all_dice_cnn.append(d_cnn)

                # Rasterizer Dice
                pred_ras = (raster_512[b, m].numpy() > 0.3).astype(np.uint8)
                pred_ras_dil = cv2.dilate(pred_ras, kernel)
                d_ras = dice_metric(pred_ras_dil > 0, gt_dil > 0)
                all_dice_raster.append(d_ras)

    model.train()
    return {
        "dice":        float(np.mean(all_dice_cnn))    if all_dice_cnn    else 0.0,
        "dice_cnn":    float(np.mean(all_dice_cnn))    if all_dice_cnn    else 0.0,
        "dice_raster": float(np.mean(all_dice_raster)) if all_dice_raster else 0.0,
        "exist_acc":   float(np.mean(all_exist_acc))   if all_exist_acc   else 0.0,
    }


def main():
    args = parse_args()
    device = get_device(args.device)

    print("=" * 80)
    print("🚀 [EXP_12] Training SurgicalCurveFormer v2 (Omni-Geometric Master Synthesis)")
    print(f"   Device:         {device}")
    print(f"   Dataset:        {args.dataset_dir}")
    print(f"   Epochs:         {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"   AMP:            {args.amp}  |  Depth: {args.use_depth}")
    print(f"   ACPI top-K:     {args.acpi_top_k}  |  Hold Epochs: {args.hold_epochs}")
    print(f"   Save Directory: {args.save_dir}")
    print("=" * 80)

    # ── W&B Setup ──────────────────────────────────────────────────────────
    use_wandb = args.wandb
    if use_wandb and args.wandb_key:
        try:
            import wandb
            wandb.login(key=args.wandb_key)
            wandb.init(
                project="Surgical_AI_EXP12_SurgicalCurveFormerV2",
                entity="10423057-vietnamese-german-university",
                name=f"EXP12_OmniGeometric_v2_k{args.acpi_top_k}",
                config=vars(args),
            )
        except Exception as e:
            print(f"⚠️  W&B init failed: {e}. Continuing without remote logging.")
            use_wandb = False



    # ── Dataset & Loaders ──────────────────────────────────────────────────
    train_ds = SurgicalCurveFormerDataset(
        dataset_dir=args.dataset_dir, mode="train",
        image_size=512, use_depth=args.use_depth,
        acpi_stride=32, render_size=128, bezier_degree=5,
    )
    val_ds = SurgicalCurveFormerDataset(
        dataset_dir=args.dataset_dir, mode="val",
        image_size=512, use_depth=args.use_depth,
        acpi_stride=32, render_size=128, bezier_degree=5,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    print(f"   Dataset Loaded: {len(train_ds)} train samples, {len(val_ds)} val samples")

    # ── Model Assembly ─────────────────────────────────────────────────────
    model = SurgicalCurveFormerV2(
        num_classes=3, top_k=args.acpi_top_k, bezier_order=5,
        fpn_dim=256, grid_size=128, image_size=512,
    ).to(device)

    # ── Optimizer & Scheduler ──────────────────────────────────────────────
    param_groups = model.get_param_groups(base_lr=args.lr, backbone_lr_mult=args.backbone_lr_mult)
    optimizer = Adam(param_groups, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    # ── Master Loss Suite ──────────────────────────────────────────────────
    criterion = SurgicalCurveFormerV2Loss(
        num_classes=3, num_hcr_stages=3,
        lambda_s=10.0, lambda_ind=1.0, ind_pos_weight=15.0,
        lambda_cs=1.0, lambda_crv=2.0, lambda_ac_cldice=1.0,
        lambda_dice=0.5, lambda_exist=1.0, exist_pos_weight=3.0,
        hold_epochs=args.hold_epochs, total_epochs=args.epochs,
        lambda_d_min=args.lambda_d_min, sigmas_px=[8.0, 16.0, 20.0],
        image_size=512, grid_size=128,
    ).to(device)

    # ── AMP Scaler ─────────────────────────────────────────────────────────
    use_cuda_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    best_dice = 0.0

    # ── Training Loop ──────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x = batch["image"].to(device)

            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                out = model(x)
                loss_dict = criterion(out, batch, epoch=epoch)
                loss = loss_dict["loss"]

            if use_cuda_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            epoch_loss += loss.item()

            if step % 20 == 0:
                print(
                    f"  Ep {epoch:2d}/{args.epochs}  Step {step:3d}/{len(train_loader)}  "
                    f"L={loss.item():.4f}  L_s={loss_dict['loss_s'].item():.3f}  "
                    f"L_crv={loss_dict['loss_crv'].item():.3f}  L_cldice={loss_dict['loss_cldice'].item():.3f}  "
                    f"λ_d={loss_dict['lambda_d']:.3f}"
                )

        scheduler.step()
        train_loss = epoch_loss / len(train_loader)
        el_time = time.time() - t0

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        val_dice = val_metrics["dice"]
        print(
            f"Epoch {epoch:2d}/{args.epochs} [{el_time:.1f}s]  Train Loss: {train_loss:.4f}  "
            f"Val Dice (CNN): {val_metrics['dice_cnn']*100:.2f}%  "
            f"Val Dice (Cauchy): {val_metrics['dice_raster']*100:.2f}%  "
            f"Exist Acc: {val_metrics['exist_acc']*100:.2f}%"
        )

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_dice_cnn": val_metrics["dice_cnn"],
                "val_dice_raster": val_metrics["dice_raster"],
                "val_exist_acc": val_metrics["exist_acc"],
                "lambda_d": loss_dict["lambda_d"],
                "lr": scheduler.get_last_lr()[0],
            })

        # Save Checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_dice": val_dice,
                "config": vars(args),
            }, ckpt_path)
            print(f"  ⭐ Saved new best checkpoint to {ckpt_path} (Dice: {val_dice*100:.2f}%)")

        latest_path = os.path.join(args.save_dir, "latest_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_dice": val_dice,
            "config": vars(args),
        }, latest_path)

    print(f"\n✅ Training complete! Best Validation Dice: {best_dice*100:.2f}%")


if __name__ == "__main__":
    main()
