"""
EXP_11: SurgicalCurveFormer Training Script
=============================================
Kaggle CUDA + local macOS compatible.
Uses BCRNet sigmoid annealing, AdamW optimizer, AMP, and W&B logging.
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
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

exp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if exp_root not in sys.path:
    sys.path.insert(0, exp_root)

from configs.exp11_config import EXP11Config, resolve_dataset_dir
from models.surgical_curve_former import SurgicalCurveFormer
from models.losses import SurgicalCurveFormerLoss, compute_lambda_d
from utils.dataset import SurgicalCurveFormerDataset


def parse_args():
    cfg = EXP11Config()
    p = argparse.ArgumentParser(description="EXP_11: Train SurgicalCurveFormer")
    p.add_argument("--dataset_dir",       type=str,   default=resolve_dataset_dir())
    p.add_argument("--epochs",            type=int,   default=cfg.num_epochs)
    p.add_argument("--batch_size",        type=int,   default=cfg.batch_size)
    p.add_argument("--lr",                type=float, default=5e-5)          # was 1e-5, raised for faster curve head learning
    p.add_argument("--backbone_lr_mult",  type=float, default=cfg.backbone_lr_mult)
    p.add_argument("--weight_decay",      type=float, default=cfg.weight_decay)
    p.add_argument("--anneal_center",     type=float, default=20.0)           # was 10, now 20 for longer dense phase
    p.add_argument("--anneal_slope",      type=float, default=4.0)            # was 2, now 4 for slower transition
    p.add_argument("--lambda_d_min",      type=float, default=0.05)           # NEW: floor prevents CNN forgetting
    p.add_argument("--sigma_start_px",    type=float, default=30.0)           # NEW: wide Gaussian for early gradients
    p.add_argument("--sigma_end_px",      type=float, default=2.0)            # NEW: narrow final for precision
    p.add_argument("--sigma_anneal_epochs",type=int,  default=30)             # NEW: sigma reaches 2px by epoch 30
    p.add_argument("--save_dir",          type=str,   default=cfg.save_dir)
    p.add_argument("--acpi_top_k",        type=int,   default=cfg.acpi_top_k)
    p.add_argument("--hcr_stages",        type=int,   default=3)
    p.add_argument("--amp",               action="store_true", default=cfg.use_amp)
    p.add_argument("--use_depth",         action="store_true", default=cfg.use_depth)
    p.add_argument("--wandb",             action="store_true")
    p.add_argument("--wandb_key",         type=str,   default=cfg.wandb_key)
    p.add_argument("--device",            type=str,   default="")
    p.add_argument("--num_workers",       type=int,   default=cfg.num_workers)
    return p.parse_args()


def get_device(forced: str = "") -> torch.device:
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dice_metric(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> float:
    """Binary Dice score for evaluation."""
    inter = (pred_mask & gt_mask).sum()
    return (2.0 * inter + eps) / (pred_mask.sum() + gt_mask.sum() + eps)


@torch.no_grad()
def evaluate(model, loader, device, dilate_px: int = 30) -> dict:
    """Quick validation: Dice + existence accuracy."""
    model.eval()
    all_dice = []
    all_exist_acc = []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))

    for batch in loader:
        x = batch["image"].to(device)
        out = model(x)
        B = x.shape[0]

        # Soft masks (B, M, R, R) → threshold at 0.3 → binarize
        soft = out["soft_masks"].cpu().float()  # (B, M, R, R)
        exist_p = out["exist_probs"].cpu()       # (B, M)
        exist_gt = batch["active_mask"].float()  # (B, M)

        # Existence accuracy
        pred_exist = (exist_p > 0.5).float()
        exist_acc = (pred_exist == exist_gt).float().mean().item()
        all_exist_acc.append(exist_acc)

        # Dice on full-res masks (resize soft_masks to 512)
        import torch.nn.functional as F
        soft_512 = F.interpolate(soft, size=(512, 512), mode="bilinear", align_corners=False)
        gt_masks = batch["target_masks"]  # (B, M, 512, 512)
        active = batch["active_mask"]     # (B, M)

        for b in range(B):
            for m in range(soft_512.shape[1]):
                if not active[b, m].item():
                    continue
                pred_np = (soft_512[b, m].numpy() > 0.3).astype(np.uint8)
                gt_np   = (gt_masks[b, m].numpy() > 0.5).astype(np.uint8)
                # Dilate GT as per BCRNet/TopoNet evaluation protocol (30px)
                gt_dil = cv2.dilate(gt_np, kernel)
                pred_dil = cv2.dilate(pred_np, kernel)
                d = dice_metric(pred_dil > 0, gt_dil > 0)
                all_dice.append(d)

    model.train()
    return {
        "dice": float(np.mean(all_dice)) if all_dice else 0.0,
        "exist_acc": float(np.mean(all_exist_acc)) if all_exist_acc else 0.0,
    }


def main():
    args = parse_args()
    device = get_device(args.device)

    print("=" * 70)
    print("🚀 [EXP_11] Training SurgicalCurveFormer")
    print(f"   BCRNet ACPI + HCR + EXP_10 Existence Gate + Soft Rasterizer Dice")
    print(f"   Device:     {device}")
    print(f"   Dataset:    {args.dataset_dir}")
    print(f"   Epochs:     {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"   AMP:        {args.amp}  |  Depth: {args.use_depth}")
    print(f"   ACPI top-K: {args.acpi_top_k}  |  HCR stages: {args.hcr_stages}")
    print("=" * 70)

    # ── W&B Setup ──────────────────────────────────────────────────────────
    use_wandb = args.wandb
    if use_wandb and args.wandb_key:
        try:
            import wandb
            wandb.login(key=args.wandb_key)
            wandb.init(
                project="Surgical_AI_EXP11_SurgicalCurveFormer",
                name=f"EXP11_ACPI_HCR_k{args.acpi_top_k}",
                config=vars(args),
            )
        except Exception as e:
            print(f"⚠️  W&B init failed: {e}. Continuing without logging.")
            use_wandb = False

    # ── Dataset ────────────────────────────────────────────────────────────
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
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"   Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    # ── Model ──────────────────────────────────────────────────────────────
    model = SurgicalCurveFormer(
        num_classes=3, image_size=512, in_chans=4 if args.use_depth else 3,
        fpn_channels=256, bezier_ctrl_pts=6, acpi_top_k=args.acpi_top_k,
        hcr_stages=args.hcr_stages, hcr_embed_dim=256, hcr_heads=8,
        hcr_n_ref_pts=26, hcr_deform_pts=4,
        vit_backbone="vit_base_patch16_224", vit_embed_dim=768,
        vit_pretrained=True, exist_attn_heads=8,
        raster_render_size=128, raster_num_samples=64, raster_sigma_px=2.0,
    ).to(device)

    # ── Optimizer with differential LR ────────────────────────────────────
    param_groups = model.get_param_groups(
        base_lr=args.lr, backbone_lr_mult=args.backbone_lr_mult
    )
    optimizer = Adam(param_groups, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    # ── Loss ───────────────────────────────────────────────────────────────
    criterion = SurgicalCurveFormerLoss(
        num_hcr_stages=args.hcr_stages,
        lambda_s=10.0, lambda_ind=1.0, lambda_cs=1.0,
        lambda_crv=1.0, lambda_exist=1.5, lambda_dice=5.0,
        anneal_center=args.anneal_center,
        anneal_slope=args.anneal_slope,
        lambda_d_min=args.lambda_d_min,
        sigma_start_px=args.sigma_start_px,
        sigma_end_px=args.sigma_end_px,
        sigma_anneal_epochs=args.sigma_anneal_epochs,
        raster_render_size=128, raster_num_samples=64,
        raster_sigma_px=2.0, target_size=512,
    ).to(device)

    # ── AMP scaler ─────────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    # ── Checkpoint dir ─────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_dice = 0.0

    # ── Training Loop ──────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        epoch_losses = {
            "L_total": [], "L_s": [], "L_ind": [], "L_cs": [], "L_crv": [],
            "L_exist": [], "L_dice": []
        }
        lam_d = compute_lambda_d(epoch, center=10.0, slope=2.0)

        for step, batch in enumerate(train_loader):
            x = batch["image"].to(device, non_blocking=True)
            target = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
                if k != "img_path" and isinstance(v, torch.Tensor)
            }
            # Rename to match criterion's expected keys
            target["target_masks"]        = target.pop("target_masks", target.get("target_masks"))
            target["acpi_target_score"]   = target.get("acpi_target_score")
            target["target_ctrl_pts"]     = target.get("target_ctrl_pts")
            target["active_mask"]         = target.get("active_mask")
            target["target_render_masks"] = target.get("target_render_masks")

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                out = model(x)
                losses = criterion(out, target, epoch=epoch)
                loss = losses["L_total"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k].append(losses[k].item() if torch.is_tensor(losses[k]) else losses[k])

            if step % 20 == 0:
                sigma_now = losses.get("current_sigma_px", torch.tensor(0.0))
                sigma_val = sigma_now.item() if torch.is_tensor(sigma_now) else sigma_now
                print(
                    f"  Ep {epoch+1:>3}/{args.epochs}  Step {step:>4}/{len(train_loader)}"
                    f"  L={loss.item():.4f}  L_s={losses['L_s'].item():.3f}"
                    f"  L_crv={losses['L_crv'].item():.3f}  λ_d={lam_d:.3f}"
                    f"  σ={sigma_val:.1f}px"
                )

        scheduler.step()

        # ── Epoch summary ─────────────────────────────────────────────────
        avg_loss = np.mean(epoch_losses["L_total"]) if epoch_losses["L_total"] else 0.0
        elapsed = time.time() - t0

        log_dict = {f"train/{k}": np.mean(v) for k, v in epoch_losses.items() if v}
        log_dict["train/lambda_d"] = lam_d
        log_dict["train/epoch"] = epoch + 1

        print(
            f"Epoch {epoch+1:>3}/{args.epochs}  "
            f"L={avg_loss:.4f}  λ_d={lam_d:.3f}  "
            f"LR={optimizer.param_groups[-1]['lr']:.2e}  "
            f"Time={elapsed:.1f}s"
        )

        # ── Validation ───────────────────────────────────────────────────
        if len(val_ds) > 0 and (epoch + 1) % 5 == 0:
            metrics = evaluate(model, val_loader, device, dilate_px=30)
            dice = metrics["dice"]
            log_dict["val/dice_30px"] = dice
            log_dict["val/exist_acc"] = metrics["exist_acc"]
            print(
                f"  ↳ Val Dice (30px dilate): {dice*100:.2f}%  "
                f"Exist Acc: {metrics['exist_acc']*100:.1f}%"
            )

            if dice > best_dice:
                best_dice = dice
                ckpt_path = os.path.join(args.save_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "args": vars(args),
                }, ckpt_path)
                print(f"  ✅ New best Dice={best_dice*100:.2f}% → saved to {ckpt_path}")

        # ── Periodic checkpoint ───────────────────────────────────────────
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_ep{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "args": vars(args),
            }, ckpt_path)

        if use_wandb:
            try:
                import wandb
                wandb.log(log_dict)
            except Exception:
                pass

    print(f"\n🏁 Training complete. Best Val Dice: {best_dice*100:.2f}%")
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
