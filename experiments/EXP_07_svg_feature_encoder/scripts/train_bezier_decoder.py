"""
EXP_07 — Train SVG Bézier Spline Transformer.

Full pipeline:
  1. SVG Vector-Aware Encoder (Swin backbone + geometric neck)
  2. Iterative Bézier Spline Decoder (6 layers, coordinate probing + refinement)
  3. Four loss functions: L_curve (Chamfer), L_cls (Focal), L_endpoint, L_smooth
  4. Validation with Dice & IoU metrics (rasterized curves)
  5. Fixed anchor progression tracking, WandB logging, checkpointing
"""

import os
import sys
import time
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*Glyph.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path setup
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.dirname(os.path.dirname(EXP_DIR))
for p in [EXP_DIR, WORKSPACE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configs.config_svg_encoder import config
from models.svg_vector_encoder import SVGVectorAwareFeatureEncoder
from models.bezier_decoder import BezierSplineDecoder
from models.bezier_losses import BezierSplineLoss, evaluate_bezier_curve
from utils.dataset_bezier import SurgicalBezierDataset


# ═══════════════════════════════════════════════════════════════════════
# Validation Metrics (Dice & IoU via rasterized curves — no gradient)
# ═══════════════════════════════════════════════════════════════════════

def rasterize_bezier_mask(control_points: np.ndarray, img_size: int = 1024, thickness: int = 20) -> np.ndarray:
    """Rasterize a single Bézier curve into a binary mask."""
    t = np.linspace(0.0, 1.0, 100).reshape(-1, 1)
    p0, c1, c2, p3 = control_points[0], control_points[1], control_points[2], control_points[3]
    curve = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * c1 + 3 * (1 - t) * (t ** 2) * c2 + (t ** 3) * p3
    pts_pix = (curve * img_size).astype(np.int32).reshape((-1, 1, 2))

    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    cv2.polylines(mask, [pts_pix], isClosed=False, color=1, thickness=thickness)
    return mask.astype(np.float32)


def compute_val_metrics(
    pred_control_points: torch.Tensor,
    pred_class_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_classes: torch.Tensor,
    valid_mask: torch.Tensor,
    matches: list,
    thickness: int = 20,
    eps: float = 1e-6,
) -> dict:
    """
    Compute Dice and IoU between rasterized predicted Bézier and GT masks.
    These are pure evaluation metrics — no gradients.
    """
    B = pred_control_points.shape[0]
    dices, ious = [], []

    for b in range(B):
        for pred_idx, gt_idx in matches[b]:
            # Rasterize predicted curve
            cp = pred_control_points[b, pred_idx].cpu().numpy()
            pred_mask = rasterize_bezier_mask(cp, img_size=1024, thickness=thickness)

            # GT mask
            gt_m = gt_masks[b, gt_idx].cpu().numpy()
            gt_bin = (gt_m > 0.5).astype(np.float32)

            # Dice
            inter = (pred_mask * gt_bin).sum()
            dice = (2.0 * inter + eps) / (pred_mask.sum() + gt_bin.sum() + eps)
            dices.append(dice)

            # IoU
            union = ((pred_mask + gt_bin) > 0.5).astype(np.float32).sum()
            iou = (inter + eps) / (union + eps)
            ious.append(iou)

    return {
        "dice": np.mean(dices) if dices else 0.0,
        "iou": np.mean(ious) if ious else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════
# Anchor Progression Visualization
# ═══════════════════════════════════════════════════════════════════════

QUERY_COLORS = ["#ff3366", "#00ffcc", "#ffaa00", "#7c4dff", "#00e5ff",
                "#ff6d00", "#76ff03", "#ff1744", "#448aff", "#eeff41"]


def render_anchor_overlay(
    image_tensor, gt_polylines, gt_masks, valid_mask, gt_classes,
    pred_control_points, pred_class_logits, matches, loss_dict, total_loss,
    epoch, save_path,
):
    """Render Bézier prediction overlay for anchor progression tracking."""
    try:
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_rgb = np.clip(img_np * std + mean, 0.0, 1.0)

        class_names = ["NoObj", "Ridge", "Silhouette", "Falciform", "Gallbladder"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0d1117")

        # Left: GT contours + predicted Bézier curves
        ax1.set_facecolor("#161b22")
        ax1.imshow(img_rgb)

        # Draw GT
        active_gt = [i for i in range(valid_mask.shape[0]) if valid_mask[i] > 0]
        for idx, gi in enumerate(active_gt):
            gt_p = gt_polylines[gi].cpu().numpy()
            u = gt_p[:, 0] * 1024.0
            v = gt_p[:, 1] * 1024.0
            c_id = int(gt_classes[gi].item())
            c_name = class_names[c_id] if c_id < len(class_names) else f"Cls{c_id}"
            ax1.plot(u, v, color="#00ffcc", linewidth=2.5, label=f"GT: {c_name}" if idx == 0 else None)

        # Draw predicted Bézier curves
        cp_np = pred_control_points.cpu().numpy()
        t = np.linspace(0.0, 1.0, 80).reshape(-1, 1)
        for pred_idx, gt_idx in matches:
            cp = cp_np[pred_idx]
            p0, c1, c2, p3 = cp[0], cp[1], cp[2], cp[3]
            curve = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * c1 + 3 * (1 - t) * (t ** 2) * c2 + (t ** 3) * p3
            curve_px = curve * 1024.0
            col = QUERY_COLORS[pred_idx % len(QUERY_COLORS)]

            ax1.plot(curve_px[:, 0], curve_px[:, 1], color=col, linewidth=2.5, linestyle="--")
            ax1.scatter([p0[0] * 1024, p3[0] * 1024], [p0[1] * 1024, p3[1] * 1024],
                       color="white", s=50, zorder=5, edgecolors=col, linewidths=1.5)
            ax1.scatter([c1[0] * 1024, c2[0] * 1024], [c1[1] * 1024, c2[1] * 1024],
                       color=col, s=30, marker="s", zorder=5)

        ax1.set_title(f"Epoch {epoch:02d} | Bézier Curves (Cyan=GT, Colored=Pred)",
                      color="white", fontsize=12, fontweight="bold")
        ax1.axis("off")

        # Right: Metrics card
        ax2.set_facecolor("#161b22")
        ax2.axis("off")
        card = (
            f"[ANCHOR DIAGNOSTICS] Epoch {epoch:02d}\n"
            f"{'─' * 45}\n"
            f"TOTAL LOSS:     {total_loss:.4f}\n\n"
            f"L_curve:        {loss_dict.get('l_curve', 0):.4f}  (Chamfer)\n"
            f"L_cls:          {loss_dict.get('l_cls', 0):.4f}  (Focal)\n"
            f"L_endpoint:     {loss_dict.get('l_endpoint', 0):.4f}  (P0/P3)\n"
            f"L_smooth:       {loss_dict.get('l_smooth', 0):.4f}  (Curvature)\n\n"
            f"Matched Queries: {len(matches)} / {cp_np.shape[0]}\n"
            f"Active GT:       {len(active_gt)}\n"
        )
        ax2.text(0.05, 0.95, card, transform=ax2.transAxes, color="#e6edf3",
                 fontsize=11, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#0d1117", edgecolor="#30363d"))

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=130, facecolor=fig.get_facecolor())
        plt.close("all")
        return save_path
    except Exception as e:
        print(f"⚠️ Anchor overlay error: {e}")
        plt.close("all")
        return None


# ═══════════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Train EXP_07 SVG Bézier Spline Transformer")
    parser.add_argument("--dataset_dir", type=str, default=config.dataset_dir)
    parser.add_argument("--epochs", type=int, default=config.num_epochs)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--lr", type=float, default=config.learning_rate)
    parser.add_argument("--save_dir", type=str, default="checkpoints/EXP_07")
    parser.add_argument("--viz_interval", type=int, default=config.viz_interval)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Surgical_AI_Bezier")
    parser.add_argument("--wandb_run_name", type=str, default="EXP_07_SVG_Bezier_Spline")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("🚀 Starting Training: EXP_07 SVG Bézier Spline Transformer")
    print("=" * 70)
    print(f"📁 Dataset: {args.dataset_dir}")
    print(f"⚙️  Batch={args.batch_size}, LR={args.lr}, Epochs={args.epochs}")
    print(f"🎯 Losses: λ_curve={config.lambda_curve}, λ_cls={config.lambda_cls}, "
          f"λ_endpoint={config.lambda_endpoint}, λ_smooth={config.lambda_smooth}")

    # ── WandB ──
    use_wandb = args.wandb or ("WANDB_API_KEY" in os.environ)
    if use_wandb:
        try:
            import wandb
            api_key = os.environ.get("WANDB_API_KEY", "")
            if api_key:
                wandb.login(key=api_key)
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config={
                **vars(args),
                "lambda_curve": config.lambda_curve, "lambda_cls": config.lambda_cls,
                "lambda_endpoint": config.lambda_endpoint, "lambda_smooth": config.lambda_smooth,
                "num_queries": config.num_queries, "num_decoder_layers": config.num_decoder_layers,
            })
            print(f"📊 WandB initialized: {args.wandb_project}/{args.wandb_run_name}")
        except Exception as e:
            print(f"⚠️ WandB init failed: {e}")
            use_wandb = False

    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"💻 Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("💻 Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("💻 Device: CPU")

    os.makedirs(args.save_dir, exist_ok=True)
    anchor_dir = os.path.join(args.save_dir, "anchor_progression")
    os.makedirs(anchor_dir, exist_ok=True)

    # ── Dataset ──
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Dataset path not found: {args.dataset_dir}")
        return

    train_ds = SurgicalBezierDataset(
        dataset_dir=args.dataset_dir, num_instances=config.num_queries,
        num_polyline_points=50, mode="train", stroke_thickness=config.stroke_thickness_dice,
    )
    val_ds = SurgicalBezierDataset(
        dataset_dir=args.dataset_dir, num_instances=config.num_queries,
        num_polyline_points=50, mode="val", stroke_thickness=config.stroke_thickness_dice,
    )
    print(f"📊 Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=len(train_ds) > args.batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=config.num_workers, drop_last=False,
    ) if len(val_ds) > 0 else None

    # ── Fixed Anchor Frames ──
    fixed_train_anchor = None
    for i in range(min(len(train_ds), 50)):
        s = train_ds[i]
        if s["valid_mask"].sum() >= 2:
            fixed_train_anchor = {k: v.unsqueeze(0).to(device) for k, v in s.items()}
            break
    if fixed_train_anchor is None and len(train_ds) > 0:
        fixed_train_anchor = {k: v.unsqueeze(0).to(device) for k, v in train_ds[0].items()}

    fixed_val_anchor = None
    if len(val_ds) > 0:
        for i in range(min(len(val_ds), 50)):
            s = val_ds[i]
            if s["valid_mask"].sum() >= 2:
                fixed_val_anchor = {k: v.unsqueeze(0).to(device) for k, v in s.items()}
                break
        if fixed_val_anchor is None:
            fixed_val_anchor = {k: v.unsqueeze(0).to(device) for k, v in val_ds[0].items()}

    # ── Model ──
    encoder = SVGVectorAwareFeatureEncoder(config).to(device)
    decoder = BezierSplineDecoder(
        num_queries=config.num_queries,
        num_layers=config.num_decoder_layers,
        d_model=config.embed_dim,
        nhead=config.num_heads,
        num_sample_t=config.num_sample_t,
        num_classes=config.num_classes,
    ).to(device)
    loss_fn = BezierSplineLoss(config).to(device)

    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
    print(f"🧠 Model: Encoder={sum(p.numel() for p in encoder.parameters()):,} | "
          f"Decoder={sum(p.numel() for p in decoder.parameters()):,} | Total={total_params:,}")

    # ── Optimizer (backbone 0.1x LR) ──
    backbone_params, head_params = [], []
    for name, param in list(encoder.named_parameters()) + list(decoder.named_parameters()):
        if not param.requires_grad:
            continue
        if "swin_encoder" in name or "pixel_decoder" in name or "fallback_backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * config.backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Training Loop ──
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        epoch_loss_dict = {"l_curve": 0.0, "l_cls": 0.0, "l_endpoint": 0.0, "l_smooth": 0.0}
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{args.epochs} [Train]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            gt_poly = batch["target_polylines"].to(device)
            gt_cls = batch["target_classes"].to(device)
            gt_masks = batch["target_masks"].to(device)
            valid = batch["valid_mask"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                enc_out = encoder(images)
                dec_out = decoder(enc_out)

                loss_out = loss_fn(
                    dec_out["final_control_points"],
                    dec_out["class_logits"],
                    gt_poly, gt_cls, valid,
                )
                loss = loss_out["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                max_norm=config.gradient_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            for k in epoch_loss_dict:
                epoch_loss_dict[k] += loss_out["loss_dict"].get(k, 0.0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "curve": f"{loss_out['loss_dict']['l_curve']:.4f}",
                "cls": f"{loss_out['loss_dict']['l_cls']:.4f}",
            })

        scheduler.step()
        elapsed = time.time() - start_time
        n_batches = max(len(train_loader), 1)
        avg_train_loss = epoch_loss / n_batches
        for k in epoch_loss_dict:
            epoch_loss_dict[k] /= n_batches

        # ── Validation ──
        avg_val_loss = avg_train_loss
        val_loss_dict = dict(epoch_loss_dict)
        val_dice, val_iou = 0.0, 0.0

        if val_loader is not None and len(val_loader) > 0:
            encoder.eval()
            decoder.eval()
            val_epoch_loss = 0.0
            val_epoch_loss_dict = {"l_curve": 0.0, "l_cls": 0.0, "l_endpoint": 0.0, "l_smooth": 0.0}
            all_dices, all_ious = [], []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch:2d}/{args.epochs} [Val]", leave=False):
                    images = batch["image"].to(device)
                    gt_poly = batch["target_polylines"].to(device)
                    gt_cls = batch["target_classes"].to(device)
                    gt_masks_v = batch["target_masks"].to(device)
                    valid = batch["valid_mask"].to(device)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        enc_out = encoder(images)
                        dec_out = decoder(enc_out)
                        v_loss_out = loss_fn(
                            dec_out["final_control_points"],
                            dec_out["class_logits"],
                            gt_poly, gt_cls, valid,
                        )

                    val_epoch_loss += v_loss_out["loss"].item()
                    for k in val_epoch_loss_dict:
                        val_epoch_loss_dict[k] += v_loss_out["loss_dict"].get(k, 0.0)

                    # Dice & IoU (no gradient)
                    m = compute_val_metrics(
                        dec_out["final_control_points"], dec_out["class_logits"],
                        gt_masks_v, gt_cls, valid, v_loss_out["matches"],
                        thickness=config.stroke_thickness_dice,
                    )
                    all_dices.append(m["dice"])
                    all_ious.append(m["iou"])

            n_val = max(len(val_loader), 1)
            avg_val_loss = val_epoch_loss / n_val
            for k in val_epoch_loss_dict:
                val_loss_dict[k] = val_epoch_loss_dict[k] / n_val
            val_dice = np.mean(all_dices) if all_dices else 0.0
            val_iou = np.mean(all_ious) if all_ious else 0.0

        print(
            f"Epoch [{epoch:2d}/{args.epochs}] ({elapsed:.1f}s) | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Dice: {val_dice * 100:.1f}% | Val IoU: {val_iou * 100:.1f}%",
            flush=True,
        )

        # ── Anchor Progression ──
        if fixed_train_anchor is not None:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    a_enc = encoder(fixed_train_anchor["image"])
                    a_dec = decoder(a_enc)
                    a_loss = loss_fn(
                        a_dec["final_control_points"], a_dec["class_logits"],
                        fixed_train_anchor["target_polylines"],
                        fixed_train_anchor["target_classes"],
                        fixed_train_anchor["valid_mask"],
                    )

                anchor_path = os.path.join(anchor_dir, f"epoch_{epoch:02d}_train_anchor.png")
                render_anchor_overlay(
                    fixed_train_anchor["image"][0],
                    fixed_train_anchor["target_polylines"][0],
                    fixed_train_anchor["target_masks"][0],
                    fixed_train_anchor["valid_mask"][0],
                    fixed_train_anchor["target_classes"][0],
                    a_dec["final_control_points"][0],
                    a_dec["class_logits"][0],
                    a_loss["matches"][0],
                    a_loss["loss_dict"],
                    a_loss["loss"].item(),
                    epoch, anchor_path,
                )

        if fixed_val_anchor is not None:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    v_enc = encoder(fixed_val_anchor["image"])
                    v_dec = decoder(v_enc)
                    v_loss = loss_fn(
                        v_dec["final_control_points"], v_dec["class_logits"],
                        fixed_val_anchor["target_polylines"],
                        fixed_val_anchor["target_classes"],
                        fixed_val_anchor["valid_mask"],
                    )

                anchor_path = os.path.join(anchor_dir, f"epoch_{epoch:02d}_val_anchor.png")
                render_anchor_overlay(
                    fixed_val_anchor["image"][0],
                    fixed_val_anchor["target_polylines"][0],
                    fixed_val_anchor["target_masks"][0],
                    fixed_val_anchor["valid_mask"][0],
                    fixed_val_anchor["target_classes"][0],
                    v_dec["final_control_points"][0],
                    v_dec["class_logits"][0],
                    v_loss["matches"][0],
                    v_loss["loss_dict"],
                    v_loss["loss"].item(),
                    epoch, anchor_path,
                )

        print(f"  [ANCHOR] Epoch {epoch:02d} saved to '{anchor_dir}/'", flush=True)

        # ── WandB Logging ──
        if use_wandb:
            try:
                import wandb
                log = {
                    "epoch": epoch,
                    "train/total_loss": avg_train_loss,
                    "train/l_curve": epoch_loss_dict["l_curve"],
                    "train/l_cls": epoch_loss_dict["l_cls"],
                    "train/l_endpoint": epoch_loss_dict["l_endpoint"],
                    "train/l_smooth": epoch_loss_dict["l_smooth"],
                    "val/total_loss": avg_val_loss,
                    "val/l_curve": val_loss_dict["l_curve"],
                    "val/l_cls": val_loss_dict["l_cls"],
                    "val/l_endpoint": val_loss_dict["l_endpoint"],
                    "val/l_smooth": val_loss_dict["l_smooth"],
                    "val/dice_pct": val_dice * 100.0,
                    "val/iou_pct": val_iou * 100.0,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                # Log anchor images
                train_anchor_img = os.path.join(anchor_dir, f"epoch_{epoch:02d}_train_anchor.png")
                val_anchor_img = os.path.join(anchor_dir, f"epoch_{epoch:02d}_val_anchor.png")
                if os.path.exists(train_anchor_img):
                    log["anchor/train"] = wandb.Image(train_anchor_img, caption=f"Ep{epoch} Train Anchor")
                if os.path.exists(val_anchor_img):
                    log["anchor/val"] = wandb.Image(val_anchor_img, caption=f"Ep{epoch} Val Anchor")
                wandb.log(log)
            except Exception as e:
                print(f"⚠️ WandB log error: {e}")

        # ── Checkpoint ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.save_dir, "best_bezier_spline.pth")
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "config": config,
            }, ckpt_path)
            print(f"  💾 Best checkpoint saved: '{ckpt_path}' (Val Loss: {best_val_loss:.4f})")

    print("=" * 70)
    print(f"✅ EXP_07 Training Complete! Best Val Loss = {best_val_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
