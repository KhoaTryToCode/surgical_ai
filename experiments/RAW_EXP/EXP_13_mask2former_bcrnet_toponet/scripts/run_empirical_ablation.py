#!/usr/bin/env python3
"""
EXP_13: Empirical Ablation Benchmark Suite
==========================================
Trains and validates each of the 4 core architectures under identical conditions:
1. 'toponet'     : ResNet FPN + Soft Centerline clDice (No masked attention, no Bézier)
2. 'mask2former' : ResNet FPN + Multi-Scale Masked Attention (No clDice, no Bézier)
3. 'bcrnet'      : ResNet FPN + 5th-Order Bézier + Chamfer Loss (No masked attention, no clDice)
4. 'master_exp13': Full Master Synthesis (Mask2Former + TopoNet Loss + BCRNet)

Strictly evaluates on L3D Validation Set (122 images).
Outputs empirical metrics: Per-class Dice, Macro Mean Dice, IoU, ASSD.
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import json
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP13_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_DIR = os.path.dirname(os.path.dirname(EXP13_DIR))

for p in [EXP13_DIR, WORKSPACE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configs.exp13_config import EXP13Config
from utils.dataset import Mask2FormerBCRNetDataset
from models.mask2former_bcrnet import Mask2FormerBCRNet
from models.joint_losses import JointUnifiedLoss
from utils.bezier_ops import evaluate_bezier_torch


def compute_metrics(pred_sem: np.ndarray, gt_sem: np.ndarray, num_classes: int = 3):
    """
    Computes per-class Dice, Macro Mean Dice, IoU, and ASSD approximation.
    Classes: 1: Ridge, 2: Silhouette, 3: Falciform Ligament.
    """
    smooth = 1e-5
    dices = {}
    ious = {}
    assds = {}

    for c in range(1, num_classes + 1):
        p_c = (pred_sem == c).astype(np.uint8)
        g_c = (gt_sem == c).astype(np.uint8)

        intersection = np.sum(p_c * g_c)
        union = np.sum(p_c) + np.sum(g_c)

        # Per-class Dice
        dice_c = (2.0 * intersection + smooth) / (union + smooth)
        dices[c] = float(dice_c)

        # Per-class IoU
        iou_c = (intersection + smooth) / (np.sum(p_c | g_c) + smooth)
        ious[c] = float(iou_c)

        # ASSD (in pixels) using distance transform
        if np.sum(p_c) > 0 and np.sum(g_c) > 0:
            dt_gt = cv2.distanceTransform(1 - g_c, cv2.DIST_L2, 3)
            dt_pred = cv2.distanceTransform(1 - p_c, cv2.DIST_L2, 3)
            d_p2g = np.mean(dt_gt[p_c > 0])
            d_g2p = np.mean(dt_pred[g_c > 0])
            assds[c] = float(0.5 * (d_p2g + d_g2p))
        else:
            assds[c] = 80.0

    macro_dice = float(np.mean([dices[c] for c in range(1, num_classes + 1)]))
    macro_iou = float(np.mean([ious[c] for c in range(1, num_classes + 1)]))
    macro_assd = float(np.mean([assds[c] for c in range(1, num_classes + 1)]))

    return {
        "macro_dice": macro_dice,
        "macro_iou": macro_iou,
        "macro_assd": macro_assd,
        "per_class_dice": dices,
        "per_class_iou": ious,
        "per_class_assd": assds,
    }


def rasterize_bezier_curves(curve_pts: np.ndarray, scores: np.ndarray, H: int, W: int, linewidth: int = 15):
    """
    Rasterizes predicted Bézier curves into a 4-class semantic map with standard line dilation.
    curve_pts: (3, 26, 2) in normalized [0, 1] coordinates.
    scores: (3,) confidence scores.
    """
    sem_map = np.zeros((H, W), dtype=np.int64)
    for c in range(3):
        if scores[c] >= 0.15:
            pts = (curve_pts[c] * np.array([W - 1, H - 1])).astype(np.int32)
            mask_c = np.zeros((H, W), dtype=np.uint8)
            for i in range(len(pts) - 1):
                cv2.line(mask_c, tuple(pts[i]), tuple(pts[i + 1]), 1, linewidth)
            sem_map[mask_c > 0] = c + 1
    return sem_map


def train_and_eval_model(model_type: str, args):
    device = torch.device(args.device)
    cfg = EXP13Config()
    cfg.image_size = args.image_size
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.num_m2f_decoder_layers = 3
    cfg.hcr_stages = 2
    cfg.skel_num_iter = 5

    # Configure loss weights based on model_type
    if model_type == "toponet":
        # TopoNet: Standard CE + soft clDice (No masked attention, No Bézier)
        cfg.lambda_m2f_ce = 1.0
        cfg.lambda_m2f_mask = 0.0
        cfg.lambda_toponet = 2.0
        cfg.lambda_crv = 0.0
        cfg.lambda_tangent = 0.0
        cfg.lambda_contain = 0.0
    elif model_type == "mask2former":
        # Mask2Former: Masked Cross-Attention + CE + Mask Dice (No clDice, No Bézier)
        cfg.lambda_m2f_ce = 1.0
        cfg.lambda_m2f_mask = 3.0
        cfg.lambda_toponet = 0.0
        cfg.lambda_crv = 0.0
        cfg.lambda_tangent = 0.0
        cfg.lambda_contain = 0.0
    elif model_type == "bcrnet":
        # BCRNet: Bézier Curve Regression + Chamfer L1 + Tangents (No Masked Attn, No clDice)
        cfg.lambda_m2f_ce = 0.5
        cfg.lambda_m2f_mask = 0.0
        cfg.lambda_toponet = 0.0
        cfg.lambda_crv = 4.0
        cfg.lambda_tangent = 1.0
        cfg.lambda_contain = 0.0
    elif model_type == "master_exp13":
        # Master Synthesis: Mask2Former + TopoNet clDice + BCRNet HCR + Mutual Containment
        cfg.lambda_m2f_ce = 1.0
        cfg.lambda_m2f_mask = 2.0
        cfg.lambda_toponet = 1.5
        cfg.lambda_crv = 3.0
        cfg.lambda_tangent = 0.5
        cfg.lambda_contain = 0.5

    print(f"\n{'='*70}")
    print(f"🚀 [EMPIRICAL ABLATION] Model: {model_type.upper()}")
    print(f"   Epochs: {cfg.epochs} | Batch: {cfg.batch_size} | LR: {cfg.lr} | Resolution: {cfg.image_size}x{cfg.image_size}")
    print(f"   Loss Weights: CE={cfg.lambda_m2f_ce}, Mask={cfg.lambda_m2f_mask}, Topo={cfg.lambda_toponet}, Crv={cfg.lambda_crv}")
    print(f"{'='*70}")

    # Datasets
    train_ds = Mask2FormerBCRNetDataset(args.dataset_dir, mode="Train", image_size=cfg.image_size, use_depth=True)
    val_ds = Mask2FormerBCRNetDataset(args.dataset_dir, mode="Val", image_size=cfg.image_size, use_depth=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    model = Mask2FormerBCRNet(cfg).to(device)
    loss_fn = JointUnifiedLoss(
        num_classes=cfg.num_classes,
        lambda_m2f_ce=cfg.lambda_m2f_ce,
        lambda_m2f_mask=cfg.lambda_m2f_mask,
        lambda_toponet=cfg.lambda_toponet,
        lambda_crv=cfg.lambda_crv,
        lambda_tangent=cfg.lambda_tangent,
        lambda_contain=cfg.lambda_contain,
        skel_num_iter=cfg.skel_num_iter,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    best_val_dice = 0.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{model_type}] Ep {epoch:02d}/{cfg.epochs:02d} Train", leave=False)

        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            optimizer.zero_grad()
            images = batch["image"].to(device)
            for k in ["gt_semantic", "gt_masks", "gt_ctrl_pts", "gt_sample_pts", "gt_exists"]:
                batch[k] = batch[k].to(device)

            outputs = model(images, target_size=(cfg.image_size, cfg.image_size))
            loss_dict = loss_fn(outputs, batch)
            loss = loss_dict["loss_total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        scheduler.step()
        denom = min(len(train_loader), args.max_train_batches or len(train_loader))
        avg_train_loss = train_loss / max(1, denom)

        # Validation Evaluation
        model.eval()
        val_macro_dices = []
        val_macro_ious = []
        val_macro_assds = []
        val_ridge_dices = []
        val_sil_dices = []
        val_falc_dices = []

        with torch.no_grad():
            for v_idx, batch in enumerate(val_loader):
                if args.max_val_batches and v_idx >= args.max_val_batches:
                    break
                images = batch["image"].to(device)
                gt_sem = batch["gt_semantic"].squeeze(0).cpu().numpy()

                outputs = model(images, target_size=(cfg.image_size, cfg.image_size))

                if model_type == "bcrnet":
                    # BCRNet evaluates rasterized 5th-order Bézier curves
                    curve_pts = outputs["final_curve_pts"][0].cpu().numpy()  # (3, 5, 26, 2)
                    scores = outputs["final_scores"][0].cpu().numpy()        # (3, 5)
                    best_k = np.argmax(scores, axis=-1)
                    best_curves = np.stack([curve_pts[c, best_k[c]] for c in range(3)], axis=0)
                    best_scores = np.stack([scores[c, best_k[c]] for c in range(3)], axis=0)
                    pred_sem = rasterize_bezier_curves(best_curves, best_scores, cfg.image_size, cfg.image_size)
                else:
                    # TopoNet, Mask2Former, Master Synthesis evaluate semantic logits
                    pred_sem = outputs["semantic_logits"].argmax(dim=1).squeeze(0).cpu().numpy()

                m = compute_metrics(pred_sem, gt_sem)
                val_macro_dices.append(m["macro_dice"])
                val_macro_ious.append(m["macro_iou"])
                val_macro_assds.append(m["macro_assd"])
                val_ridge_dices.append(m["per_class_dice"][1])
                val_sil_dices.append(m["per_class_dice"][2])
                val_falc_dices.append(m["per_class_dice"][3])

        mean_val_dice = float(np.mean(val_macro_dices)) * 100.0
        mean_val_iou = float(np.mean(val_macro_ious)) * 100.0
        mean_val_assd = float(np.mean(val_macro_assds))
        mean_ridge = float(np.mean(val_ridge_dices)) * 100.0
        mean_sil = float(np.mean(val_sil_dices)) * 100.0
        mean_falc = float(np.mean(val_falc_dices)) * 100.0

        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_dice": mean_val_dice,
            "val_iou": mean_val_iou,
            "val_assd": mean_val_assd,
            "val_ridge": mean_ridge,
            "val_sil": mean_sil,
            "val_falc": mean_falc,
        })

        print(
            f"   Ep {epoch:02d}/{cfg.epochs:02d} | Train Loss: {avg_train_loss:.3f} | "
            f"Val Dice: {mean_val_dice:.2f}% (Ridge: {mean_ridge:.1f}%, Sil: {mean_sil:.1f}%, Falc: {mean_falc:.1f}%) | "
            f"IoU: {mean_val_iou:.2f}% | ASSD: {mean_val_assd:.1f}px"
        )

    print(f"✅ [{model_type.upper()}] Training Complete! Best Val Dice: {best_val_dice:.2f}%\n")
    return {
        "model_type": model_type,
        "best_val_dice": best_val_dice,
        "final_metrics": history[-1],
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Empirical Ablation Benchmark")
    parser.add_argument("--dataset_dir", type=str, default="data/L3D")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--models", nargs="+", default=["toponet", "mask2former", "bcrnet", "master_exp13"])
    parser.add_argument("--max_train_batches", type=int, default=None, help="Limit training batches per epoch")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Limit validation batches per epoch")
    parser.add_argument("--output_json", type=str, default="experiments/EXP_13_mask2former_bcrnet_toponet/results/empirical_ablation_results.json")
    args = parser.parse_args()

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    print("=" * 80)
    print("🔬 EMPIRICAL BENCHMARK SUITE: ALL 4 PARADIGMS")
    print(f"   Models to run: {args.models}")
    print(f"   Dataset:       {args.dataset_dir}")
    print(f"   Epochs:        {args.epochs} per model | Batch: {args.batch_size} | Size: {args.image_size}x{args.image_size}")
    print(f"   Device:        {args.device}")
    print("=" * 80)

    results = {}
    for model_type in args.models:
        res = train_and_eval_model(model_type, args)
        results[model_type] = res

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("📊 FINAL EMPIRICAL ABLATION COMPARISON TABLE (L3D VALIDATION SET)")
    print("=" * 80)
    print(f"{'Model Architecture':<22} | {'Val Dice':<10} | {'Ridge Dice':<11} | {'Sil Dice':<10} | {'Falc Dice':<10} | {'Val IoU':<9} | {'ASSD (px)':<9}")
    print("-" * 95)
    for m_type in args.models:
        final = results[m_type]["final_metrics"]
        print(
            f"{m_type.upper():<22} | {final['val_dice']:>8.2f}% | "
            f"{final['val_ridge']:>9.2f}% | {final['val_sil']:>8.2f}% | {final['val_falc']:>8.2f}% | "
            f"{final['val_iou']:>7.2f}% | {final['val_assd']:>7.2f}px"
        )
    print("=" * 80)
    print(f"Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
