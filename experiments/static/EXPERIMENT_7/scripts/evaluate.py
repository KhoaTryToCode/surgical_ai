import os
import sys
import time
import json
import zipfile
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# NumPy 2.0 compatibility
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
    np.PINF = np.inf
    np.NINF = -np.inf

# Ensure workspace root is on sys.path
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_7.utils.dataset import L3DManifoldDataset, IMAGENET_MEAN, IMAGENET_STD
from experiments.EXPERIMENT_7.utils.metrics import evaluate_frame_metrics
from experiments.EXPERIMENT_7.models.manifold_steered_mask2former import load_manifold_steered_model

# Canonical colors
COLOR_MAP_RGB = {
    0: (0, 0, 0),       # BG
    1: (34, 197, 94),   # Ridge: Green (#22c55e)
    2: (239, 68, 68),   # Silhouette: Red (#ef4444)
    3: (59, 130, 246)   # Falciform: Blue (#3b82f6)
}

ATLAS_COLORS_BGR = [
    (0, 215, 255),  # 0: falc_root (Gold)
    (255, 0, 220),  # 1: falc_mid (Magenta)
    (255, 255, 0),  # 2: falc_notch (Cyan)
    (0, 255, 255),  # 3: left_tip (Yellow)
    (0, 140, 255),  # 4: left_sil (Orange)
    (100, 255, 100),# 5: left_ridge (Light green)
    (200, 200, 0),  # 6: left_body
    (255, 100, 255),# 7: right_tip (Pink)
    (0, 100, 255),  # 8: right_sil (Dark orange)
    (50, 200, 50),  # 9: right_ridge
    (0, 200, 200),  # 10: right_body
]


def rasterize_class_map(masks_queries_logits, class_queries_logits, canvas_size=1024):
    """
    Standard Mask2Former post-processing:
    Argmax over query probabilities multiplied by sigmoid mask probabilities.
    """
    masks_queries_logits = masks_queries_logits.float()
    class_queries_logits = class_queries_logits.float()
    
    # 1. Resize mask logits to canvas_size
    masks = torch.nn.functional.interpolate(
        masks_queries_logits.unsqueeze(0),
        size=(canvas_size, canvas_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).sigmoid()  # (100, H, W)
    
    # 2. Query classification probabilities
    cls_probs = torch.softmax(class_queries_logits, dim=-1)  # (100, 5), last is BG
    fg_cls_probs = cls_probs[:, :4]  # (100, 4)
    
    # 3. Probability combination
    sem_probs = torch.einsum("qc,qhw->chw", fg_cls_probs, masks)  # (4, H, W)
    
    # 4. Argmax and background suppression
    pred_map = sem_probs.argmax(dim=0).cpu().numpy().astype(np.int64)
    max_prob = sem_probs.max(dim=0)[0].cpu().numpy()
    pred_map[max_prob < 0.25] = 0
    return pred_map


def render_patient_40_diagnostic(orig_rgb_norm, gt_mask, pred_map, pred_coords, gt_coords, gt_vis, pred_vis, out_path, metrics):
    """
    Renders 4-panel diagnostic montage (2048x2048) for Patient 40 validation frames.
    """
    rgb = (orig_rgb_norm.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    
    # Panel 2: GT Overlay + Visible GT Atlas Points
    p2 = bgr.copy()
    for c_id in [1, 2, 3]:
        mask_c = (gt_mask == c_id)
        if mask_c.any():
            col = COLOR_MAP_RGB[c_id][::-1]
            p2[mask_c] = (p2[mask_c] * 0.4 + np.array(col) * 0.6).astype(np.uint8)
    for k in range(min(11, len(gt_vis))):
        if gt_vis[k] > 0.5:
            pt = (gt_coords[k] * float(W)).astype(int)
            cv2.circle(p2, tuple(pt), 10, (0, 0, 0), -1)
            cv2.circle(p2, tuple(pt), 7, ATLAS_COLORS_BGR[k % len(ATLAS_COLORS_BGR)], -1)
            cv2.circle(p2, tuple(pt), 2, (255, 255, 255), -1)
            
    # Panel 3: Pred Overlay + Predicted Atlas Points
    p3 = bgr.copy()
    for c_id in [1, 2, 3]:
        mask_c = (pred_map == c_id)
        if mask_c.any():
            col = COLOR_MAP_RGB[c_id][::-1]
            p3[mask_c] = (p3[mask_c] * 0.4 + np.array(col) * 0.6).astype(np.uint8)
    for k in range(min(11, len(pred_vis))):
        # Only draw if predicted visible
        if torch.sigmoid(torch.tensor(pred_vis[k])).item() > 0.5:
            pt = (pred_coords[k] * float(W)).astype(int)
            cv2.circle(p3, tuple(pt), 10, (0, 0, 0), -1)
            cv2.circle(p3, tuple(pt), 7, ATLAS_COLORS_BGR[k % len(ATLAS_COLORS_BGR)], -1)
            cv2.circle(p3, tuple(pt), 2, (255, 255, 255), -1)

    # Panel 4: True Error Map (Green=TP, Yellow=Class Error, Cyan=FP, Red=FN)
    p4 = np.zeros_like(bgr)
    p4[:] = (18, 22, 28)
    tp_correct  = (pred_map == gt_mask) & (gt_mask > 0)
    class_error = (pred_map > 0) & (gt_mask > 0) & (pred_map != gt_mask)
    fp          = (pred_map > 0) & (gt_mask == 0)
    fn          = (pred_map == 0) & (gt_mask > 0)
    
    p4[tp_correct]  = (34, 197, 94)   # Green
    p4[class_error] = (0, 190, 255)   # Yellow/Orange
    p4[fp]          = (240, 160, 40)  # Cyan
    p4[fn]          = (0, 0, 230)     # Red

    # Stitch 2x2 Montage
    top = np.hstack([bgr, p2])
    bot = np.hstack([p3, p4])
    montage = np.vstack([top, bot])
    
    # Headers
    cv2.putText(montage, "1. Input Laparoscopic Frame", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(montage, "2. Ground Truth + 11-Atlas Points", (W + 30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(montage, f"3. Prediction (EXP_7) [Dice: {metrics['macro_dice']*100:.1f}%]", (30, H + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 150), 2)
    cv2.putText(montage, "4. Error Map (Grn=TP, Ylw=ClsErr, Red=FN, Cyn=FP)", (W + 30, H + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, montage, [cv2.IMWRITE_JPEG_QUALITY, 88])


def create_results_zip(source_dir, output_zip_path):
    """
    Packages evaluation outputs, checkpoints, and Patient 40 diagnostics into results.zip.
    """
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.zip'):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    print(f"📦 Results archive packaged: {output_zip_path} ({os.path.getsize(output_zip_path)/(1024*1024):.2f} MB)")


def run_evaluation(model, split="Val", data_dir=None, out_dir=None, device="cpu", render_p40=True):
    model.eval()
    dataset = L3DManifoldDataset(split=split, data_dir=data_dir, is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    records = []
    latencies = []
    p40_dir = os.path.join(out_dir, "patient_40_diagnostics") if out_dir else None
    
    print(f"\n--- Running Evaluation on '{split}' ({len(dataset)} frames) ---")
    t_start = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            img = batch["image"].to(device)
            gt_mask = batch["mask"][0].numpy()
            gt_coords = batch["coords"][0].numpy()
            gt_vis = batch["visibilities"][0].numpy()
            stem = batch["stem"][0]
            is_p40 = ("Patient_40" in stem)
            
            t0 = time.time()
            outputs = model(pixel_values=img)
            lat_ms = (time.time() - t0) * 1000.0
            latencies.append(lat_ms)
            
            pred_map = rasterize_class_map(outputs["masks_queries_logits"][0], outputs["class_queries_logits"][0])
            pred_coords = outputs["pred_coords"][0].cpu().numpy()
            pred_vis = outputs["pred_vis"][0].cpu().numpy()
            
            frame_metrics = evaluate_frame_metrics(
                pred_map=pred_map,
                target_map=gt_mask,
                pred_coords=pred_coords,
                gt_coords=gt_coords,
                gt_vis=gt_vis,
                canvas_size=1024
            )
            
            frame_metrics["stem"] = stem
            frame_metrics["is_patient_40"] = is_p40
            frame_metrics["latency_ms"] = lat_ms
            records.append(frame_metrics)
            
            # Render Patient 40 diagnostic montages
            if render_p40 and is_p40 and p40_dir:
                p40_out = os.path.join(p40_dir, f"{stem}_diagnostic.jpg")
                orig_rgb_norm = img[0].cpu().numpy()
                render_patient_40_diagnostic(orig_rgb_norm, gt_mask, pred_map, pred_coords, gt_coords, gt_vis, pred_vis, p40_out, frame_metrics)
                
            if (i + 1) % 25 == 0 or (i + 1) == len(loader):
                print(f"  [{split} {i+1}/{len(loader)}] Macro Dice: {np.mean([r['macro_dice'] for r in records])*100:.2f}% | Latency: {np.mean(latencies):.1f}ms")

    df = pd.DataFrame(records)
    summary = {
        "split": split,
        "total_frames": len(df),
        "macro_dice": float(df["macro_dice"].mean()),
        "macro_iou": float(df["macro_iou"].mean()),
        "macro_assd": float(df["macro_assd"].mean()),
        "ridge_dice": float(df["ridge_dice"].mean()),
        "sil_dice": float(df["sil_dice"].mean()),
        "falc_dice": float(df["falc_dice"].mean()),
        "fg_dice": float(df["fg_dice"].mean()),
        "mean_latency_ms": float(np.mean(latencies)),
        "fps": float(1000.0 / max(np.mean(latencies), 1e-4)),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != "cpu" else "CPU"
    }
    
    p40_df = df[df["is_patient_40"]]
    if len(p40_df) > 0:
        summary["patient_40_dice"] = float(p40_df["macro_dice"].mean())
        summary["patient_40_assd"] = float(p40_df["macro_assd"].mean())
        summary["patient_40_count"] = int(len(p40_df))
    else:
        summary["patient_40_dice"] = 0.0
        summary["patient_40_assd"] = 80.0
        summary["patient_40_count"] = 0
        
    if "mean_atlas_err_px" in df.columns:
        valid_errs = df["mean_atlas_err_px"].dropna()
        summary["mean_atlas_err_px"] = float(valid_errs.mean()) if len(valid_errs) > 0 else None
        
    return summary, df


def main():
    parser = argparse.ArgumentParser(description="Evaluate EXPERIMENT_7 (Manifold-Steered Mask2Former)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--data_dir", type=str, default=None, help="L3D dataset root")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_splits", type=str, default="both", choices=["val", "both"])
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.checkpoint)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"🖥️ Loading model from checkpoint: {args.checkpoint}")
    model = load_manifold_steered_model(args.checkpoint, device=args.device)

    # 1. Validation Set
    val_summary, val_df = run_evaluation(model, split="Val", data_dir=args.data_dir, out_dir=args.out_dir, device=args.device)
    val_df.to_csv(os.path.join(args.out_dir, "val_predictions.csv"), index=False)

    summary_combined = {"val_summary": val_summary}

    # 2. Test Set
    if args.eval_splits == "both":
        test_summary, test_df = run_evaluation(model, split="Test", data_dir=args.data_dir, out_dir=args.out_dir, device=args.device, render_p40=False)
        test_df.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)
        summary_combined["test_summary"] = test_summary

    # Save metrics summary
    summary_path = os.path.join(args.out_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_combined, f, indent=4)
    print(f"📄 Saved metrics summary to: {summary_path}")

    # Package results.zip
    zip_path = os.path.join(args.out_dir, "results.zip")
    create_results_zip(args.out_dir, zip_path)

if __name__ == "__main__":
    main()
