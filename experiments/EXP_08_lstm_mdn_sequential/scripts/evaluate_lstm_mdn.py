import os
import sys
import argparse
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial.distance import cdist

# Add experiment root to path
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXP_DIR not in sys.path:
    sys.path.append(EXP_DIR)

from configs.exp08_config import config
from utils.dataset_sequential import SequentialLandmarkDataset
from models.surgical_lstm_mdn import SurgicalLSTM_MDN


def compute_chamfer_distance_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Bidirectional Chamfer Distance between two 2D polylines (K, 2).
    Lower = better structural alignment.
    """
    dists = cdist(p1, p2, metric='euclidean')  # (K1, K2)
    d1 = np.mean(np.min(dists, axis=1))
    d2 = np.mean(np.min(dists, axis=0))
    return 0.5 * (d1 + d2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate EXP_08 CNN-LSTM-MDN Sequential Landmark Detection")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--dataset_dir", type=str, default=config.dataset_dir, help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate (val/test)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("📊 Evaluating EXP_08 CNN-LSTM-MDN Sequential Landmark Detection")
    print("=" * 70)
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"📁 Dataset: {args.dataset_dir}")
    print(f"📋 Split: {args.split}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")
    
    # 1. Dataset
    eval_dataset = SequentialLandmarkDataset(
        args.dataset_dir,
        num_instances=config.num_instances,
        num_points=config.num_points,
        mode=args.split,
        image_size=config.image_size,
        stroke_thickness=config.mask_stroke_thickness
    )
    
    if len(eval_dataset) == 0 and args.split == "val":
        print("⚠️ No val split found, trying 'test'...")
        eval_dataset = SequentialLandmarkDataset(
            args.dataset_dir,
            num_instances=config.num_instances,
            num_points=config.num_points,
            mode="test",
            image_size=config.image_size,
            stroke_thickness=config.mask_stroke_thickness
        )
    
    print(f"📊 Evaluation Dataset Size: {len(eval_dataset)} images")
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 2. Model
    model = SurgicalLSTM_MDN(config).to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        epoch = ckpt.get("epoch", "N/A")
        print(f"✅ Loaded checkpoint (Epoch {epoch})")
    else:
        print(f"❌ Checkpoint '{args.checkpoint}' not found!")
        return
    
    model.eval()
    
    # 3. Evaluation
    class_names = ["Ridge", "Silhouette", "Falciform", "Gallbladder"]
    S = config.image_size
    K = config.num_points
    stroke_t = config.mask_stroke_thickness
    
    # Per-class metrics storage
    per_class_dices = {c: [] for c in range(1, config.num_classes + 1)}
    per_class_chamfer = {c: [] for c in range(1, config.num_classes + 1)}
    per_class_poly_err = {c: [] for c in range(1, config.num_classes + 1)}
    
    all_hard_dices = []
    all_chamfer = []
    all_poly_err = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device)
            target_polylines = batch["target_polylines"]  # (B, N, K, 2)
            target_masks = batch["target_masks"]          # (B, N, S, S)
            valid_mask = batch["valid_mask"]               # (B, N)
            target_classes = batch["target_classes"]       # (B, N)
            
            B = images.shape[0]
            N = target_polylines.shape[1]
            
            # Autoregressive inference
            outputs = model(images)
            pred_polylines = outputs["predicted_polylines"].cpu()  # (B, num_classes, K, 2)
            
            for b in range(B):
                for i in range(N):
                    if not valid_mask[b, i]:
                        continue
                    
                    cls_id = target_classes[b, i].item()
                    if cls_id < 1 or cls_id > config.num_classes:
                        continue
                    
                    gt_poly = target_polylines[b, i].numpy()      # (K, 2) in [0, 1]
                    gt_mask = target_masks[b, i].numpy()          # (S, S)
                    pred_poly = pred_polylines[b, cls_id - 1].numpy()  # (K, 2) in [0, 1]
                    
                    # Handle bidirectional matching
                    err_fwd = np.mean(np.abs(pred_poly - gt_poly))
                    err_rev = np.mean(np.abs(pred_poly - gt_poly[::-1]))
                    if err_rev < err_fwd:
                        gt_poly_aligned = gt_poly[::-1].copy()
                    else:
                        gt_poly_aligned = gt_poly
                    
                    # 1. Polyline Error (pixels)
                    poly_err = np.mean(np.abs(pred_poly - gt_poly_aligned)) * S
                    per_class_poly_err[cls_id].append(poly_err)
                    all_poly_err.append(poly_err)
                    
                    # 2. Chamfer Distance (normalized)
                    chamfer = compute_chamfer_distance_2d(pred_poly, gt_poly)
                    per_class_chamfer[cls_id].append(chamfer)
                    all_chamfer.append(chamfer)
                    
                    # 3. Hard Dice
                    pred_mask = np.zeros((S, S), dtype=np.uint8)
                    pts_pix = (pred_poly * float(S)).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(pred_mask, [pts_pix], isClosed=False, color=1, thickness=stroke_t)
                    pred_mask_f = pred_mask.astype(np.float32)
                    gt_mask_bin = (gt_mask > 0.5).astype(np.float32)
                    
                    eps = 1e-5
                    inter = (pred_mask_f * gt_mask_bin).sum()
                    dice = (2.0 * inter + eps) / (pred_mask_f.sum() + gt_mask_bin.sum() + eps)
                    
                    per_class_dices[cls_id].append(dice)
                    all_hard_dices.append(dice)
    
    # 4. Print Results
    print(f"\n{'=' * 70}")
    print(f"📊 EXP_08 Evaluation Results")
    print(f"{'=' * 70}")
    
    print(f"\n{'Class':<20} {'Dice':>10} {'Chamfer':>10} {'PolyErr(px)':>12} {'Count':>8}")
    print("-" * 62)
    
    for cls_id in range(1, config.num_classes + 1):
        name = class_names[cls_id - 1] if cls_id - 1 < len(class_names) else f"Class {cls_id}"
        d = per_class_dices[cls_id]
        c = per_class_chamfer[cls_id]
        p = per_class_poly_err[cls_id]
        
        dice_val = np.mean(d) if d else 0.0
        chamfer_val = np.mean(c) if c else 0.0
        poly_val = np.mean(p) if p else 0.0
        count = len(d)
        
        print(f"{name:<20} {dice_val:>10.4f} {chamfer_val:>10.4f} {poly_val:>12.1f} {count:>8}")
    
    print("-" * 62)
    overall_dice = np.mean(all_hard_dices) if all_hard_dices else 0.0
    overall_chamfer = np.mean(all_chamfer) if all_chamfer else 0.0
    overall_poly = np.mean(all_poly_err) if all_poly_err else 0.0
    total_count = len(all_hard_dices)
    
    print(f"{'OVERALL':<20} {overall_dice:>10.4f} {overall_chamfer:>10.4f} {overall_poly:>12.1f} {total_count:>8}")
    print(f"\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
