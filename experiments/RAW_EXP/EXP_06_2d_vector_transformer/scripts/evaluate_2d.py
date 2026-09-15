import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial.distance import cdist

# Add experiment root to path
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXP_DIR not in sys.path:
    sys.path.append(EXP_DIR)

from configs.exp06_config import config
from utils.dataset_2d import Surgical2DVectorDataset
from models.surgical_2d_vector_transformer import Surgical2DVectorTransformer
from scripts.train_2d_vector_transformer import compute_mask_metrics

def compute_chamfer_distance_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Computes bidirectional Chamfer Distance between two 2D polylines (K, 2).
    """
    dists = cdist(p1, p2, metric='euclidean') # (K, K)
    d1 = np.mean(np.min(dists, axis=1))
    d2 = np.mean(np.min(dists, axis=0))
    return 0.5 * (d1 + d2)

def main():
    parser = argparse.ArgumentParser(description="Evaluate EXP_06 Direct 2D Vector Transformer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--dataset_dir", type=str, default=config.dataset_dir, help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    args = parser.parse_args()

    print("=" * 70)
    print("📊 Evaluating EXP_06 Direct 2D Vector Space Transformer")
    print("=" * 70)
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"📁 Dataset: {args.dataset_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Execution Device: {device}")

    # 1. Load Dataset
    val_dataset = Surgical2DVectorDataset(args.dataset_dir, mode="val")
    if len(val_dataset) == 0:
        val_dataset = Surgical2DVectorDataset(args.dataset_dir, mode="test")

    print(f"📊 Evaluation Dataset Size: {len(val_dataset)} images")
    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 2. Load Model
    model = Surgical2DVectorTransformer(config).to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(f"✅ Loaded checkpoint successfully (Epoch {ckpt.get('epoch', 'N/A')})")
    else:
        print(f"⚠️ Checkpoint '{args.checkpoint}' not found.")
        return

    model.eval()

    total_metrics = {"hard_iou": [], "hard_dice": [], "soft_iou": [], "soft_dice": []}
    chamfer_dists_norm = []
    chamfer_dists_pixels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device)
            target_masks = batch["target_masks"].to(device)
            target_polylines = batch["target_polylines"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            outputs = model(images)
            pred_masks = outputs["pred_masks"]
            pred_polylines = outputs["pred_polylines"]

            # Compute Mask Metrics
            m_res = compute_mask_metrics(pred_masks, target_masks, valid_mask)
            for k in total_metrics:
                total_metrics[k].append(m_res[k])

            # Compute 2D Polyline Chamfer Distances
            B = images.shape[0]
            for b in range(B):
                active_gts = [g for g in range(valid_mask.shape[1]) if valid_mask[b, g] > 0]
                for g in active_gts:
                    gt_p = target_polylines[b, g].cpu().numpy()
                    
                    # Match best predicted query
                    best_cd = float('inf')
                    for q in range(pred_polylines.shape[1]):
                        pred_p = pred_polylines[b, q].cpu().numpy()
                        cd = compute_chamfer_distance_2d(pred_p, gt_p)
                        if cd < best_cd:
                            best_cd = cd

                    if best_cd < float('inf'):
                        chamfer_dists_norm.append(best_cd)
                        chamfer_dists_pixels.append(best_cd * 1024.0)

    print("\n" + "=" * 70)
    print("📈 FINAL QUANTITATIVE EVALUATION RESULTS (EXP_06)")
    print("=" * 70)
    print(f"• 2D Mask Hard Dice:        {np.mean(total_metrics['hard_dice']) * 100.0:.2f}%")
    print(f"• 2D Mask Hard IoU:         {np.mean(total_metrics['hard_iou']) * 100.0:.2f}%")
    print(f"• 2D Mask Soft Dice:        {np.mean(total_metrics['soft_dice']) * 100.0:.2f}%")
    print(f"• 2D Mask Soft IoU:         {np.mean(total_metrics['soft_iou']) * 100.0:.2f}%")
    print(f"• 2D Polyline Chamfer Dist: {np.mean(chamfer_dists_pixels):.2f} pixels ({np.mean(chamfer_dists_norm):.4f} normalized)")
    print("=" * 70)

if __name__ == "__main__":
    main()
