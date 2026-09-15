import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXP_DIR not in sys.path:
    sys.path.append(EXP_DIR)

from configs.exp05_config import config
from utils.dataset_3d import Surgical3DVectorDataset
from models.surgical_3d_vector_transformer import Surgical3DVectorTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EXP_05 3D Vector Space Transformer")
    parser.add_argument("--dataset_dir", type=str, default=config.dataset_dir, help="Path to surgical dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/EXP_05/best_surgical_3d_vector.pth", help="Checkpoint path")
    return parser.parse_args()

def compute_3d_chamfer_distance(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """
    Computes bidirectional 3D Chamfer Distance between predicted and GT 3D polylines.
    """
    # pred_pts: (K, 3), gt_pts: (K, 3)
    diff_fwd = pred_pts[:, None, :] - gt_pts[None, :, :]
    dist_fwd = np.sqrt(np.sum(diff_fwd**2, axis=-1))
    
    min_pred_to_gt = np.mean(np.min(dist_fwd, axis=1))
    min_gt_to_pred = np.mean(np.min(dist_fwd, axis=0))
    return float((min_pred_to_gt + min_gt_to_pred) / 2.0)

def main():
    args = parse_args()
    print("=" * 70)
    print("🧪 Evaluation: EXP_05 Monocular 3D Vector Space Transformer")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    if not os.path.exists(args.checkpoint):
        print(f"⚠️ Checkpoint file '{args.checkpoint}' not found. Initializing model for dry-run evaluation.")
        model = Surgical3DVectorTransformer(config).to(device)
    else:
        print(f"📥 Loading checkpoint from '{args.checkpoint}'...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model = Surgical3DVectorTransformer(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    dataset = Surgical3DVectorDataset(
        dataset_dir=args.dataset_dir,
        num_instances=config.num_instances,
        num_points=config.num_points,
        mode="test"
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    chamfer_errors = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            depth = batch["depth"].to(device)
            target_poly = batch["target_polylines"].numpy()[0]
            valid_m = batch["valid_mask"].numpy()[0]

            outputs = model(images, depth)
            pred_poly = outputs["pred_polylines"].cpu().numpy()[0] # (N, K, 3)

            valid_gt = target_poly[valid_m]
            if len(valid_gt) == 0:
                continue

            for gt in valid_gt:
                # Find closest predicted polyline
                c_dists = [compute_3d_chamfer_distance(p, gt) for p in pred_poly]
                chamfer_errors.append(min(c_dists))

    mean_chamfer = float(np.mean(chamfer_errors)) if chamfer_errors else 0.0
    print("-" * 70)
    print(f"📊 Quantitative 3D Benchmark Evaluation Results ({len(dataset)} samples):")
    print(f"   • 3D Chamfer Distance Error: {mean_chamfer:.4f} (Normalized Camera Space)")
    print("=" * 70)

if __name__ == "__main__":
    main()
