import os
import sys
import argparse
import cv2
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
    parser = argparse.ArgumentParser(description="Visualize EXP_05 3D Vector Space Transformer Predictions")
    parser.add_argument("--dataset_dir", type=str, default=config.dataset_dir, help="Path to dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/EXP_05/best_surgical_3d_vector.pth", help="Checkpoint path")
    parser.add_argument("--output_dir", type=str, default="results/EXP_05_visualizations", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of sample visualizations")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"🎨 Generating 3D Vector Landmark Visualizations in '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    model = Surgical3DVectorTransformer(config).to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"✅ Loaded checkpoint from '{args.checkpoint}'")

    model.eval()

    dataset = Surgical3DVectorDataset(dataset_dir=args.dataset_dir, mode="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    colors = [
        (255, 0, 0),   # Red: Ridge
        (0, 255, 0),   # Green: Silhouette
        (0, 0, 255),   # Blue: Ligament / Vessel
        (255, 255, 0), # Yellow
    ]

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_samples:
                break

            images = batch["image"].to(device)
            depth = batch["depth"].to(device)

            outputs = model(images, depth)
            pred_cls = outputs["pred_cls"].softmax(dim=-1).cpu().numpy()[0] # (N, num_classes+1)
            pred_poly = outputs["pred_polylines"].cpu().numpy()[0]          # (N, K, 3)
            pred_masks = outputs["pred_masks"].sigmoid().cpu().numpy()[0]     # (N, 1024, 1024)

            # Convert image tensor back to BGR uint8
            img_np = batch["image"].numpy()[0]
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255.0
            img_bgr = cv2.cvtColor(np.clip(img_np, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            overlay = img_bgr.copy()

            for n in range(config.num_instances):
                cid = np.argmax(pred_cls[n])
                score = pred_cls[n, cid]
                if cid == 0 or score < 0.3:
                    continue

                color = colors[(cid - 1) % len(colors)]

                # 1. Overlay 2D Mask
                mask_bin = pred_masks[n] > 0.5
                overlay[mask_bin] = overlay[mask_bin] * 0.6 + np.array(color) * 0.4

                # 2. Draw 3D Polyline (Projected back to 2D image coordinates)
                pts_3d = pred_poly[n] # (K, 3) in [-1, 1]^3
                u_pix = (pts_3d[:, 0] * 512.0 + 512.0).astype(np.int32)
                v_pix = (pts_3d[:, 1] * 512.0 + 512.0).astype(np.int32)

                for k in range(1, len(u_pix)):
                    pt1 = (int(u_pix[k-1]), int(v_pix[k-1]))
                    pt2 = (int(u_pix[k]), int(v_pix[k]))
                    cv2.line(overlay, pt1, pt2, color=color, thickness=3)
                    cv2.circle(overlay, pt2, radius=4, color=(255, 255, 255), thickness=-1)

            out_path = os.path.join(args.output_dir, f"sample_{i+1:02d}_3d_vector.jpg")
            cv2.imwrite(out_path, overlay)
            print(f"📸 Saved visual overlay to '{out_path}'")

    print(f"✅ Visualization generation complete! Check outputs in '{args.output_dir}'")

if __name__ == "__main__":
    main()
