import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import prepare_dataset
from utils.dataset import load_image, load_mask



def evaluation(pred, gt):
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    dice = (2 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
    iou = dice / (2 - dice)
    return iou, dice


def pred_bgr_to_mask(pred_bgr):
    """
    Nearest-Neighbor color matching: maps each pixel to its closest
    class color (0=Black, 1=Red, 2=Green, 3=Blue in BGR).
    Recovers edge/boundary pixels that threshold matching would miss.
    """
    palette = np.array([
        [0, 0, 0],     # 0: Black (Background)
        [0, 0, 255],   # 1: Red (Ridge) in BGR
        [0, 255, 0],   # 2: Green (Silhouette) in BGR
        [255, 0, 0]    # 3: Blue (Ligament) in BGR
    ], dtype=np.float32)

    dists = np.linalg.norm(
        pred_bgr.astype(np.float32)[:, :, None, :] - palette[None, None, :, :],
        axis=-1
    )
    return np.argmin(dists, axis=-1).astype(np.uint8)


def colorize_mask_rgb(mask_2d):
    """Convert 2D class mask to RGB image for plotting"""
    h, w = mask_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask_2d == 1] = [255, 0, 0]    # Red
    rgb[mask_2d == 2] = [0, 255, 0]    # Green
    rgb[mask_2d == 3] = [0, 128, 255]  # Blue
    return rgb


def create_error_overlay(pred_2d, gt_2d):
    """
    Error Overlay:
    - Green: Match (True Positive)
    - Red: False Positive (Extra/Noise)
    - Yellow: False Negative (Missing landmark line/gap)
    """
    h, w = pred_2d.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    pred_fg = pred_2d > 0
    gt_fg = gt_2d > 0

    tp = pred_fg & gt_fg
    fp = pred_fg & (~gt_fg)
    fn = (~pred_fg) & gt_fg

    overlay[tp] = [0, 255, 0]    # Green
    overlay[fp] = [255, 0, 0]    # Red
    overlay[fn] = [255, 255, 0]  # Yellow
    return overlay


def main(args):
    train_file, test_file, val_file = prepare_dataset.get_split(args.data_path)

    if args.split.lower() == 'val':
        eval_files = val_file
    elif args.split.lower() == 'train':
        eval_files = train_file
    else:
        eval_files = test_file

    print(f"Analyzing {len(eval_files)} saved prediction images from: '{args.pred_path}' ({args.split.upper()} split)")

    if not os.path.exists(args.pred_path):
        raise FileNotFoundError(f"Prediction path '{args.pred_path}' does not exist! Please run test.py first.")

    pred_files = os.listdir(args.pred_path)
    results_list = []

    for img_path in tqdm(eval_files, desc="Matching pre-saved predictions"):
        img_stem = Path(img_path).stem  # e.g., 'Patient_32_0018000'
        img_name = os.path.basename(img_path)

        # 1. Exact match by full patient stem (e.g. Patient_32_0018000.png)
        cand1 = os.path.join(args.pred_path, f"{img_stem}.png")
        cand2 = os.path.join(args.pred_path, f"{img_stem}.jpg")

        # 2. Match by sanitized relative path
        rel_key = str(img_path).replace('/', '_').replace('\\', '_')
        cand3 = os.path.join(args.pred_path, f"{rel_key}.png")

        matched_file = None
        for cand in [cand1, cand2, cand3]:
            if os.path.exists(cand):
                matched_file = cand
                break

        if not matched_file:
            # Strict fallback requiring exact full stem
            for pf in pred_files:
                if pf == f"{img_stem}.png" or pf == f"{img_stem}.jpg" or pf.endswith(f"_{img_stem}.png"):
                    matched_file = os.path.join(args.pred_path, pf)
                    break

        if not matched_file:
            continue

        pred_bgr = cv2.imread(matched_file)
        if pred_bgr is None:
            continue


        rgb_img = load_image(img_path)
        gt_masks = load_mask(img_path)
        gt_2d = np.argmax(gt_masks, axis=0)

        pred_2d = pred_bgr_to_mask(pred_bgr)

        pred_channels = np.array([pred_2d == i for i in range(4)]).astype(np.uint8)
        gt_channels = np.array([gt_2d == i for i in range(4)]).astype(np.uint8)

        iou, dice = evaluation(pred_channels[1:].flatten(), gt_channels[1:].flatten())

        results_list.append({
            'name': img_name,
            'dice': dice,
            'iou': iou,
            'rgb': rgb_img,
            'pred_2d': pred_2d,
            'gt_2d': gt_2d
        })


    # Sort results by Dice ascending (worst performing first)
    results_list.sort(key=lambda x: x['dice'])

    dices = [r['dice'] for r in results_list]
    ious = [r['iou'] for r in results_list]

    print("\n=================== ANALYSIS SUMMARY ===================")
    print(f"Matched Predictions: {len(results_list)}")
    print(f"Mean Dice:   {np.mean(dices):.4f} | Median: {np.median(dices):.4f} | Min: {np.min(dices):.4f} | Max: {np.max(dices):.4f}")
    print(f"Mean IoU:    {np.mean(ious):.4f} | Median: {np.median(ious):.4f}")
    print("========================================================\n")

    num_to_show = min(args.num_samples, len(results_list))
    print(f"Displaying top {num_to_show} worst-performing sample visualizations:\n")

    for i in range(num_to_show):
        res = results_list[i]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Panel 1: RGB Image
        axes[0].imshow(res['rgb'])
        axes[0].set_title(f"Rank {i+1} Worst - Input RGB\n({res['name']})", fontsize=10)
        axes[0].axis('off')

        # Panel 2: Ground Truth
        axes[1].imshow(colorize_mask_rgb(res['gt_2d']))
        axes[1].set_title("Ground Truth\n(Red:Ridge, Green:Silh, Blue:Lig)", fontsize=10)
        axes[1].axis('off')

        # Panel 3: Saved Prediction
        axes[2].imshow(colorize_mask_rgb(res['pred_2d']))
        axes[2].set_title(f"Saved Prediction\nDice: {res['dice']:.3f} | IoU: {res['iou']:.3f}", fontsize=10)
        axes[2].axis('off')

        # Panel 4: Error Overlay Map
        axes[3].imshow(create_error_overlay(res['pred_2d'], res['gt_2d']))
        axes[3].set_title(f"Error Map\n(Green:Match, Red:FP, Yellow:FN)", fontsize=10)
        axes[3].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standalone Result Analysis & Plotting Tool")
    parser.add_argument('--pred_path', default='val_results/', help='Path to pre-saved prediction images from test.py')
    parser.add_argument('--data_path', default='data/', help='Path to original L3D dataset')
    parser.add_argument('--split', default='val', choices=['val', 'test', 'train'])
    parser.add_argument('--num_samples', type=int, default=10, help='Number of worst failing samples to display')
    args = parser.parse_args()

    main(args=args)
