import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.temporal_mask2former import TemporalMask2Former
from utils.dataset_temporal import StratifiedTemporalL3DDataset, PatientTemporalL3DDataset, resolve_l3d_root, resolve_split_dir
from utils.metrics import evaluate_frame_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Stratified Temporal Mask2Former on Laparoscopic Liver Landmark Benchmark")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pth')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'], help='Dataset split to evaluate')
    parser.add_argument('--output-dir', type=str, default='eval_results', help='Directory to save evaluation results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for foreground')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 Using Device: {device}")

    # Load Checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint.get('config', {})
    num_classes = cfg.get('model', {}).get('num_classes', 3)

    # Initialize Model
    model = TemporalMask2Former(
        base_checkpoint=cfg.get('model', {}).get('base_checkpoint', "facebook/mask2former-swin-tiny-ade-semantic"),
        num_classes=num_classes,
        hidden_dim=cfg.get('model', {}).get('hidden_dim', 256),
        temporal_gate_init=cfg.get('model', {}).get('temporal_gate_init', 0.1)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Checkpoint loaded from: {args.checkpoint} (Trained Epoch {checkpoint.get('epoch', '?')})")

    # Dataset (Stratified Option 1: Patient 40 <-> Patient 38 Swap)
    test_ds = StratifiedTemporalL3DDataset(
        split=args.split,
        clip_len=cfg.get('dataset', {}).get('clip_len', 3),
        max_gap=cfg.get('dataset', {}).get('max_gap', 240),
        image_size=cfg.get('dataset', {}).get('image_size', 1024),
        stroke_width=cfg.get('dataset', {}).get('stroke_width', 35)
    )
    print(f"📦 Evaluating {len(test_ds)} clips on '{args.split}' split (100% causal coverage)...")

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    records = []
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch['pixel_values'].to(device)  # (1, T, 3, H, W)
            target_masks = batch['target_masks'].to(device)  # (1, 3, H, W)
            pat = batch['patient'][0]
            fids = batch['frame_ids']
            target_fid = fids[-1].item() if torch.is_tensor(fids[-1]) else fids[-1]

            # In inference: past context frames use model's own predictions (autonomous inference)
            outputs = model(pixel_values)

            class_logits = outputs.class_queries_logits
            mask_logits = outputs.masks_queries_logits

            mask_probs = mask_logits.sigmoid()
            class_probs = class_logits.softmax(dim=-1)[..., :num_classes]
            sem_seg = torch.einsum('bqc, bqhw -> bchw', class_probs, mask_probs)

            sem_seg_full = nn.functional.interpolate(
                sem_seg,
                size=(target_masks.shape[-2], target_masks.shape[-1]),
                mode='bilinear',
                align_corners=False
            )

            pred_prob = sem_seg_full[0].cpu().numpy()
            gt_in = batch['gt_map'][0].cpu().numpy() if 'gt_map' in batch else target_masks[0].cpu().numpy()

            m = evaluate_frame_metrics(pred_prob, gt_in, num_classes=num_classes, threshold=args.threshold)
            m['patient'] = pat
            m['frame_id'] = target_fid
            records.append(m)

    df = pd.DataFrame(records)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f'{args.split}_per_frame_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"💾 Per-frame metrics exported to: {csv_path}")

    # Summary Statistics
    summary = {
        'split': args.split,
        'num_frames': len(df),
        'macro_dice': float(df['macro_dice'].mean()),
        'micro_dice': float(df['micro_dice'].mean()),
        'macro_iou': float(df['macro_iou'].mean()),
        'mean_assd': float(df['mean_assd'].dropna().mean()),
        'dice_ridge': float(df['dice_ridge'].mean()),
        'dice_silhouette': float(df['dice_silhouette'].mean()),
        'dice_ligament': float(df['dice_ligament'].mean()),
        'assd_ridge': float(df['assd_ridge'].dropna().mean()),
        'assd_silhouette': float(df['assd_silhouette'].dropna().mean()),
        'assd_ligament': float(df['assd_ligament'].dropna().mean())
    }

    # Per-patient Breakdown
    patient_summary = {}
    for pat, group in df.groupby('patient'):
        patient_summary[pat] = {
            'num_frames': len(group),
            'macro_dice': float(group['macro_dice'].mean()),
            'micro_dice': float(group['micro_dice'].mean()),
            'mean_assd': float(group['mean_assd'].dropna().mean())
        }
    summary['per_patient'] = patient_summary

    json_path = os.path.join(args.output_dir, f'{args.split}_summary_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"📊 {args.split.upper()} BENCHMARK RESULTS SUMMARY (EXPERIMENT 16)")
    print("=" * 60)
    print(f"Total Evaluated Frames:                 {summary['num_frames']}")
    print(f"Overall Micro-Dice (TopoNet standard): {summary['micro_dice']*100:.2f}%")
    print(f"Overall Macro-Dice (BCRNet standard):  {summary['macro_dice']*100:.2f}%")
    print(f"Overall Mean ASSD:                      {summary['mean_assd']:.2f} px")
    print("-" * 60)
    print(f"  • Ridge Dice:      {summary['dice_ridge']*100:.2f}% | ASSD: {summary['assd_ridge']:.2f} px")
    print(f"  • Silhouette Dice: {summary['dice_silhouette']*100:.2f}% | ASSD: {summary['assd_silhouette']:.2f} px")
    print(f"  • Ligament Dice:   {summary['dice_ligament']*100:.2f}% | ASSD: {summary['assd_ligament']:.2f} px")
    print("-" * 60)
    print("Patient-level breakdown:")
    for pat, stats in patient_summary.items():
        print(f"  [{pat:10s}] Frames: {stats['num_frames']:3d} | Macro: {stats['macro_dice']*100:.2f}% | Micro: {stats['micro_dice']*100:.2f}% | ASSD: {stats['mean_assd']:.2f} px")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
