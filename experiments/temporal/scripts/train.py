"""
Declarative Training Script for MinVIS (Experiment 17).
Uses the Universal Fold Trainer engine by injecting the MinVISMask2Former model
and MinVISTracker class.
"""

import sys
import argparse
from pathlib import Path

# Ensure paths work seamlessly across macOS and Kaggle
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(EXPERIMENT_ROOT))

from models.minvis_mask2former import MinVISMask2Former
from models.tracker import MinVISTracker
from utils.trainer import train_fold


def parse_args():
    parser = argparse.ArgumentParser(description="Train MinVIS on Laparoscopic Liver 5-Fold Splits")
    parser.add_argument('--fold_idx', type=int, default=0, help='Fold index (0 to 4)')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    parser.add_argument('--lr-decoder', type=float, default=1.0e-4, help='Decoder learning rate')
    parser.add_argument('--lr-backbone', type=float, default=1.0e-5, help='Backbone learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--data-root', type=str, default='data/L3D', help='Root dataset directory')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Checkpoint output directory')
    parser.add_argument('--results-dir', type=str, default='results', help='Results JSON output directory')
    parser.add_argument('--device', type=str, default=None, help='Device override (cuda/mps/cpu)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Call universal trainer with MinVIS-specific classes and single-frame input mode
    train_fold(
        fold_idx=args.fold_idx,
        model_class=MinVISMask2Former,
        model_kwargs={'num_classes': 3, 'hidden_dim': 256},
        tracker_class=MinVISTracker,
        tracker_kwargs={'num_classes': 3},
        input_mode="single_frame",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_decoder=args.lr_decoder,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        data_root=args.data_root,
        save_dir=args.save_dir,
        results_dir=args.results_dir,
        device_name=args.device,
        experiment_id="EXPERIMENT_17_minvis",
        checkpoint_prefix="best_minvis"
    )


if __name__ == '__main__':
    main()
