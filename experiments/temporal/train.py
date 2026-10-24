"""
Universal Training Entrypoint for experiments/temporal.
Select any registered model via --model (minvis, cutie, sam2).
Automatically loads the canonical 5-fold splits, trains with AMP, evaluates on video clips,
and saves outputs in standardized results/ and checkpoints/ subdirectories.

Usage Examples:
  python train.py --model minvis --fold_idx 0
  python train.py --model cutie  --fold_idx 0
  python train.py --model sam2   --fold_idx 0
"""

import os
import sys
import argparse
from pathlib import Path

# Setup paths
EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(EXP_DIR))

from models.registry import get_model_entry, list_available_models
from core.trainer import train_fold


def parse_args():
    parser = argparse.ArgumentParser(description="Universal Temporal Video Model Trainer")
    parser.add_argument('--model', type=str, default='minvis', choices=['minvis', 'cutie', 'sam2'],
                        help="Model architecture to train (minvis, cutie, sam2)")
    parser.add_argument('--fold_idx', type=int, default=0, help="Fold index (0 to 4)")
    parser.add_argument('--epochs', type=int, default=40, help="Total training epochs")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size")
    parser.add_argument('--lr-decoder', type=float, default=1.0e-4, help="Decoder learning rate")
    parser.add_argument('--lr-backbone', type=float, default=1.0e-5, help="Backbone learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.05, help="Weight decay")
    parser.add_argument('--data-root', type=str, default='data/L3D', help="Root dataset directory")
    parser.add_argument('--device', type=str, default=None, help="Device override (cuda/mps/cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model.lower().strip()

    # Look up model entry from central registry
    entry = get_model_entry(model_name)
    print(f"🌟 Selected Architecture: {model_name.upper()} ({entry['description']})")

    # Destination directories isolated by model name
    save_dir = os.path.join(str(EXP_DIR), "checkpoints", model_name)
    results_dir = os.path.join(str(EXP_DIR), "results", model_name)

    # Call the universal base trainer engine
    train_fold(
        fold_idx=args.fold_idx,
        model_class=entry['model_class'],
        model_kwargs=entry['default_model_kwargs'],
        tracker_class=entry['tracker_class'],
        tracker_kwargs=entry['default_tracker_kwargs'],
        input_mode=entry['input_mode'],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_decoder=args.lr_decoder,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        data_root=args.data_root,
        save_dir=save_dir,
        results_dir=results_dir,
        device_name=args.device,
        experiment_id=f"TEMPORAL_{model_name.upper()}",
        checkpoint_prefix=entry['checkpoint_prefix']
    )


if __name__ == '__main__':
    main()
