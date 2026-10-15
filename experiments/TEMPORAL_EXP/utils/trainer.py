"""
Universal Base Fold Trainer for Laparoscopic Landmark Video Segmentation.
Implements the Strategy/Dependency Injection pattern:
Accepts any model_class and optional tracker_class, handles all 5-fold data resolution,
training loops, evaluation parity, checkpointing, and JSON logging.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Type, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(EXPERIMENT_ROOT))

from shared.utils.split_loader import get_fold_patients, get_frozen_test_patients
from utils.dataset import SingleFrameL3DDataset, PatientVideoL3DDataset, resolve_split_dir
from utils.metrics import evaluate_frame_metrics


def get_device(requested_device: Optional[str] = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    tracker: Optional[Any],
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluates video clips across a DataLoader.
    If tracker is provided (e.g. MinVIS), tracks queries in post-processing.
    If tracker is None (e.g. Cutie / SAM 2), the model internally outputs tracked predictions.
    """
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch['pixel_values'].to(device)  # (B, T, 3, H, W)
            target_masks = batch['target_masks'].to(device)  # (B, 3, H, W)
            bs, t_len, c, h, w = pixel_values.shape

            for b in range(bs):
                pv = pixel_values[b]  # (T, 3, H, W)
                tm = target_masks[b].cpu().numpy()  # (3, H, W)

                if tracker is not None:
                    # MinVIS query-tracking paradigm
                    outputs = model(pv)
                    class_logits = outputs['class_queries_logits']
                    query_embds = outputs['query_embeddings']
                    mask_logits = outputs['masks_queries_logits']

                    mask_probs = torch.nn.functional.interpolate(
                        mask_logits.sigmoid(),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False
                    )
                    tracked_sem_seg = tracker.track_sequence(query_embds, class_logits, mask_probs)
                    pred_target = tracked_sem_seg[-1].cpu().numpy()
                else:
                    # Unified VOS / Streaming memory model paradigm (Cutie, SAM 2)
                    outputs = model(pv.unsqueeze(0))
                    # Output sem_seg has shape (1, T, 3, H, W) or (1, 3, H, W)
                    sem_seg = outputs['sem_seg']
                    if sem_seg.ndim == 5:
                        pred_target = sem_seg[0, -1].cpu().numpy()
                    else:
                        pred_target = sem_seg[0].cpu().numpy()

                frame_metric = evaluate_frame_metrics(
                    pred_sem_seg=pred_target,
                    gt_input=tm,
                    threshold=threshold
                )
                all_metrics.append(frame_metric)

    summary = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0].keys()}
    return summary


def train_fold(
    fold_idx: int,
    model_class: Type[nn.Module],
    model_kwargs: Optional[Dict[str, Any]] = None,
    tracker_class: Optional[Type[Any]] = None,
    tracker_kwargs: Optional[Dict[str, Any]] = None,
    input_mode: str = "single_frame",  # 'single_frame' or 'temporal_clip'
    epochs: int = 40,
    batch_size: int = 4,
    lr_decoder: float = 1.0e-4,
    lr_backbone: float = 1.0e-5,
    weight_decay: float = 0.05,
    data_root: str = "data/L3D",
    save_dir: str = "checkpoints",
    results_dir: str = "results",
    device_name: Optional[str] = None,
    experiment_id: str = "EXPERIMENT_17_minvis",
    checkpoint_prefix: str = "best_minvis"
):
    """
    Universal Fold Trainer.
    Instantiates the model, sets up data loaders according to input_mode,
    executes training with AMP, validates, evaluates the frozen test set, and saves JSON logs.
    """
    device = get_device(device_name)
    print(f"🚀 [Trainer] Starting {experiment_id} on Fold {fold_idx} using device: {device}")

    # 1. Resolve dataset root across local and Kaggle
    try:
        train_dir = resolve_split_dir('train', data_root)
        resolved_root = str(Path(train_dir).parent)
    except FileNotFoundError:
        resolved_root = data_root
    print(f"📁 Dataset root: '{resolved_root}'")

    # 2. Get canonical patient partitions
    train_patients, val_patients = get_fold_patients(fold_idx)
    test_patients = get_frozen_test_patients()
    print(f"👥 Fold {fold_idx} -> Train: {len(train_patients)} pts | Val: {len(val_patients)} pts | Frozen Test: {len(test_patients)} pts")

    # 3. Create Datasets based on input_mode
    if input_mode == "single_frame":
        train_dataset = SingleFrameL3DDataset(
            patient_ids=train_patients,
            data_root=resolved_root,
            image_size=1024,
            stroke_width=35
        )
    elif input_mode == "temporal_clip":
        train_dataset = PatientVideoL3DDataset(
            patient_ids=train_patients,
            data_root=resolved_root,
            image_size=1024,
            stroke_width=35,
            clip_len=3
        )
    else:
        raise ValueError(f"Unknown input_mode '{input_mode}'. Must be 'single_frame' or 'temporal_clip'.")

    val_dataset = PatientVideoL3DDataset(
        patient_ids=val_patients,
        data_root=resolved_root,
        image_size=1024,
        stroke_width=35,
        clip_len=3
    )
    test_dataset = PatientVideoL3DDataset(
        patient_ids=test_patients,
        data_root=resolved_root,
        image_size=1024,
        stroke_width=35,
        clip_len=3
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=(device.type == 'cuda'))

    print(f"📦 Train Batches: {len(train_loader)} | Val Clips: {len(val_loader)} | Frozen Test Clips: {len(test_loader)}")

    # 4. Instantiate Model & Tracker
    m_kwargs = model_kwargs or {}
    model = model_class(**m_kwargs).to(device)

    tracker = None
    if tracker_class is not None:
        t_kwargs = tracker_kwargs or {}
        tracker = tracker_class(**t_kwargs)

    # 5. Optimizer & LR Scheduler
    backbone_params, decoder_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "pixel_level_module" in name or "backbone" in name or "hiera" in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = AdamW([
        {'params': decoder_params, 'lr': lr_decoder},
        {'params': backbone_params, 'lr': lr_backbone}
    ], weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1.0e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    best_val_dice = -1.0
    best_checkpoint_path = os.path.join(save_dir, f"{checkpoint_prefix}_fold{fold_idx}.pth")
    history = []

    # 6. Training Loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(device)
            target_masks = batch['target_masks'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(pixel_values, target_masks=target_masks)
                loss = outputs['loss']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        train_loss = epoch_loss / max(1, len(train_loader))

        val_metrics = evaluate_loader(model, val_loader, tracker, device)
        val_macro_dice = val_metrics['macro_dice']

        if val_macro_dice > best_val_dice:
            best_val_dice = val_macro_dice
            torch.save({
                'epoch': epoch,
                'fold_idx': fold_idx,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'train_patients': train_patients,
                'val_patients': val_patients
            }, best_checkpoint_path)

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Loss: {train_loss:.4f} | "
              f"Val Macro Dice: {val_macro_dice:.2f}% (Ridge: {val_metrics['ridge_dice']:.1f}%, Silh: {val_metrics['silhouette_dice']:.1f}%, Lig: {val_metrics['ligament_dice']:.1f}%) | "
              f"ASSD: {val_metrics['macro_assd']:.2f}px | Best: {best_val_dice:.2f}% | Elapsed: {elapsed/60:.1f}m")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_macro_dice': val_macro_dice,
            'val_metrics': val_metrics
        })

    # 7. Final Evaluation on Frozen Test Set
    print(f"\n🏁 Training complete for Fold {fold_idx}! Loading best weights ({best_val_dice:.2f}% Dice)...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    best_val_metrics = evaluate_loader(model, val_loader, tracker, device)
    frozen_test_metrics = evaluate_loader(model, test_loader, tracker, device)

    print(f"📊 Final Best Val Metrics: Macro Dice: {best_val_metrics['macro_dice']:.2f}% | Micro: {best_val_metrics['micro_dice']:.2f}% | ASSD: {best_val_metrics['macro_assd']:.2f}px")
    print(f"🔒 Frozen Test Metrics: Macro Dice: {frozen_test_metrics['macro_dice']:.2f}% | Micro: {frozen_test_metrics['micro_dice']:.2f}% | ASSD: {frozen_test_metrics['macro_assd']:.2f}px")

    # 8. Standardized JSON Output for Aggregator
    val_out_file = os.path.join(results_dir, f"results_fold_{fold_idx}.json")
    with open(val_out_file, 'w') as f:
        json.dump({
            'experiment': experiment_id,
            'fold_idx': fold_idx,
            'best_epoch': checkpoint['epoch'],
            'train_patients': train_patients,
            'val_patients': val_patients,
            'best_val_metrics': best_val_metrics,
            'history': history
        }, f, indent=2)

    test_out_file = os.path.join(results_dir, f"test_results_fold_{fold_idx}.json")
    with open(test_out_file, 'w') as f:
        json.dump({
            'experiment': experiment_id,
            'fold_idx': fold_idx,
            'frozen_test_patients': test_patients,
            'test_metrics': frozen_test_metrics
        }, f, indent=2)

    print(f"💾 Results saved to '{val_out_file}' and '{test_out_file}'")
    return best_val_metrics, frozen_test_metrics
