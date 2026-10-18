#!/usr/bin/env python3
"""
Phase 2 Verification Script: Kaggle P2I-LReg Dataset Adapter.
Validates:
1. Patient fold partitioning (12 train, 4 val, 5 test).
2. Data loading for Synthetic (Stage 1) and Real (Stage 2).
3. Coordinate unit parity (meters -> millimeters verified).
4. End-to-end forward pass through Vis2Reg using real dataset tensors.
"""

import sys
import os
import argparse

# Ensure experiment root is in sys.path
EXPERIMENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXPERIMENT_ROOT not in sys.path:
    sys.path.insert(0, EXPERIMENT_ROOT)

import torch
from configs.default_config import get_default_config
from vis2reg.models.vis2reg import Vis2Reg
from data.dataset import P2ILRegFolds, P2ILRegSyntheticDataset, P2ILRegRealDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Test P2I-LReg Dataset Loader on Kaggle")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/kaggle/input/datasets/junezhou001/p2i-lreg/Liver_regis",
        help="Path to Liver_regis directory on Kaggle",
    )
    return parser.parse_args()


def test_data():
    args = parse_args()
    print("=" * 65)
    print("  VIS2REG PHASE 2: DATASET ADAPTER VERIFICATION")
    print("=" * 65)

    # 1. Test 5-Fold Partitioning
    print("\n[Step 1] Verifying 5-Fold Cross-Validation Partition...")
    fold_0 = P2ILRegFolds.get_fold(0)
    print(f"         Fold 0 Train ({len(fold_0['train'])} pts): {fold_0['train']}")
    print(f"         Fold 0 Val   ({len(fold_0['val'])} pts) : {fold_0['val']}")
    print(f"         Fold 0 Test  ({len(fold_0['test'])} pts): {fold_0['test']}")
    assert len(fold_0['train']) == 12, "Fold train count must be 12!"
    assert len(fold_0['val']) == 4, "Fold val count must be 4!"
    assert len(fold_0['test']) == 5, "Fold test count must be 5!"
    assert len(set(fold_0['train']) & set(fold_0['test'])) == 0, "Patient leakage detected!"

    # 2. Check Dataset Path Existence
    data_root = args.data_root
    if not os.path.exists(data_root):
        # Fallback to local search if testing off-kaggle
        alt_paths = [
            "data/Liver_regis",
            "../data/Liver_regis",
            os.path.join(os.getcwd(), "data/Liver_regis"),
        ]
        for p in alt_paths:
            if os.path.exists(p):
                data_root = p
                break

    if not os.path.exists(data_root):
        print(f"\n[Warning] Dataset root not found at: {args.data_root}")
        print("          Fold partitioning test passed. Run this script on Kaggle to verify file reads.")
        print("=" * 65)
        return

    print(f"\n[Step 2] Found dataset root at: {data_root}")

    # 3. Test Synthetic Dataset (Stage 1)
    print("\n[Step 3] Loading Synthetic Dataset sample (Patient 01)...")
    syn_dataset = P2ILRegSyntheticDataset(
        dataset_root=data_root,
        patient_ids=['01'],
        max_points=6000,
        samples_per_patient=5,
    )
    print(f"         Indexed {len(syn_dataset)} synthetic samples for Patient 01.")
    if len(syn_dataset) > 0:
        sample_syn = syn_dataset[0]
        p_pre = sample_syn['pre_points']
        p_tgt = sample_syn['target_points']
        R_gt = sample_syn['gt_R']
        t_gt = sample_syn['gt_t']

        print(f"         Pre-op CT points shape   : {tuple(p_pre.shape)}")
        print(f"         Target syn points shape  : {tuple(p_tgt.shape)}")
        print(f"         Ground-truth R shape     : {tuple(R_gt.shape)}")
        print(f"         Ground-truth t shape     : {tuple(t_gt.shape)}")

        # Verify millimeter dimension
        pre_diag = torch.linalg.norm(p_pre.max(dim=1).values - p_pre.min(dim=1).values).item()
        tgt_diag = torch.linalg.norm(p_tgt.max(dim=1).values - p_tgt.min(dim=1).values).item()
        print(f"         Pre-op bounding diagonal : {pre_diag:.2f} mm")
        print(f"         Target bounding diagonal : {tgt_diag:.2f} mm")
        assert pre_diag > 100.0, "Pre-op CT model was not scaled to mm (diagonal < 100 mm)!"

    # 4. Test Real Dataset (Stage 2)
    print("\n[Step 4] Loading Real Dataset sample (Patient 01)...")
    real_dataset = P2ILRegRealDataset(
        dataset_root=data_root,
        patient_ids=['01'],
        max_points=6000,
    )
    print(f"         Indexed {len(real_dataset)} real keyframes for Patient 01.")
    if len(real_dataset) > 0:
        sample_real = real_dataset[0]
        r_pre = sample_real['pre_points']
        r_tgt = sample_real['target_points']
        r_scale = sample_real['pre_scale'].item()
        r_mask = sample_real['intra_masks']

        print(f"         Real pre-op points shape : {tuple(r_pre.shape)}")
        print(f"         Real target points shape : {tuple(r_tgt.shape)}")
        print(f"         Real scale factor        : {r_scale:.2f} mm")
        print(f"         Real 2D mask shape       : {tuple(r_mask.shape)}")

        # Verify positive depth
        min_z = r_tgt[0, :, 2].min().item()
        max_z = r_tgt[0, :, 2].max().item()
        print(f"         Real target Z range      : [{min_z:.2f}, {max_z:.2f}] mm")
        assert min_z >= 0.0, "Negative Z points found! Background was not properly filtered."

    # 5. Model Forward Pass on Real Sample
    print("\n[Step 5] Executing Vis2Reg model forward pass with real sample...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = get_default_config()
    model = Vis2Reg(cfg).to(device)
    model.eval()

    with torch.no_grad():
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample_real.items()}
        outputs = model(batch)

    warped = outputs['warped_pcd']
    delta = outputs['deformation']
    print(f"         Warped registered output : {tuple(warped.shape)}")
    print(f"         Displacement field delta : {tuple(delta.shape)}")
    print("         Model successfully accepted real data tensors without errors!")

    print("\n" + "=" * 65)
    print("  RESULT: PHASE 2 DATASET ADAPTER VERIFIED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    test_data()
