#!/usr/bin/env python3
"""
Unit verification test for EXPERIMENT_4: Landmark Master Tokens + Patch Bézier Decoder
Tests AST syntax, Decoder forward/backward, loss masking, gradient flow, rasterization, and diagnostic rendering.
Works standalone without requiring external HuggingFace weights locally.
"""
import os
import sys
import ast
import tempfile
import numpy as np
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from experiments.EXPERIMENT_4.models.landmark_bezier_decoder import LandmarkBezierPatchDecoder
from experiments.EXPERIMENT_4.models.landmark_losses import LandmarkBezierLoss
from experiments.EXPERIMENT_4.models.bezier_utils import rasterize_bezier_predictions
from experiments.EXPERIMENT_4.utils.metrics import _render_patient40_panel, compute_frame_metrics

def test_ast():
    print("1. Checking AST syntax of all EXPERIMENT_4 python files...")
    exp4_dir = os.path.join(WORKSPACE_ROOT, 'experiments/EXPERIMENT_4')
    checked = 0
    for root, _, files in os.walk(exp4_dir):
        for f in files:
            if f.endswith('.py'):
                file_path = os.path.join(root, f)
                with open(file_path, 'r', encoding='utf-8') as pyf:
                    ast.parse(pyf.read(), filename=f)
                checked += 1
    print(f"   ✓ All {checked} Python files parsed successfully with valid AST syntax.")

def test_decoder_forward_backward():
    print("\n2. Testing LandmarkBezierPatchDecoder Forward and Backward Pass (CPU)...")
    device = torch.device('cpu')
    
    decoder = LandmarkBezierPatchDecoder(
        embed_dim=256,
        grid_size=8,
        num_classes=4,
        num_decoder_layers=2, # use 2 layers for fast local smoke test
        num_landmarks=3,
        single_scale=True
    ).to(device)
    
    B = 2
    # Mock pixel decoder outputs: [stride-32 (32x32), stride-16 (64x64), stride-8 (128x128)]
    multi_scale_features = [
        torch.randn(B, 256, 32, 32, device=device),
        torch.randn(B, 256, 64, 64, device=device),
        torch.randn(B, 256, 128, 128, device=device)
    ]
    
    pred_class, pred_bezier, pred_presence, pred_centroid = decoder(multi_scale_features)
    
    print(f"   pred_class shape   : {pred_class.shape} (expected [{B}, 64, 4])")
    print(f"   pred_bezier shape  : {pred_bezier.shape} (expected [{B}, 64, 4, 2])")
    print(f"   pred_presence shape: {pred_presence.shape} (expected [{B}, 3])")
    print(f"   pred_centroid shape: {pred_centroid.shape} (expected [{B}, 3, 2])")
    
    assert pred_class.shape == (B, 64, 4), f"Wrong pred_class shape: {pred_class.shape}"
    assert pred_bezier.shape == (B, 64, 4, 2), f"Wrong pred_bezier shape: {pred_bezier.shape}"
    assert pred_presence.shape == (B, 3), f"Wrong pred_presence shape: {pred_presence.shape}"
    assert pred_centroid.shape == (B, 3, 2), f"Wrong pred_centroid shape: {pred_centroid.shape}"
    assert (pred_bezier >= 0.0).all() and (pred_bezier <= 1.0).all(), "pred_bezier out of [0, 1] range"
    assert (pred_centroid >= 0.0).all() and (pred_centroid <= 1.0).all(), "pred_centroid out of [0, 1] range"
    
    # Test loss with missing landmark in sample 1
    criterion = LandmarkBezierLoss(continuity_phase_epoch=2)
    target_class = torch.randint(0, 4, (B, 64), device=device)
    target_bezier = torch.rand(B, 64, 4, 2, device=device)
    active_mask = target_class > 0
    
    target_lm_presence = torch.tensor([
        [1.0, 1.0, 1.0], # sample 0: all 3 present
        [1.0, 1.0, 0.0]  # sample 1: Falciform absent (missing)
    ], device=device)
    
    target_lm_centroid = torch.tensor([
        [[0.5, 0.6], [0.5, 0.3], [0.4, 0.4]],
        [[0.4, 0.7], [0.6, 0.2], [0.0, 0.0]]
    ], device=device)
    
    # Phase 1
    loss_p1, dict_p1 = criterion(pred_class, pred_bezier, pred_presence, pred_centroid,
                                 target_class, target_bezier, active_mask,
                                 target_lm_presence, target_lm_centroid, epoch=1)
    print(f"   ✓ Phase 1 loss computed: {loss_p1.item():.4f}, cont_loss: {dict_p1['cont_loss']}, lm_com: {dict_p1['lm_com']:.4f}")
    assert dict_p1['cont_loss'] == 0.0, "Phase 1 cont_loss should be 0.0"
    
    # Phase 2
    loss_p2, dict_p2 = criterion(pred_class, pred_bezier, pred_presence, pred_centroid,
                                 target_class, target_bezier, active_mask,
                                 target_lm_presence, target_lm_centroid, epoch=2)
    print(f"   ✓ Phase 2 loss computed: {loss_p2.item():.4f}, cont_loss: {dict_p2['cont_loss']:.4f}, lm_com: {dict_p2['lm_com']:.4f}")
    
    # Backward pass
    loss_p2.backward()
    lm_token_grad = decoder.landmark_tokens.grad
    assert lm_token_grad is not None and lm_token_grad.abs().sum() > 0, "No gradient backpropagated to landmark_tokens!"
    print("   ✓ Backward pass successful! Landmark tokens received valid gradients.")

def test_rasterization_and_diagnostics():
    print("\n3. Testing Rasterization and Diagnostic Panel Rendering...")
    B = 2
    pred_cls = np.random.randint(0, 4, size=(B, 64))
    pred_bez = np.random.rand(B, 64, 4, 2).astype(np.float32)
    
    masks = rasterize_bezier_predictions(pred_cls, pred_bez, grid_size=8, canvas_size=1024, stroke_width=35)
    print(f"   Rasterized canvas shape: {masks.shape}, dtype: {masks.dtype}, max class: {masks.max()}")
    assert masks.shape == (B, 1024, 1024)
    
    dummy_img = torch.randn(3, 1024, 1024)
    gt_2d = masks[0]
    pred_2d = masks[1]
    
    metrics = compute_frame_metrics(pred_2d, gt_2d)
    print(f"   Frame metrics computed: Macro Dice={metrics['macro_dice']:.4f}, ASSD={metrics['macro_assd']:.2f}px")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_c = np.array([[0.4, 0.7], [0.5, 0.3], [0.3, 0.5]])
        gt_c   = np.array([[0.4, 0.68], [0.5, 0.32], [0.3, 0.48]])
        _render_patient40_panel(dummy_img, gt_2d, pred_2d, "Patient_40_test.png", tmpdir,
                                pred_centroids=pred_c, gt_centroids=gt_c)
        out_file = os.path.join(tmpdir, "Patient_40_test_diag.png")
        assert os.path.exists(out_file), "Diagnostic montage was not saved"
        print(f"   ✓ Diagnostic montage created successfully: {os.path.getsize(out_file)} bytes")

def main():
    print("=" * 70)
    print("🧪 EXPERIMENT_4 LOCAL VERIFICATION TEST")
    print("=" * 70)
    test_ast()
    test_decoder_forward_backward()
    test_rasterization_and_diagnostics()
    print("\n" + "=" * 70)
    print("🎉 ALL EXPERIMENT_4 VERIFICATION CHECKS PASSED!")
    print("=" * 70)

if __name__ == '__main__':
    main()
