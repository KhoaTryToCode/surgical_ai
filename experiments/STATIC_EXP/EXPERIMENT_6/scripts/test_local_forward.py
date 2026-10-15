"""
Verification script for EXPERIMENT_6.
Tests:
  1. Gaussian heatmap generation (visible vs. absent points).
  2. Peak coordinate and visibility decoding.
  3. Model forward pass with dummy tensor (1, 3, 1024, 1024).
  4. Multi-task loss backward autograd check.
"""
import os
import sys
import torch

_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiments.EXPERIMENT_6.utils.heatmap_utils import generate_gaussian_heatmaps, extract_peak_coords
from experiments.EXPERIMENT_6.models.heatmap_steered_m2f import HeatmapSteeredMask2Former
from experiments.EXPERIMENT_6.models.losses import HeatmapSteeredLoss


def test_heatmap_generation():
    print("🧪 1. Testing Gaussian Heatmap Generation...")
    coords = torch.tensor([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1], [0.0, 0.0]], dtype=torch.float32)
    vis = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32) # Points 2 and 3 are absent
    
    hm = generate_gaussian_heatmaps(coords, vis, map_size=64, sigma=2.0)
    assert hm.shape == (4, 64, 64), f"Wrong shape: {hm.shape}"
    
    # Check visible points
    assert torch.isclose(hm[0].max(), torch.tensor(1.0)), f"Visible peak not 1.0: {hm[0].max()}"
    assert torch.isclose(hm[1].max(), torch.tensor(1.0)), f"Visible peak not 1.0: {hm[1].max()}"
    
    # Check absent points are clean zeros
    assert (hm[2] == 0.0).all(), "Absent point 2 is not all zeros!"
    assert (hm[3] == 0.0).all(), "Absent point 3 is not all zeros!"
    print("   ✅ Heatmap generation verified: visible points have Gaussian peaks, absent points are clean 0.0.")


def test_peak_decoding():
    print("🧪 2. Testing Peak Coordinate Decoding...")
    coords = torch.tensor([[[0.5, 0.5], [0.25, 0.75], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    vis = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    hm = generate_gaussian_heatmaps(coords, vis, map_size=64, sigma=2.0)
    
    pred_coords, pred_vis, confs = extract_peak_coords(hm, threshold=0.3)
    assert pred_coords.shape == (1, 4, 2)
    assert pred_vis.shape == (1, 4)
    
    # Check coordinate accuracy
    err_0 = torch.norm(pred_coords[0, 0] - coords[0, 0])
    assert err_0 < 0.03, f"Decoded peak 0 error too large: {err_0}"
    assert pred_vis[0, 0] == 1.0 and pred_vis[0, 1] == 1.0
    assert pred_vis[0, 2] == 0.0 and pred_vis[0, 3] == 0.0
    print("   ✅ Peak coordinate decoding verified: accurately recovers coords and existence flags.")


def test_model_forward():
    print("🧪 3. Testing Model Forward Pass & Autograd...")
    device = torch.device('cpu')
    model = HeatmapSteeredMask2Former(num_labels=4).to(device)
    model.train()
    
    dummy_img = torch.randn(1, 3, 1024, 1024, device=device)
    mask_labels = [torch.zeros(3, 1024, 1024, device=device)]
    class_labels = [torch.tensor([1, 2, 3], device=device)]
    gt_heatmaps = torch.zeros(1, 4, 64, 64, device=device)
    gt_heatmaps[0, 0, 32, 32] = 1.0
    
    outputs = model(dummy_img, mask_labels=mask_labels, class_labels=class_labels)
    
    assert 'pred_heatmaps' in outputs, "Missing pred_heatmaps in output!"
    assert outputs['pred_heatmaps'].shape == (1, 4, 64, 64)
    assert outputs['masks_queries_logits'].shape == (1, 100, 256, 256)
    assert outputs['class_queries_logits'].shape == (1, 100, 5)
    
    # Test loss & backward
    criterion = HeatmapSteeredLoss(lambda_m2f=1.0, lambda_heatmap=1.0)
    loss, loss_dict = criterion(outputs, gt_heatmaps)
    print(f"   Loss dict: {loss_dict}")
    
    loss.backward()
    print("   ✅ Backward autograd passed cleanly without errors!")


if __name__ == '__main__':
    test_heatmap_generation()
    test_peak_decoding()
    test_model_forward()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
