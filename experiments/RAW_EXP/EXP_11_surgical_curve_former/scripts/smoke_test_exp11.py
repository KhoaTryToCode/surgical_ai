"""
EXP_11: Mathematical Smoke Test
=================================
Validates the entire SurgicalCurveFormer pipeline on synthetic data
BEFORE any real training is attempted.

Checks verified:
  [1] Bernstein partition-of-unity: each row of B sums to 1.0
  [2] Bézier endpoint interpolation: B(0) = P0, B(1) = P5
  [3] ACPI identity: when Δb = 0, b_i^j = c_i  (spatial anchor init)
  [4] ACPI bounded: all output ctrl pts ∈ (0, 1)^2
  [5] BCRNet annealing: lambda_d(0) ≈ 1, lambda_d(∞) ≈ 0, lambda_d(10) = 0.5
  [6] Full model forward pass: correct output shapes
  [7] Loss computation: computes finite, non-NaN total loss
  [8] Backward pass: gradients flow through ctrl_pts ← HCR ← ACPI
  [9] Soft rasterizer: output in [0, 1], gradients non-zero
  [10] Dataset __getitem__: returns correct keys and tensor shapes
"""
import os
import sys
import math

exp11_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
workspace_root = os.path.abspath(os.path.join(exp11_root, ".."))
if exp11_root not in sys.path:
    sys.path.insert(0, exp11_root)

import numpy as np
import torch

# Force CPU for smoke test (no GPU required)
DEVICE = torch.device("cpu")
PASS = "✅ PASS"
FAIL = "❌ FAIL"


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    if not condition:
        raise AssertionError(f"SMOKE TEST FAILED: {name}\n{detail}")


def test_bernstein_partition_of_unity():
    """[1] Bernstein basis rows must sum to 1.0 for all t ∈ [0,1]."""
    from models.bezier_ops import bernstein_basis_matrix
    for degree in [3, 5]:
        B = bernstein_basis_matrix(num_samples=100, degree=degree)
        row_sums = B.sum(axis=1)
        max_err = np.abs(row_sums - 1.0).max()
        check(
            f"Bernstein partition-of-unity (degree={degree})",
            max_err < 1e-5,
            f"max |row_sum - 1| = {max_err:.2e}"
        )


def test_bezier_endpoints():
    """[2] B(t=0) must equal P0 and B(t=1) must equal P5."""
    from models.bezier_ops import bernstein_basis_matrix
    K = 6
    degree = K - 1
    ctrl = torch.tensor([
        [0.1, 0.2], [0.25, 0.4], [0.5, 0.6],
        [0.6, 0.7], [0.75, 0.8], [0.9, 0.9]
    ], dtype=torch.float32)  # (6, 2)

    # B at t=0: b_{0,5}(0) = 1, all others = 0 → P(0) = P0
    # B at t=1: b_{5,5}(1) = 1, all others = 0 → P(1) = P5
    from models.bezier_ops import evaluate_bezier_torch
    sampled = evaluate_bezier_torch(ctrl.unsqueeze(0), num_samples=100)  # (1, 100, 2)
    P0_pred = sampled[0, 0]   # t=0
    P5_pred = sampled[0, -1]  # t=1

    err0 = (P0_pred - ctrl[0]).abs().max().item()
    err5 = (P5_pred - ctrl[-1]).abs().max().item()
    check("Bézier endpoint B(0) = P0", err0 < 1e-4, f"err={err0:.2e}")
    check("Bézier endpoint B(1) = P5", err5 < 1e-4, f"err={err5:.2e}")


def test_acpi_identity():
    """[3] When Δb = 0, ACPI output must equal spatial anchor c_i."""
    from models.bezier_ops import build_coord_grid, acpi_bounded_offset

    H_f, W_f = 16, 16  # f4 for 512px ÷ 32
    grid = build_coord_grid(H_f, W_f)  # (1, 1, 2, H, W)
    B_, M_, K = 2, 3, 6
    delta_b = torch.zeros(B_, M_, K * 2, H_f, W_f)  # All-zero offsets

    # ACPI formula: b = sigma(0 + logit(c)) = sigma(logit(c)) = c
    ctrl_map = acpi_bounded_offset(delta_b, grid)  # (B, M, K, 2, H, W)
    # Expected: ctrl_map should equal grid (broadcast over B, M, K)
    grid_expanded = grid.squeeze(0).squeeze(0)  # (2, H, W)
    # ctrl_map[:, :, :, :, :] should all equal grid_expanded
    ctrl_ref = ctrl_map[0, 0, 0]  # (2, H, W)
    err = (ctrl_ref - grid_expanded).abs().max().item()
    check("ACPI identity (Δb=0 → b=c)", err < 1e-4, f"max err={err:.2e}")


def test_acpi_bounded():
    """[4] All ACPI output ctrl pts must be in (0, 1)^2."""
    from models.bezier_ops import build_coord_grid, acpi_bounded_offset

    H_f, W_f = 16, 16
    grid = build_coord_grid(H_f, W_f)
    B_, M_, K = 2, 3, 6
    # Large random offsets — output must still be in (0, 1)
    delta_b = torch.randn(B_, M_, K * 2, H_f, W_f) * 10.0

    ctrl_map = acpi_bounded_offset(delta_b, grid)
    in_bounds = (ctrl_map > 0.0).all() and (ctrl_map < 1.0).all()
    min_v = ctrl_map.min().item()
    max_v = ctrl_map.max().item()
    check(
        "ACPI bounded (all ctrl pts ∈ (0,1)^2)",
        in_bounds,
        f"range=[{min_v:.4f}, {max_v:.4f}]"
    )


def test_annealing():
    """[5] BCRNet sigmoid annealing: lambda_d(0) ≈ 1, lambda_d(10) = 0.5, lambda_d(∞) ≈ 0."""
    from models.losses import compute_lambda_d

    lam_0  = compute_lambda_d(epoch=0,   center=10.0, slope=2.0)
    lam_10 = compute_lambda_d(epoch=10,  center=10.0, slope=2.0)
    lam_50 = compute_lambda_d(epoch=50,  center=10.0, slope=2.0)

    # At epoch 0: z = (0-10)/2 = -5, sigma(-5) ≈ 0.0067, lambda_d ≈ 0.993
    check("Annealing lambda_d(epoch=0)  ≈ 1.0", lam_0 > 0.98, f"lambda_d={lam_0:.4f}")
    check("Annealing lambda_d(epoch=10) = 0.5", abs(lam_10 - 0.5) < 0.01, f"lambda_d={lam_10:.4f}")
    check("Annealing lambda_d(epoch=50) ≈ 0.0", lam_50 < 0.02, f"lambda_d={lam_50:.4f}")


def test_full_model_forward():
    """[6] Full model forward pass: verify output shapes are correct."""
    from models.surgical_curve_former import SurgicalCurveFormer

    B = 2
    model = SurgicalCurveFormer(
        num_classes=3,
        image_size=512,
        in_chans=4,
        fpn_channels=64,       # Reduced for speed in smoke test
        bezier_ctrl_pts=6,
        acpi_top_k=4,
        hcr_stages=2,
        hcr_embed_dim=64,
        hcr_heads=4,
        hcr_n_ref_pts=10,      # Reduced N for speed
        hcr_deform_pts=2,
        vit_backbone="vit_tiny_patch16_224",  # Tiny for CI speed
        vit_embed_dim=192,
        vit_pretrained=False,
        exist_attn_heads=4,
        raster_render_size=32,
        raster_num_samples=16,
    ).to(DEVICE)
    model.eval()

    x = torch.randn(B, 4, 512, 512, device=DEVICE)

    with torch.no_grad():
        out = model(x)

    M = 3
    K_top = 4
    K_ctrl = 6
    N_ref = 10
    R = 32

    check("seg_logits_list length = 4", len(out["seg_logits_list"]) == 4)
    check("seg_logits_list[0] shape",
          out["seg_logits_list"][0].shape == (B, M, 128, 128),
          str(out["seg_logits_list"][0].shape))
    check("proposals shape",
          out["proposals"].shape == (B, M, K_top, K_ctrl, 2),
          str(out["proposals"].shape))
    check("proposals in (0,1)",
          (out["proposals"] > 0).all() and (out["proposals"] < 1).all())
    check("final_ctrl_pts shape",
          out["final_ctrl_pts"].shape == (B, M, K_top, K_ctrl, 2),
          str(out["final_ctrl_pts"].shape))
    check("exist_logits shape",
          out["exist_logits"].shape == (B, M),
          str(out["exist_logits"].shape))
    check("exist_probs in [0,1]",
          (out["exist_probs"] >= 0).all() and (out["exist_probs"] <= 1).all())
    check("soft_masks shape",
          out["soft_masks"].shape == (B, M, R, R),
          str(out["soft_masks"].shape))
    check("soft_masks in [0,1]",
          (out["soft_masks"] >= 0).all() and (out["soft_masks"] <= 1).all())
    check("score_logits shape (ACPI)",
          out["score_logits"].shape[0] == B and out["score_logits"].shape[1] == M)

    return model, out


def test_loss_and_backward():
    """[7,8] Loss computes finite scalar; gradients flow back to ACPI ctrl pts."""
    from models.surgical_curve_former import SurgicalCurveFormer
    from models.losses import SurgicalCurveFormerLoss

    B = 2
    model = SurgicalCurveFormer(
        num_classes=3, image_size=256, in_chans=4,
        fpn_channels=32, bezier_ctrl_pts=6, acpi_top_k=4,
        hcr_stages=2, hcr_embed_dim=32, hcr_heads=2,
        hcr_n_ref_pts=10, hcr_deform_pts=2,
        vit_backbone="vit_tiny_patch16_224", vit_embed_dim=192,
        vit_pretrained=False, exist_attn_heads=2,
        raster_render_size=32, raster_num_samples=8,
    ).to(DEVICE).train()

    criterion = SurgicalCurveFormerLoss(
        num_hcr_stages=2, raster_render_size=32,
        raster_num_samples=8, target_size=256
    ).to(DEVICE)

    x = torch.randn(B, 4, 256, 256, device=DEVICE)
    out = model(x)

    M = 3
    H_f, W_f = 256 // 32, 256 // 32  # 8 × 8

    target_dict = {
        "target_masks":
            torch.zeros(B, M, 256, 256, device=DEVICE),
        "acpi_target_score":
            torch.zeros(B, M, H_f, W_f, device=DEVICE),
        "target_ctrl_pts":
            torch.rand(B, M, 6, 2, device=DEVICE),
        "active_mask":
            torch.tensor([[True, True, False]] * B, device=DEVICE),
        "target_render_masks":
            torch.zeros(B, M, 32, 32, device=DEVICE),
    }

    loss_dict = criterion(out, target_dict, epoch=5)
    total = loss_dict["L_total"]

    check("Loss is finite scalar", total.isfinite().item() and total.ndim == 0,
          f"L_total={total.item():.4f}")
    check("Loss is non-negative", total.item() >= 0, f"L_total={total.item():.4f}")

    # Backward
    total.backward()

    # Check gradients on ACPI conv head (first class, out_conv)
    acpi_grad = model.acpi.heads[0].out_conv.weight.grad
    has_grad = acpi_grad is not None and acpi_grad.abs().max().item() > 0
    check("Gradients flow to ACPI out_conv", has_grad,
          f"max_grad={acpi_grad.abs().max().item():.2e}" if acpi_grad is not None else "grad=None")

    # Check gradients on HCR offset MLP
    hcr_grad = model.hcr.stages[0].offset_mlp[-2].weight.grad
    has_hcr_grad = hcr_grad is not None and hcr_grad.abs().max().item() > 0
    check("Gradients flow to HCR offset_mlp", has_hcr_grad,
          f"max_grad={hcr_grad.abs().max().item():.2e}" if hcr_grad is not None else "grad=None")


def test_soft_rasterizer_gradients():
    """[9] Soft rasterizer output ∈ [0,1] and gradients flow to ctrl_pts."""
    from models.losses import SoftRasterizer

    raster = SoftRasterizer(render_size=32, num_samples=16, sigma_px=2.0, target_size=256)
    B, M, K_top, K_ctrl = 2, 3, 4, 6
    ctrl_pts = torch.rand(B, M, K_top, K_ctrl, 2, requires_grad=True)

    masks = raster(ctrl_pts)
    check("Soft rasterizer in [0,1]",
          (masks >= 0).all() and (masks <= 1).all(),
          f"range=[{masks.min().item():.4f}, {masks.max().item():.4f}]")

    # Compute a dummy loss and backprop
    loss = (1.0 - masks).mean()
    loss.backward()

    grad_max = ctrl_pts.grad.abs().max().item()
    check("Soft rasterizer gradient to ctrl_pts > 0", grad_max > 0,
          f"max grad={grad_max:.2e}")


def test_dataset_shapes():
    """[10] Dataset returns correctly shaped tensors."""
    from utils.dataset import SurgicalCurveFormerDataset

    ds = SurgicalCurveFormerDataset(
        dataset_dir="/nonexistent/path",
        mode="train",
        image_size=256,
        use_depth=True,
        acpi_stride=32,
        render_size=32,
        bezier_degree=5,
    )
    sample = ds[0]

    M = 3
    K = 6  # degree 5
    H_f = W_f = 256 // 32  # 8
    R = 32

    check("image shape",        sample["image"].shape == (4, 256, 256),      str(sample["image"].shape))
    check("target_masks shape", sample["target_masks"].shape == (M, 256, 256), str(sample["target_masks"].shape))
    check("target_ctrl_pts",    sample["target_ctrl_pts"].shape == (M, K, 2), str(sample["target_ctrl_pts"].shape))
    check("active_mask",        sample["active_mask"].shape == (M,),          str(sample["active_mask"].shape))
    check("acpi_target_score",  sample["acpi_target_score"].shape == (M, H_f, W_f), str(sample["acpi_target_score"].shape))
    check("target_render_masks",sample["target_render_masks"].shape == (M, R, R),   str(sample["target_render_masks"].shape))


def main():
    print()
    print("=" * 65)
    print("  EXP_11 SurgicalCurveFormer — Mathematical Smoke Tests")
    print("=" * 65)

    tests = [
        ("Bernstein Partition-of-Unity",    test_bernstein_partition_of_unity),
        ("Bézier Endpoint Interpolation",   test_bezier_endpoints),
        ("ACPI Identity (Δb=0 → b=c)",     test_acpi_identity),
        ("ACPI Bounded Output ∈ (0,1)^2",  test_acpi_bounded),
        ("BCRNet Sigmoid Annealing",         test_annealing),
        ("Full Model Forward Pass Shapes",  test_full_model_forward),
        ("Loss + Backward Gradient Flow",   test_loss_and_backward),
        ("Soft Rasterizer Gradients",       test_soft_rasterizer_gradients),
        ("Dataset Output Shapes",           test_dataset_shapes),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[Test] {name}")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌  {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ❌  EXCEPTION: {e}")
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 65)
    print(f"  Results: {passed} passed, {failed} failed / {len(tests)} total")
    print("=" * 65)

    if failed > 0:
        print("\n⚠️  FIX FAILING TESTS BEFORE STARTING TRAINING.")
        sys.exit(1)
    else:
        print("\n🚀  All math verified. Safe to proceed to training.")


if __name__ == "__main__":
    main()
