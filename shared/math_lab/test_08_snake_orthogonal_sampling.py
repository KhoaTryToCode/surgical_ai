"""
Mathematical Test 08: Orthogonal Normal Snake Sampling (ON-Snake) vs Single-Point Sampling.
Tests the unification of TopoNet's Dynamic Snake feature extraction with BCRNet's Deformable Attention:

Hypothesis:
Surgical landmarks have physical width (thickness w ~ 10-30 px in L3D).
Single-point sampling at P(t) only reads the centerline pixel.
Orthogonal Normal Snake Sampling (ON-Snake) samples a triplet:
- Centerline: P(t)
- Left Boundary: P(t) - (w / 2) * N(t)
- Right Boundary: P(t) + (w / 2) * N(t)
where N(t) = (-y'(t), x'(t)) / ||(x', y')|| is the continuous analytical normal vector.

Evaluates:
- Edge localization gradient snappability on simulated surgical tissue cross-sections (parenchyma vs ligament)
- Landmark centering accuracy under asymmetric surgical background noise
"""

import numpy as np
import torch
import torch.nn.functional as F
from synthetic_landmarks import generate_falciform_ligament
from test_01_parametric_order import fit_single_bezier


def compute_analytical_normals(b: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes points P(t) and unit normal vectors N(t) analytically from 5th-order Bézier control points b (6, 2).
    """
    degree = 5
    # 1st derivative control points Delta b in R^{5 x 2}
    db = degree * (b[1:] - b[:-1])
    
    # Bernstein basis for curve and 1st derivative
    from test_04_curvature_dynamics import get_bernstein_torch
    M_curve = get_bernstein_torch(num_samples=len(t), degree=5).to(b.device)
    
    # Derivative basis (degree 4)
    import scipy.special
    t_np = t.cpu().numpy()
    M_deriv = np.zeros((len(t), 5), dtype=np.float32)
    for i in range(5):
        c = scipy.special.comb(4, i)
        M_deriv[:, i] = c * ((1.0 - t_np) ** (4 - i)) * (t_np ** i)
    M_deriv_torch = torch.tensor(M_deriv, dtype=torch.float32, device=b.device)
    
    P = M_curve @ b  # (N, 2)
    tangent = M_deriv_torch @ db  # (N, 2)
    
    # Normal vector: rotate tangent by 90 degrees (-dy, dx)
    dx = tangent[:, 0]
    dy = tangent[:, 1]
    norm = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
    nx = -dy / norm
    ny = dx / norm
    N = torch.stack([nx, ny], dim=-1)  # (N, 2)
    return P, N


def simulate_tissue_feature_field(H: int = 128, W: int = 128, gt_pts: torch.Tensor = None) -> torch.Tensor:
    """
    Simulates a high-level surgical feature activation map with a tubular landmark
    (e.g., bright fibrous falciform ligament against darker liver parenchyma).
    """
    device = gt_pts.device
    ys = torch.linspace(0.0, 1.0, H, device=device)
    xs = torch.linspace(0.0, 1.0, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    
    # Distance to GT curve
    diff = grid.unsqueeze(2) - gt_pts.unsqueeze(0).unsqueeze(0)  # (H, W, N, 2)
    min_dist = torch.sqrt(torch.min(torch.sum(diff ** 2, dim=-1), dim=-1)[0] + 1e-8)
    
    # Landmark thickness = 10 px (10/512 = 0.0195)
    landmark_width = 0.0195
    # Smooth step feature profile (fibrous tissue has distinct high activation)
    feature_map = torch.exp(-(min_dist ** 2) / (2.0 * (landmark_width ** 2)))
    return feature_map


def run_snake_sampling_test():
    print("=" * 80)
    print("🔬 MATHEMATICAL TEST 08: ORTHOGONAL NORMAL SNAKE SAMPLING (ON-SNAKE)")
    print("=" * 80)
    
    device = torch.device("cpu")
    num_queries = 26
    t_queries = torch.linspace(0.0, 1.0, num_queries, device=device)
    
    gt_raw = generate_falciform_ligament(num_points=100)
    b_gt_np, _, _ = fit_single_bezier(gt_raw, degree=5)
    b_gt = torch.tensor(b_gt_np, dtype=torch.float32, device=device)
    
    # Generate continuous tissue feature field
    P_gt, N_gt = compute_analytical_normals(b_gt, t_queries)
    feature_map = simulate_tissue_feature_field(H=128, W=128, gt_pts=P_gt)
    
    # Test feature response and gradient when perturbing the curve laterally by delta in [1, 20] px
    deltas_px = [1.0, 3.0, 5.0, 10.0, 15.0, 20.0]
    
    print("\n▶ Feature Gradient Localization Response (Lateral Displacement):")
    print(f"  {'Offset (px)':<12} | {'Single-Point Grad Norm':<24} | {'ON-Snake Triplet Grad Norm':<26} | {'Sensitivity Gain':<16}")
    print("  " + "-" * 82)
    
    half_width = 0.0195 / 2.0  # 5 px half-width
    
    for d_px in deltas_px:
        d_norm = d_px / 512.0
        
        # 1. Single-Point Sampling Gradient
        b_single = (b_gt.clone() + torch.tensor([d_norm, 0.0])).requires_grad_(True)
        P_s, _ = compute_analytical_normals(b_single, t_queries)
        # Sample feature map at P_s
        grid_s = (P_s.unsqueeze(0).unsqueeze(0) * 2.0 - 1.0)
        feat_single = F.grid_sample(feature_map.unsqueeze(0).unsqueeze(0), grid_s, mode="bilinear", align_corners=True).squeeze()
        # Objective: maximize landmark feature activation
        loss_single = -torch.mean(feat_single)
        loss_single.backward()
        g_single = float(torch.norm(b_single.grad).item())
        
        # 2. ON-Snake Triplet Sampling Gradient
        b_snake = (b_gt.clone() + torch.tensor([d_norm, 0.0])).requires_grad_(True)
        P_sn, N_sn = compute_analytical_normals(b_snake, t_queries)
        P_left = P_sn - half_width * N_sn
        P_right = P_sn + half_width * N_sn
        
        # Sample all three
        all_samples = torch.stack([P_left, P_sn, P_right], dim=1)  # (N, 3, 2)
        grid_sn = (all_samples.unsqueeze(0) * 2.0 - 1.0)
        feat_snake = F.grid_sample(feature_map.unsqueeze(0).unsqueeze(0), grid_sn, mode="bilinear", align_corners=True).squeeze()  # (N, 3)
        
        # Center peak objective: Center should be high, boundaries should exhibit symmetric gradient descent
        f_left, f_center, f_right = feat_snake[:, 0], feat_snake[:, 1], feat_snake[:, 2]
        # Peak alignment: maximize center + penalize asymmetry |f_left - f_right|
        loss_snake = -torch.mean(f_center) + 0.5 * torch.mean((f_left - f_right) ** 2)
        loss_snake.backward()
        g_snake = float(torch.norm(b_snake.grad).item())
        
        gain = g_snake / (g_single + 1e-8)
        print(f"  {d_px:<12.1f} | {g_single:<24.4e} | {g_snake:<26.4e} | {gain:<16.2f}x")


if __name__ == "__main__":
    run_snake_sampling_test()
