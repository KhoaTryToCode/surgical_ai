"""
Mathematical Test 19: Latent Space Dimensionality, Separability & Lipschitz Continuity.
Analyzes the latent space representations of SurgicalCurveFormer v2:

Evaluates:
1. Effective Dimensionality & Singular Value Spectrum (Roy-Vetterli Effective Rank):
   R_eff = exp( - sum p_i * ln(p_i) ), where p_i = sigma_i / sum(sigma)
   Verifies whether the latent space spans >= 12 degrees of freedom (preventing dimensional collapse).
2. Category Orthogonality & Fisher Separation Ratio:
   Inter-Class Cosine Distance vs Intra-Class Dispersion: S = d_inter / d_intra.
3. Geodesic Curve Smoothness & Lipschitz Continuity:
   L = max ||q_{i+1} - q_i|| / Delta t along the 26 curve vertex tokens.
4. Operator Norm & Spectral Radius:
   sigma_max(W) of attention layers ensuring dynamical stability.
"""

import sys, os
sys.path.insert(0, os.path.abspath("experiments/EXP_12_surgical_curve_former_v2"))

import numpy as np
import torch
from models.surgical_curve_former_v2 import SurgicalCurveFormerV2


def analyze_latent_space():
    print("=" * 80)
    print("🔬 MATHEMATICAL TEST 19: LATENT SPACE MATHEMATICAL CORRECTNESS")
    print("=" * 80)
    
    device = torch.device("cpu")
    model = SurgicalCurveFormerV2(
        num_classes=3, top_k=10, bezier_order=5, fpn_dim=256, grid_size=128, image_size=512
    ).to(device)
    model.eval()
    
    # 1. Forward pass on mock surgical image
    torch.manual_seed(42)
    mock_input = torch.randn(2, 4, 512, 512, device=device)
    
    with torch.no_grad():
        outputs = model(mock_input)
        
    # Extract HCR queries across stages
    # In HCR v2, q has shape (B, M, K, N, C) = (2, 3, 10, 26, 256)
    # Let's inspect stage 3 queries directly
    hcr = model.hcr
    fpn_feats = model.resnet_fpn(mock_input)
    init_curves = outputs["acpi_curves"]
    init_scores = outputs["acpi_scores"]
    
    B, M, K, _, _ = init_curves.shape
    curves_flat = init_curves.view(B, M, K, 12)
    q0 = hcr.init_query_proj(curves_flat).unsqueeze(3).expand(-1, -1, -1, 26, -1)
    
    ref_pts1, norm1 = hcr.sample_curve_and_normals(init_curves)
    q1, dp1, _ = hcr.stage1(q0, fpn_feats[2], fpn_feats[3], ref_pts1, norm1)
    c1 = hcr.refit_curves(ref_pts1 + dp1)
    
    ref_pts2, norm2 = hcr.sample_curve_and_normals(c1)
    q2, dp2, _ = hcr.stage2(q1, fpn_feats[1], fpn_feats[2], ref_pts2, norm2)
    c2 = hcr.refit_curves(ref_pts2 + dp2)
    
    ref_pts3, norm3 = hcr.sample_curve_and_normals(c2)
    q3, dp3, _ = hcr.stage3(q2, fpn_feats[0], fpn_feats[1], ref_pts3, norm3)
    
    # Analyze Latent Space at Stage 3: (B=2, M=3, K=10, N=26, C=256)
    tokens = q3[0].detach()  # (3, 10, 26, 256)
    
    # ── 1. Singular Value Spectrum & Effective Rank ─────────────────────────────
    # Flatten all 3 * 10 * 26 = 780 tokens of dim 256
    X = tokens.view(-1, 256).numpy()  # (780, 256)
    # Center X
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Normalized energy spectrum
    p = S / np.sum(S)
    entropy = -np.sum(p * np.log(p + 1e-12))
    eff_rank = np.exp(entropy)
    
    top5_energy = np.sum(S[:5] ** 2) / np.sum(S ** 2) * 100.0
    top20_energy = np.sum(S[:20] ** 2) / np.sum(S ** 2) * 100.0
    
    print("\n▶ 1. Intrinsic Dimensionality & Spectral Rank (Target: R_eff >= 12 degrees of freedom):")
    print(f"  • Total Embedding Dimension:        256 channels")
    print(f"  • Total Active Tokens per Frame:    780 tokens")
    print(f"  • Top-5 Singular Values Energy:     {top5_energy:.2f}%")
    print(f"  • Top-20 Singular Values Energy:    {top20_energy:.2f}%")
    print(f"  • Roy-Vetterli Effective Rank:      {eff_rank:.2f} dimensions")
    print(f"  • Mathematical Dimensional Status:  {'✅ HEALTHY (R_eff >= 12)' if eff_rank >= 12 else '❌ COLLAPSED'}")
    
    # ── 2. Category Orthogonality & Fisher Separation ──────────────────────────
    # Mean category embedding: (3, 256)
    cat_means = np.mean(tokens.view(3, -1, 256).numpy(), axis=1)  # (3, 256)
    # Normalize
    cat_means_norm = cat_means / (np.linalg.norm(cat_means, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarities between distinct classes
    cos_01 = np.dot(cat_means_norm[0], cat_means_norm[1])
    cos_02 = np.dot(cat_means_norm[0], cat_means_norm[2])
    cos_12 = np.dot(cat_means_norm[1], cat_means_norm[2])
    inter_cos_mean = (cos_01 + cos_02 + cos_12) / 3.0
    
    # Intra-class dispersion (variance of tokens around their class mean)
    intra_vars = []
    for m in range(3):
        cls_tokens = tokens[m].view(-1, 256).numpy()
        cls_tokens_norm = cls_tokens / (np.linalg.norm(cls_tokens, axis=1, keepdims=True) + 1e-8)
        cos_to_mean = np.dot(cls_tokens_norm, cat_means_norm[m])
        intra_vars.append(np.std(cos_to_mean))
    intra_dispersion = np.mean(intra_vars)
    
    fisher_ratio = (1.0 - inter_cos_mean) / (intra_dispersion + 1e-6)
    
    print("\n▶ 2. Category Orthogonality & Linear Separability:")
    print(f"  • Falciform vs Ridge Cosine Similarity:    {cos_01:.4f}")
    print(f"  • Falciform vs Silhouette Cosine Sim:     {cos_02:.4f}")
    print(f"  • Ridge vs Silhouette Cosine Sim:         {cos_12:.4f}")
    print(f"  • Mean Inter-Category Orthogonal Angle:   {np.rad2deg(np.arccos(np.clip(inter_cos_mean, -1, 1))):.2f}°")
    print(f"  • Fisher Separation Ratio:                {fisher_ratio:.2f} (Target S > 1.0)")
    print(f"  • Separability Status:                    {'✅ LINEARLY SEPARABLE' if fisher_ratio > 1.0 else '⚠️ OVERLAPPING'}")
    
    # ── 3. Geodesic Smoothness along Curves (Lipschitz Bound) ─────────────────
    # Examine tokens along the 26 points of proposal 0 of Falciform: (26, 256)
    curve_tokens = tokens[0, 0].numpy()  # (26, 256)
    diffs = np.linalg.norm(curve_tokens[1:] - curve_tokens[:-1], axis=-1)  # 25 diffs
    delta_t = 1.0 / 24.0
    lipschitz_estimates = diffs / delta_t
    max_L = np.max(lipschitz_estimates)
    mean_L = np.mean(lipschitz_estimates)
    
    print("\n▶ 3. Geodesic Curve Smoothness & Lipschitz Continuity:")
    print(f"  • Mean Inter-Vertex Latent Step:          {np.mean(diffs):.4f}")
    print(f"  • Maximum Lipschitz Constant L:           {max_L:.2f}")
    print(f"  • Mean Lipschitz Constant L:              {mean_L:.2f}")
    print(f"  • Continuity Status:                      {'✅ SMOOTH MANIFOLD (L bounded)' if max_L < 100.0 else '⚠️ DISCONTINUOUS'}")
    
    # ── 4. Operator Spectral Norms ───────────────────────────────────────────
    # Compute maximum singular values of projection weights
    w_sample = model.hcr.stage1.on_snake_attn.sample_fusion.weight.data.numpy()
    w_out = model.hcr.stage1.on_snake_attn.out_proj.weight.data.numpy()
    s_sample = np.linalg.norm(w_sample, ord=2)
    s_out = np.linalg.norm(w_out, ord=2)
    
    print("\n▶ 4. Operator Norms & Dynamical Stability (Spectral Radius):")
    print(f"  • ON-Snake Sample Fusion ||W_sample||_2:  {s_sample:.3f}")
    print(f"  • Output Projection ||W_out||_2:          {s_out:.3f}")
    print(f"  • Dynamical Status:                       {'✅ STABLE (Spectral Norm Bounded)' if max(s_sample, s_out) < 10.0 else '⚠️ UNSTABLE'}")


if __name__ == "__main__":
    analyze_latent_space()
