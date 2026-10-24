import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNLossSuite(nn.Module):
    """
    Multi-Task Loss Suite for EXP_08 CNN-LSTM-MDN Sequential Landmark Detection.
    
    Five loss terms, computed per-instance and averaged across all valid instances in the batch:
    
    1. L_mdn:   Gaussian Mixture NLL  — captures coordinate uncertainty
    2. L_point: Expected-Point Smooth-L1 — anchors predictions to GT coordinates  
    3. L_dir:   Directional Cosine    — enforces tangent alignment along the polyline
    4. L_mask:  Rasterized Mask Dice  — dense spatial overlap supervision
    5. L_eos:   End-of-Sequence BCE   — teaches when to stop drawing
    
    Total: L = λ_mdn·L_mdn + λ_point·L_point + λ_dir·L_dir + λ_mask·L_mask + λ_eos·L_eos
    """
    
    def __init__(self, config):
        super().__init__()
        self.lambda_mdn = config.lambda_mdn
        self.lambda_point = config.lambda_point
        self.lambda_dir = config.lambda_dir
        self.lambda_mask = config.lambda_mask
        self.lambda_eos = config.lambda_eos
        self.soft_mask_sigma = config.soft_mask_sigma
        self.mask_render_size = config.mask_render_size
        self.M = config.mdn_num_components
    
    def _gaussian_nll(self, mdn_params_list: list, gt_polyline: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian Mixture Negative Log-Likelihood.
        
        For each step t, the log-likelihood under the mixture model is:
            log p(u_t, v_t) = log Σ_j π_j · N(u_t | μ_x_j, σ_x_j) · N(v_t | μ_y_j, σ_y_j)
        
        Uses log-sum-exp for numerical stability.
        Independent (non-correlated) bivariate Gaussians: N(u,v) = N(u)·N(v).
        
        Args:
            mdn_params_list: list of K dicts, each with pi(M), mu_x(M), mu_y(M), sigma_x(M), sigma_y(M)
            gt_polyline: (K, 2) ground truth coordinates in [0, 1]^2
            
        Returns:
            scalar NLL loss (lower = better fit)
        """
        K = gt_polyline.shape[0]
        nll_total = 0.0
        
        for t in range(K):
            params = mdn_params_list[t]
            u_gt = gt_polyline[t, 0]  # scalar
            v_gt = gt_polyline[t, 1]  # scalar
            
            pi = params["pi"]          # (M,)
            mu_x = params["mu_x"]      # (M,)
            mu_y = params["mu_y"]      # (M,)
            sigma_x = params["sigma_x"]  # (M,)
            sigma_y = params["sigma_y"]  # (M,)
            
            # Log probability of each component (independent bivariate Gaussian)
            # log N(u | μ_x, σ_x) = -0.5 * ((u - μ_x)/σ_x)^2 - log(σ_x) - 0.5*log(2π)
            log_norm_x = -0.5 * ((u_gt - mu_x) / sigma_x) ** 2 - torch.log(sigma_x) - 0.5 * math.log(2 * math.pi)
            log_norm_y = -0.5 * ((v_gt - mu_y) / sigma_y) ** 2 - torch.log(sigma_y) - 0.5 * math.log(2 * math.pi)
            
            # log p(u, v | component j) = log N_x + log N_y (independent)
            log_component = log_norm_x + log_norm_y  # (M,)
            
            # log p(u, v) = log Σ_j π_j exp(log_component_j)
            #             = log_sum_exp(log(π_j) + log_component_j)
            log_pi = torch.log(pi.clamp(min=1e-8))  # (M,)
            log_mixture = torch.logsumexp(log_pi + log_component, dim=0)  # scalar
            
            nll_total = nll_total - log_mixture
        
        return nll_total / K
    
    def _point_smooth_l1(self, expected_points: torch.Tensor, gt_polyline: torch.Tensor) -> torch.Tensor:
        """
        Expected-Point Smooth-L1 Loss.
        
        p̂_t = Σ_j π_j [μ_x_j, μ_y_j]  (already computed as expected_points)
        L_point = (1/K) Σ_t SmoothL1(p̂_t, p_t^gt)
        
        Args:
            expected_points: (K, 2) expected coordinates from MDN mixture
            gt_polyline: (K, 2) ground truth coordinates
            
        Returns:
            scalar Smooth-L1 loss
        """
        return F.smooth_l1_loss(expected_points, gt_polyline, reduction='mean', beta=0.01)
    
    def _directional_cosine(self, expected_points: torch.Tensor, gt_polyline: torch.Tensor) -> torch.Tensor:
        """
        Directional Cosine Alignment Loss.
        
        Enforces that predicted tangent vectors align with GT tangent vectors:
        ê_t = p̂_t - p̂_{t-1},  e_t^gt = p_t^gt - p_{t-1}^gt
        L_dir = (1/(K-1)) Σ_{t=2}^{K} (1 - cos(ê_t, e_t^gt))
        
        Args:
            expected_points: (K, 2)
            gt_polyline: (K, 2)
            
        Returns:
            scalar directional loss
        """
        K = expected_points.shape[0]
        if K < 2:
            return torch.tensor(0.0, device=expected_points.device)
        
        # Compute edge vectors
        pred_edges = expected_points[1:] - expected_points[:-1]  # (K-1, 2)
        gt_edges = gt_polyline[1:] - gt_polyline[:-1]           # (K-1, 2)
        
        # Cosine similarity per edge
        eps = 1e-6
        pred_norm = pred_edges.norm(dim=-1, keepdim=True).clamp(min=eps)  # (K-1, 1)
        gt_norm = gt_edges.norm(dim=-1, keepdim=True).clamp(min=eps)      # (K-1, 1)
        
        cos_sim = (pred_edges * gt_edges).sum(dim=-1) / (pred_norm.squeeze(-1) * gt_norm.squeeze(-1))  # (K-1,)
        cos_sim = cos_sim.clamp(-1.0, 1.0)
        
        return (1.0 - cos_sim).mean()
    
    def _rasterize_soft_mask(self, expected_points: torch.Tensor, H: int, W: int, sigma: float) -> torch.Tensor:
        """
        Differentiable soft mask rendering via Gaussian splatting along the polyline.
        
        For each pixel (i, j), compute its minimum distance to any line segment in the polyline,
        then apply a Gaussian kernel: mask(i,j) = exp(-d^2 / (2σ^2)).
        
        This is differentiable w.r.t. expected_points through the distance computation.
        
        Args:
            expected_points: (K, 2) predicted polyline in [0, 1]^2
            H, W: mask dimensions
            sigma: Gaussian kernel width in pixels
            
        Returns:
            soft_mask: (H, W) differentiable soft mask in [0, 1]
        """
        device = expected_points.device
        K = expected_points.shape[0]
        
        # Scale points to pixel space
        points_px = expected_points.clone()
        points_px[:, 0] = points_px[:, 0] * (W - 1)  # u → x
        points_px[:, 1] = points_px[:, 1] * (H - 1)  # v → y
        
        # Create pixel coordinate grid
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H, W)
        
        # For computational efficiency, compute distance to each segment and take minimum
        # Each segment is from points_px[k] to points_px[k+1]
        min_dist_sq = torch.full((H, W), float('inf'), device=device)
        
        for k in range(K - 1):
            p0 = points_px[k]    # (2,) [x, y]
            p1 = points_px[k+1]  # (2,) [x, y]
            
            # Vector from p0 to p1
            seg = p1 - p0  # (2,)
            seg_len_sq = (seg * seg).sum().clamp(min=1e-8)
            
            # Project each pixel onto the segment: t = dot(pixel - p0, seg) / |seg|^2
            # pixel_to_p0: (H, W, 2)
            dx = grid_x - p0[0]  # (H, W)
            dy = grid_y - p0[1]  # (H, W)
            
            t = (dx * seg[0] + dy * seg[1]) / seg_len_sq  # (H, W)
            t = t.clamp(0.0, 1.0)  # Clamp to segment
            
            # Closest point on segment
            closest_x = p0[0] + t * seg[0]  # (H, W)
            closest_y = p0[1] + t * seg[1]  # (H, W)
            
            # Squared distance from pixel to closest point
            dist_sq = (grid_x - closest_x) ** 2 + (grid_y - closest_y) ** 2  # (H, W)
            
            min_dist_sq = torch.minimum(min_dist_sq, dist_sq)
        
        # Also include distance to isolated last point
        dx_last = grid_x - points_px[-1, 0]
        dy_last = grid_y - points_px[-1, 1]
        dist_sq_last = dx_last ** 2 + dy_last ** 2
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq_last)
        
        # Gaussian kernel
        soft_mask = torch.exp(-min_dist_sq / (2.0 * sigma ** 2))  # (H, W)
        
        return soft_mask
    
    def _dice_loss(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        Dice Loss: 1 - (2 * |P ∩ G| + ε) / (|P| + |G| + ε)
        
        Args:
            pred_mask: (H, W) soft predicted mask in [0, 1]
            gt_mask: (H, W) binary GT mask
            
        Returns:
            scalar Dice loss
        """
        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()
        return 1.0 - (2.0 * intersection + eps) / (union + eps)
    
    def _eos_bce(self, mdn_params_list: list, num_real_points: int) -> torch.Tensor:
        """
        End-of-Sequence Binary Cross-Entropy Loss.
        
        e_t^gt = 0 for t ≤ K (real points), 1 for t > K (padding).
        Since we teacher-force exactly K steps, all steps have eos_gt = 0,
        but we add a gentle signal at the last step to push eos toward 1.
        
        Args:
            mdn_params_list: list of K dicts with 'eos' probability
            num_real_points: K (number of actual points in the polyline)
            
        Returns:
            scalar BCE loss
        """
        K = len(mdn_params_list)
        eos_preds = torch.stack([p["eos"] for p in mdn_params_list])  # (K,)
        
        # Target: 0 for all steps except the last, which is 1 (end of sequence)
        eos_targets = torch.zeros(K, device=eos_preds.device)
        eos_targets[-1] = 1.0  # Last point signals end
        
        return F.binary_cross_entropy(eos_preds, eos_targets, reduction='mean')
    
    def forward(
        self,
        instance_outputs: list,
        gt_polylines: list,
        gt_masks: list
    ) -> tuple:
        """
        Compute total loss across all valid instances in the batch.
        
        Args:
            instance_outputs: list of dicts from decoder.forward_teacher_forced(),
                              each containing mdn_params, expected_points, raw_outputs
            gt_polylines: list of (K, 2) GT polyline tensors
            gt_masks: list of (H, W) GT binary mask tensors
            
        Returns:
            total_loss: scalar
            loss_dict: dict of individual loss values for logging
        """
        num_instances = len(instance_outputs)
        if num_instances == 0:
            device = gt_polylines[0].device if len(gt_polylines) > 0 else torch.device('cpu')
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"l_mdn": 0.0, "l_point": 0.0, "l_dir": 0.0, "l_mask": 0.0, "l_eos": 0.0}
        
        l_mdn_total = 0.0
        l_point_total = 0.0
        l_dir_total = 0.0
        l_mask_total = 0.0
        l_eos_total = 0.0
        
        for idx in range(num_instances):
            out = instance_outputs[idx]
            gt_poly = gt_polylines[idx]   # (K, 2)
            gt_mask = gt_masks[idx]       # (H, W)
            
            mdn_params = out["mdn_params"]
            expected_pts = out["expected_points"]  # (K, 2)
            K = gt_poly.shape[0]
            
            # 1. Gaussian Mixture NLL
            l_mdn = self._gaussian_nll(mdn_params, gt_poly)
            
            # 2. Expected-Point Smooth-L1
            l_point = self._point_smooth_l1(expected_pts, gt_poly)
            
            # 3. Directional Cosine
            l_dir = self._directional_cosine(expected_pts, gt_poly)
            
            # 4. Rasterized Mask Dice
            H, W = gt_mask.shape
            soft_mask = self._rasterize_soft_mask(expected_pts, H, W, self.soft_mask_sigma)
            l_mask = self._dice_loss(soft_mask, gt_mask)
            
            # 5. End-of-Sequence BCE
            l_eos = self._eos_bce(mdn_params, K)
            
            l_mdn_total = l_mdn_total + l_mdn
            l_point_total = l_point_total + l_point
            l_dir_total = l_dir_total + l_dir
            l_mask_total = l_mask_total + l_mask
            l_eos_total = l_eos_total + l_eos
        
        # Average across instances
        n = float(num_instances)
        l_mdn_avg = l_mdn_total / n
        l_point_avg = l_point_total / n
        l_dir_avg = l_dir_total / n
        l_mask_avg = l_mask_total / n
        l_eos_avg = l_eos_total / n
        
        # Weighted total
        total_loss = (
            self.lambda_mdn * l_mdn_avg +
            self.lambda_point * l_point_avg +
            self.lambda_dir * l_dir_avg +
            self.lambda_mask * l_mask_avg +
            self.lambda_eos * l_eos_avg
        )
        
        def _to_float(v):
            return v.item() if isinstance(v, torch.Tensor) else float(v)
        
        loss_dict = {
            "l_mdn": _to_float(l_mdn_avg),
            "l_point": _to_float(l_point_avg),
            "l_dir": _to_float(l_dir_avg),
            "l_mask": _to_float(l_mask_avg),
            "l_eos": _to_float(l_eos_avg),
            "total": _to_float(total_loss)
        }
        
        return total_loss, loss_dict
