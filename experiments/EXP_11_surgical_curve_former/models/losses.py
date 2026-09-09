"""
EXP_11: SurgicalCurveFormer Loss Suite
=======================================
All loss functions grounded in BCRNet (arXiv:2506.15279) and EXP_10 dual_domain_loss.py.

Components:
  L_s    = Deep CNN decoder multi-scale Dice (4 FPN levels)
  L_ind  = ACPI Induction BCE  (GT: 1 at GT curve midpoints, 0 elsewhere)
  L_cs   = Per-stage HCR curve confidence Focal Loss
  L_crv  = Per-stage Bézier geometry L1 loss (ctrl pts + sampled pts)
  L_exist= Landmark existence BCE  (EXP_10)
  L_dice = Differentiable soft Gaussian rasterizer Dice  (EXP_10)

Annealing (BCRNet):
  lambda_d = 1 - sigma((epoch - center) / slope)
  where sigma is the sigmoid function.

  Early training (epoch << center = 10):
      lambda_d ≈ 1  → focus entirely on dense supervision (L_s, L_ind)
  Late training (epoch >> 10):
      lambda_d ≈ 0  → focus entirely on curve refinement (L_cs, L_crv)

Total loss:
  L = lambda_d * (lambda_s * L_s + lambda_ind * L_ind)
    + (1 - lambda_d) * sum_h [lambda_cs * L_cs^h + lambda_crv * L_crv^h]
    + lambda_exist * L_exist
    + lambda_dice * L_dice
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bezier_ops import evaluate_bezier_torch


# ─────────────────────────────────────────────────────────────────────────────
# Sigmoid Annealing Schedule (BCRNet)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lambda_d(epoch: int, center: float = 10.0, slope: float = 2.0) -> float:
    """
    BCRNet sigmoid annealing weight for dense supervision.

        lambda_d(epoch) = 1 - sigma((epoch - center) / slope)

    where sigma(z) = 1 / (1 + exp(-z)).

    Args:
        epoch:  Current training epoch (0-indexed).
        center: Transition midpoint (default 10). At epoch=center, lambda_d = 0.5.
        slope:  Transition speed (larger = faster decay).

    Returns:
        lambda_d ∈ (0, 1): weight for dense losses (L_s, L_ind).
        (1 - lambda_d):     weight for curve refinement losses (L_cs, L_crv).
    """
    import math
    z = (float(epoch) - center) / slope
    sigma = 1.0 / (1.0 + math.exp(-z))
    return 1.0 - sigma


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for binary classification:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        p_t = p if y=1 else (1-p)
        alpha_t = alpha if y=1 else (1-alpha)

    Applied to existence logits and HCR confidence logits.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  any shape — raw unnormalized scores.
            targets: same shape — float binary targets in {0, 1}.

        Returns:
            loss: scalar mean focal loss.
        """
        # Numerically stable via log-sum-exp decomposition
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        fl = alpha_t * (1.0 - p_t) ** self.gamma * bce
        return fl.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Soft Dice Loss
# ─────────────────────────────────────────────────────────────────────────────

class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss for differentiable mask supervision.

    Dice = 2 * |pred ∩ gt| / (|pred|^2 + |gt|^2 + eps)

    Applied to:
    1. Multi-level deep CNN decoder outputs (L_s, 4 levels)
    2. Soft rasterizer outputs (L_dice, end-to-end)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, C, H, W)  predicted probability maps in [0, 1].
            target: (B, C, H, W)  ground truth binary masks in {0, 1}.

        Returns:
            loss: scalar (1 - mean Dice score).
        """
        B, C = pred.shape[:2]
        pred_flat = pred.view(B, C, -1)
        target_flat = target.view(B, C, -1)

        intersection = (pred_flat * target_flat).sum(dim=-1)  # (B, C)
        cardinality = pred_flat.pow(2).sum(dim=-1) + target_flat.pow(2).sum(dim=-1)  # (B, C)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)  # (B, C)
        return 1.0 - dice.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable Soft Rasterizer (ported from EXP_10)
# ─────────────────────────────────────────────────────────────────────────────

class SoftRasterizer(nn.Module):
    """
    Differentiable Gaussian soft rasterizer for Bézier curves.

    For each predicted curve, samples N_samp dense trajectory points via
    the Bernstein basis, then computes a Gaussian distance field over the
    rendering grid (R × R):

        mask[x, y] = exp(- d_min(x,y)^2 / (2 * sigma_norm^2))

    where d_min is the minimum squared distance from (x, y) to any trajectory point,
    and sigma_norm = sigma_px / target_size.

    This is fully differentiable: gradients flow from the Dice loss back
    through the distance field → trajectory samples → Bernstein matrix → ctrl_pts.

    Ported and extended from EXP_10 soft_rasterizer.py.
    """
    def __init__(
        self,
        render_size: int = 128,
        num_samples: int = 64,
        sigma_px: float = 2.0,
        target_size: int = 512,
    ):
        super().__init__()
        self.R = render_size
        self.N_samp = num_samples
        self.sigma_px = sigma_px

        # Precompute rendering grid: (1, 1, R, R, 2) in [0, 1]^2
        ys = torch.linspace(0.0, 1.0, render_size)
        xs = torch.linspace(0.0, 1.0, render_size)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)   # (R, R, 2)
        # Shape for broadcasting: (1, 1, 1, R, R, 2) — dims: B, M, K, R, R, 2
        self.register_buffer("grid", grid.unsqueeze(0).unsqueeze(0).unsqueeze(0))

        # Normalized sigma^2: sigma in [0,1] units
        sigma_norm = sigma_px / float(target_size)
        self.inv_two_sigma_sq = 1.0 / (2.0 * sigma_norm ** 2)

    def forward(
        self,
        ctrl_pts: torch.Tensor,
        exist_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Renders Bézier curves into soft probability masks.

        Args:
            ctrl_pts:    (B, M, K_top, K_ctrl, 2)  best proposals' ctrl pts in (0,1)^2.
            exist_probs: (B, M)                     per-class existence probabilities (optional).

        Returns:
            soft_masks: (B, M, R, R)  per-class soft raster masks in [0, 1].
        """
        B, M, K_top, K_ctrl, _ = ctrl_pts.shape
        R = self.R
        grid = self.grid  # (1, 1, 1, R, R, 2)

        # Sample N dense trajectory points per proposal
        ctrl_flat = ctrl_pts.view(B * M * K_top, K_ctrl, 2)
        traj_flat = evaluate_bezier_torch(ctrl_flat, num_samples=self.N_samp)  # (B*M*K, N, 2)
        traj = traj_flat.view(B, M, K_top, self.N_samp, 2)

        # ─── Vectorized min-distance field ──────────────────────────────────
        # grid:     (1, 1, 1, R, R, 2)
        # traj:     (B, M, K_top, N, 2) → reshape for broadcasting
        # We want dist_sq: (B, M, R, R) = min over K_top and N
        # Expand: traj_exp → (B, M, K_top, 1, 1, N, 2)
        traj_exp = traj.unsqueeze(3).unsqueeze(3)  # (B, M, K_top, 1, 1, N, 2)
        # grid_exp: (1, 1, 1, R, R, 1, 2)
        grid_exp = grid.unsqueeze(-2)              # (1, 1, 1, R, R, 1, 2)

        # dist_sq: (B, M, K_top, R, R, N)
        diff = grid_exp - traj_exp
        dist_sq = diff.pow(2).sum(dim=-1)  # (B, M, K_top, R, R, N)

        # Min over N trajectory points and K_top proposals
        # min_dist_sq: (B, M, R, R)
        min_dist_sq = dist_sq.min(dim=-1).values.min(dim=2).values  # (B, M, R, R)

        # Gaussian soft mask: (B, M, R, R)
        soft_mask = torch.exp(-min_dist_sq * self.inv_two_sigma_sq)

        # Modulate by existence probability if provided
        if exist_probs is not None:
            exist_mod = exist_probs.unsqueeze(-1).unsqueeze(-1)  # (B, M, 1, 1)
            soft_mask = soft_mask * exist_mod

        return soft_mask  # (B, M, R, R)


# ─────────────────────────────────────────────────────────────────────────────
# ACPI Induction Loss
# ─────────────────────────────────────────────────────────────────────────────

class ACPIInductionLoss(nn.Module):
    """
    BCRNet Induction Loss for ACPI.

    L_ind = BCE(hat_s_init, s*)

    where s* is 1 at the GT curve midpoints (on f4 = 32×32 map)
    and 0 at all other locations. This provides a dense spatial
    anchor signal that prevents cold-start collapse.

    Args:
        score_logits: (B, M, H_f, W_f)  raw ACPI confidence score logits.
        target_scores: (B, M, H_f, W_f)  GT midpoint indicator masks (float {0, 1}).
    """
    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        # Upweight positive midpoint pixels (very sparse: ~3 pixels per curve on 32x32)
        self.pos_weight = pos_weight

    def forward(
        self,
        score_logits: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            score_logits:  (B, M, H_f, W_f) raw logits.
            target_scores: (B, M, H_f, W_f) binary GT midpoint mask.

        Returns:
            loss: scalar BCE loss with positive weighting.
        """
        device = score_logits.device
        pw = torch.tensor([self.pos_weight], device=device, dtype=score_logits.dtype)
        loss = F.binary_cross_entropy_with_logits(
            score_logits, target_scores.float(),
            pos_weight=pw,
        )
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# HCR Geometry Loss (per stage)
# ─────────────────────────────────────────────────────────────────────────────

class HCRCurveLoss(nn.Module):
    """
    Per-stage Bézier geometry loss for HCR.

    Composed of:
    1. Control point L1:   L1(ctrl_pred, ctrl_gt)
    2. Sampled curve L1:   L1(B·ctrl_pred, B·ctrl_gt)  [25 uniform samples]

    Applied only to the best-matched proposal per GT curve
    (via Hungarian matching on sampled curve Chamfer distance).
    """
    def __init__(self, num_ctrl_pts: int = 6, num_samples: int = 25):
        super().__init__()
        self.K = num_ctrl_pts
        self.N = num_samples

    def forward(
        self,
        ctrl_pts_pred: torch.Tensor,
        ctrl_pts_gt: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ctrl_pts_pred: (B, M, K_top, 6, 2)  predicted proposals in (0,1)^2.
            ctrl_pts_gt:   (B, M, 6, 2)          GT control points per class.
            active_mask:   (B, M)                 bool — True if class present.

        Returns:
            loss: scalar geometry loss (best-proposal L1).
        """
        B, M, K_top, K_ctrl, _ = ctrl_pts_pred.shape
        device = ctrl_pts_pred.device

        if active_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Sample both predicted and GT curves
        # pred_sampled: (B, M, K_top, N, 2)
        ctrl_flat_pred = ctrl_pts_pred.view(B * M * K_top, K_ctrl, 2)
        samp_flat_pred = evaluate_bezier_torch(ctrl_flat_pred, num_samples=self.N)
        pred_sampled = samp_flat_pred.view(B, M, K_top, self.N, 2)

        # gt_sampled: (B, M, N, 2)
        ctrl_flat_gt = ctrl_pts_gt.view(B * M, K_ctrl, 2)
        samp_flat_gt = evaluate_bezier_torch(ctrl_flat_gt, num_samples=self.N)
        gt_sampled = samp_flat_gt.view(B, M, self.N, 2)

        # ─── Best-proposal matching via minimum curve distance ────────────────
        # dist: (B, M, K_top) — mean L2 of sampled curve points to GT
        # gt_sampled expanded: (B, M, 1, N, 2)
        diff_samp = pred_sampled - gt_sampled.unsqueeze(2)  # (B, M, K_top, N, 2)
        dist_per_prop = diff_samp.pow(2).sum(dim=-1).mean(dim=-1)  # (B, M, K_top)
        best_idx = dist_per_prop.argmin(dim=-1)  # (B, M)

        # Gather best proposal ctrl pts: (B, M, K_ctrl, 2)
        idx_exp = best_idx.unsqueeze(-1).unsqueeze(-1).expand(B, M, 1, K_ctrl, 2)
        best_pred_ctrl = ctrl_pts_pred.gather(dim=2, index=idx_exp).squeeze(2)  # (B, M, K_ctrl, 2)

        # Gather best proposal samples
        idx_samp_exp = best_idx.unsqueeze(-1).unsqueeze(-1).expand(B, M, 1, self.N, 2)
        best_pred_samp = pred_sampled.gather(dim=2, index=idx_samp_exp).squeeze(2)  # (B, M, N, 2)

        # ─── Apply only on active (present) classes ───────────────────────────
        loss_ctrl = torch.tensor(0.0, device=device)
        loss_samp = torch.tensor(0.0, device=device)
        n_active = active_mask.sum().item()

        if n_active > 0:
            loss_ctrl = F.smooth_l1_loss(
                best_pred_ctrl[active_mask],
                ctrl_pts_gt[active_mask],
                beta=0.02,
            )
            loss_samp = F.l1_loss(
                best_pred_samp[active_mask],
                gt_sampled[active_mask],
            )

        return loss_ctrl + loss_samp


# ─────────────────────────────────────────────────────────────────────────────
# Full SurgicalCurveFormer Loss
# ─────────────────────────────────────────────────────────────────────────────

class SurgicalCurveFormerLoss(nn.Module):
    """
    Full EXP_11 Loss Suite.

    Weighted multi-task loss combining BCRNet annealing + EXP_10 components.

    L = lambda_d * (lambda_s * L_s + lambda_ind * L_ind)
      + (1 - lambda_d) * Σ_h [lambda_cs * L_cs^h + lambda_crv * L_crv^h]
      + lambda_exist * L_exist
      + lambda_dice * L_dice_soft
    """

    def __init__(
        self,
        num_hcr_stages: int = 3,
        lambda_s: float = 10.0,
        lambda_ind: float = 1.0,
        lambda_cs: float = 1.0,
        lambda_crv: float = 1.0,
        lambda_exist: float = 1.5,
        lambda_dice: float = 5.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        anneal_center: float = 10.0,
        anneal_slope: float = 2.0,
        raster_render_size: int = 128,
        raster_num_samples: int = 64,
        raster_sigma_px: float = 2.0,
        target_size: int = 512,
    ):
        super().__init__()
        self.num_stages = num_hcr_stages
        self.lambda_s = lambda_s
        self.lambda_ind = lambda_ind
        self.lambda_cs = lambda_cs
        self.lambda_crv = lambda_crv
        self.lambda_exist = lambda_exist
        self.lambda_dice = lambda_dice
        self.anneal_center = anneal_center
        self.anneal_slope = anneal_slope

        # Sub-modules
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.soft_dice = SoftDiceLoss()
        self.induction_loss = ACPIInductionLoss(pos_weight=10.0)
        self.curve_losses = nn.ModuleList([HCRCurveLoss() for _ in range(num_hcr_stages)])
        self.rasterizer = SoftRasterizer(
            render_size=raster_render_size,
            num_samples=raster_num_samples,
            sigma_px=raster_sigma_px,
            target_size=target_size,
        )

    def forward(self, pred_dict: dict, target_dict: dict, epoch: int) -> dict:
        """
        Compute total loss with sigmoid annealing.

        Args:
            pred_dict: {
              'seg_logits_list':  [(B, M, H, W)] × 4    CNN decoder multi-level
              'score_logits':      (B, M, H_f, W_f)      ACPI confidence map
              'proposals':         (B, M, K_top, 6, 2)   ACPI top-K proposals
              'all_conf_logits':  [(B, M, K)] × num_stages  HCR confidence per stage
              'final_ctrl_pts':   (B, M, K_top, 6, 2)    HCR output
              'exist_logits':     (B, M)                  existence gate
              'exist_probs':      (B, M)                  sigmoid of exist_logits
            }
            target_dict: {
              'target_masks':      (B, M, H, W)           GT binary masks (512×512)
              'acpi_target_score': (B, M, H_f, W_f)       GT midpoint indicator
              'target_ctrl_pts':   (B, M, 6, 2)           GT Bézier ctrl pts
              'active_mask':       (B, M) bool             class is present
              'target_render_masks': (B, M, R, R)          GT masks at render size
            }
            epoch: current training epoch (0-indexed).

        Returns:
            dict with all individual loss components and total loss.
        """
        device = pred_dict["exist_logits"].device
        lambda_d = compute_lambda_d(epoch, center=self.anneal_center, slope=self.anneal_slope)
        lambda_d = float(lambda_d)

        losses = {}

        # ─── L_s: Multi-level CNN decoder Dice (dense supervision) ─────────────
        seg_list = pred_dict.get("seg_logits_list", [])
        target_masks = target_dict["target_masks"].to(device)  # (B, M, H, W)

        L_s = torch.tensor(0.0, device=device)
        if seg_list:
            for seg_logits in seg_list:
                seg_prob = torch.sigmoid(seg_logits)
                # Resize GT to match seg_logits spatial size
                H_seg = seg_logits.shape[-2]
                W_seg = seg_logits.shape[-1]
                if target_masks.shape[-2] != H_seg or target_masks.shape[-1] != W_seg:
                    gt_resized = F.interpolate(
                        target_masks, size=(H_seg, W_seg), mode="nearest"
                    )
                else:
                    gt_resized = target_masks
                L_s = L_s + self.soft_dice(seg_prob, gt_resized)
            L_s = L_s / len(seg_list)
        losses["L_s"] = L_s

        # ─── L_ind: ACPI Induction BCE ──────────────────────────────────────────
        score_logits = pred_dict["score_logits"]    # (B, M, H_f, W_f)
        acpi_target = target_dict["acpi_target_score"].to(device)
        L_ind = self.induction_loss(score_logits, acpi_target)
        losses["L_ind"] = L_ind

        # ─── L_cs: HCR Confidence Focal Loss (per stage) ───────────────────────
        # HCR conf logits: (B, M, K_top) — best-proposal score should be 1, others 0
        active_mask = target_dict["active_mask"].to(device)  # (B, M) bool
        target_ctrl_pts = target_dict["target_ctrl_pts"].to(device).float()  # (B, M, 6, 2)

        L_cs_total = torch.tensor(0.0, device=device)
        L_crv_total = torch.tensor(0.0, device=device)

        for h, conf_logits_h in enumerate(pred_dict.get("all_conf_logits", [])):
            # conf_logits_h: (B, M, K_top)
            # Build pseudo-targets: 1 for the best-matched proposal, 0 for others
            B_h, M_h, K_top_h = conf_logits_h.shape
            proposals_h = pred_dict["proposals"]  # (B, M, K_top, 6, 2)
            ctrl_flat_pred = proposals_h.view(B_h * M_h * K_top_h, 6, 2)
            samp_flat_pred = evaluate_bezier_torch(ctrl_flat_pred, num_samples=25)
            pred_samp = samp_flat_pred.view(B_h, M_h, K_top_h, 25, 2)

            ctrl_flat_gt = target_ctrl_pts.view(B_h * M_h, 6, 2)
            samp_flat_gt = evaluate_bezier_torch(ctrl_flat_gt, num_samples=25)
            gt_samp = samp_flat_gt.view(B_h, M_h, 25, 2)

            diff = pred_samp - gt_samp.unsqueeze(2)
            dist = diff.pow(2).sum(dim=-1).mean(dim=-1)  # (B, M, K_top)
            best_idx = dist.argmin(dim=-1)  # (B, M)

            conf_targets = torch.zeros_like(conf_logits_h)
            for b_ in range(B_h):
                for m_ in range(M_h):
                    if active_mask[b_, m_]:
                        conf_targets[b_, m_, best_idx[b_, m_]] = 1.0

            L_cs_total = L_cs_total + self.focal(conf_logits_h, conf_targets)

            # Curve geometry loss at this stage
            L_crv_h = self.curve_losses[h](proposals_h, target_ctrl_pts, active_mask)
            L_crv_total = L_crv_total + L_crv_h

        if len(pred_dict.get("all_conf_logits", [])) > 0:
            n_stages = len(pred_dict["all_conf_logits"])
            L_cs_total = L_cs_total / n_stages
            L_crv_total = L_crv_total / n_stages

        losses["L_cs"] = L_cs_total
        losses["L_crv"] = L_crv_total

        # ─── L_exist: Existence BCE (EXP_10) ───────────────────────────────────
        exist_logits = pred_dict["exist_logits"]  # (B, M)
        target_exists = active_mask.float()
        L_exist = F.binary_cross_entropy_with_logits(exist_logits, target_exists)
        losses["L_exist"] = L_exist

        # ─── L_dice: Differentiable Soft Rasterizer Dice (EXP_10) ──────────────
        final_ctrl_pts = pred_dict["final_ctrl_pts"]  # (B, M, K_top, 6, 2)
        exist_probs = pred_dict["exist_probs"]         # (B, M)
        soft_masks = self.rasterizer(final_ctrl_pts, exist_probs)  # (B, M, R, R)

        target_render = target_dict.get("target_render_masks", None)
        if target_render is not None:
            target_render = target_render.to(device).float()  # (B, M, R, R)
            L_dice = self.soft_dice(soft_masks, target_render)
        else:
            # Fallback: resize full-res GT to render size
            R = soft_masks.shape[-1]
            target_resized = F.interpolate(target_masks, size=(R, R), mode="nearest")
            L_dice = self.soft_dice(soft_masks, target_resized)
        losses["L_dice"] = L_dice

        # ─── Total with BCRNet Annealing ────────────────────────────────────────
        dense_component = lambda_d * (
            self.lambda_s   * L_s +
            self.lambda_ind * L_ind
        )
        curve_component = (1.0 - lambda_d) * (
            self.lambda_cs  * L_cs_total +
            self.lambda_crv * L_crv_total
        )
        exist_dice_component = (
            self.lambda_exist * L_exist +
            self.lambda_dice  * L_dice
        )

        total = dense_component + curve_component + exist_dice_component

        losses["L_dense"] = dense_component
        losses["L_curve"] = curve_component
        losses["L_total"] = total
        losses["lambda_d"] = torch.tensor(lambda_d)

        return losses
