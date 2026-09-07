import torch
import torch.nn as nn
import torch.nn.functional as F
from .bezier_utils import sample_cubic_bezier_batch


class MacroFocalLoss(nn.Module):
    """
    Focal Loss for macro-patch classification (0 = background, 1..C = landmarks).
    """
    def __init__(self, gamma: float = 2.0, class_weights: tuple = (0.20, 1.0, 1.0, 2.5, 3.0)):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B*N, num_classes + 1)
            targets: (B*N,) long tensor
        """
        ce_loss = F.cross_entropy(logits, targets, weight=self.weights.to(logits.device), reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class MacroPatchLoss(nn.Module):
    """
    Multi-Task Loss for EXP_10 Macro-Patch Geometric ViT.
    
    Components:
    1. L_cls:    Focal classification loss on (B, 64, C+1)
    2. L_ctrl:   Smooth L1 loss on 4 Bézier control points (active patches)
    3. L_sample: L1 loss on 10 sampled points along the curve (active patches)
    4. L_tan:    Tangent cosine alignment loss at endpoints
    5. L_cont:   Inter-macro endpoint continuity loss on adjacent active patches
    """
    def __init__(
        self,
        lambda_cls: float = 2.0,
        lambda_ctrl: float = 5.0,
        lambda_sample: float = 5.0,
        lambda_tan: float = 1.0,
        lambda_cont: float = 1.5,
        lambda_tan_cont: float = 1.0,
        macro_grid_size: int = 8,
        macro_patch_size: int = 64
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_ctrl = lambda_ctrl
        self.lambda_sample = lambda_sample
        self.lambda_tan = lambda_tan
        self.lambda_cont = lambda_cont
        self.lambda_tan_cont = lambda_tan_cont
        self.macro_grid_size = macro_grid_size
        self.macro_patch_size = macro_patch_size
        
        self.focal_cls = MacroFocalLoss(gamma=2.0)

    def forward(self, pred_dict: dict, target_dict: dict) -> dict:
        device = pred_dict["flat_logits"].device
        
        flat_logits = pred_dict["flat_logits"]     # (B, 64, C+1)
        flat_beziers = pred_dict["flat_beziers"]   # (B, 64, 4, 2)
        macro_beziers = pred_dict["macro_beziers"] # (B, 8, 8, 4, 2)
        
        target_classes = target_dict["target_classes"].to(device)  # (B, 8, 8)
        target_beziers = target_dict["target_beziers"].to(device)  # (B, 8, 8, 4, 2)
        active_mask = target_dict["active_mask"].to(device)        # (B, 8, 8) boolean
        
        B = flat_logits.shape[0]
        
        # 1. Macro-Patch Classification Loss
        flat_targets = target_classes.view(B * 64)
        loss_cls = self.focal_cls(flat_logits.view(B * 64, -1), flat_targets)
        
        # 2. Regression Losses on Active Macro-Patches
        flat_active = active_mask.view(B * 64)
        num_active = flat_active.sum().item()
        
        if num_active > 0:
            pred_active_ctrl = flat_beziers.view(B * 64, 4, 2)[flat_active]
            gt_active_ctrl = target_beziers.view(B * 64, 4, 2)[flat_active]
            
            # Smooth L1 on control points
            loss_ctrl = F.smooth_l1_loss(pred_active_ctrl, gt_active_ctrl, beta=0.02)
            
            # Sampled points along the cubic curve
            pred_sampled = sample_cubic_bezier_batch(pred_active_ctrl, num_samples=10)
            gt_sampled = sample_cubic_bezier_batch(gt_active_ctrl, num_samples=10)
            loss_sample = F.l1_loss(pred_sampled, gt_sampled)
            
            # Tangent direction alignment inside patch
            pred_tan_start = pred_active_ctrl[:, 1] - pred_active_ctrl[:, 0]
            gt_tan_start = gt_active_ctrl[:, 1] - gt_active_ctrl[:, 0]
            pred_tan_end = pred_active_ctrl[:, 3] - pred_active_ctrl[:, 2]
            gt_tan_end = gt_active_ctrl[:, 3] - gt_active_ctrl[:, 2]
            
            cos_start = F.cosine_similarity(pred_tan_start, gt_tan_start + 1e-6, dim=-1)
            cos_end = F.cosine_similarity(pred_tan_end, gt_tan_end + 1e-6, dim=-1)
            loss_tan = (1.0 - cos_start).mean() + (1.0 - cos_end).mean()
        else:
            loss_ctrl = torch.tensor(0.0, device=device)
            loss_sample = torch.tensor(0.0, device=device)
            loss_tan = torch.tensor(0.0, device=device)
            
        # 3. Inter-Macro Endpoint (C0) and Tangent (C1) Continuity Loss
        loss_cont = torch.tensor(0.0, device=device)
        loss_tan_cont = torch.tensor(0.0, device=device)
        cont_pairs = 0
        G = self.macro_grid_size
        P = self.macro_patch_size
        
        # Check horizontal adjacent neighbors: (r, c) and (r, c+1)
        for r in range(G):
            for c in range(G - 1):
                mask_pair = active_mask[:, r, c] & active_mask[:, r, c + 1] & (target_classes[:, r, c] == target_classes[:, r, c + 1])
                if mask_pair.sum() > 0:
                    # C0 Continuity: endpoint distance
                    exit_pt = c * P + macro_beziers[mask_pair, r, c, 3] * P
                    entry_pt = (c + 1) * P + macro_beziers[mask_pair, r, c + 1, 0] * P
                    loss_cont = loss_cont + F.l1_loss(exit_pt / P, entry_pt / P)
                    
                    # C1 Continuity: tangent vector alignment
                    vec_exit = macro_beziers[mask_pair, r, c, 3] - macro_beziers[mask_pair, r, c, 2]
                    vec_entry = macro_beziers[mask_pair, r, c + 1, 1] - macro_beziers[mask_pair, r, c + 1, 0]
                    cos_sim = F.cosine_similarity(vec_exit, vec_entry + 1e-6, dim=-1)
                    loss_tan_cont = loss_tan_cont + (1.0 - cos_sim).mean()
                    cont_pairs += 1
                    
        # Check vertical adjacent neighbors: (r, c) and (r+1, c)
        for r in range(G - 1):
            for c in range(G):
                mask_pair = active_mask[:, r, c] & active_mask[:, r + 1, c] & (target_classes[:, r, c] == target_classes[:, r + 1, c])
                if mask_pair.sum() > 0:
                    exit_pt = r * P + macro_beziers[mask_pair, r, c, 3] * P
                    entry_pt = (r + 1) * P + macro_beziers[mask_pair, r + 1, c, 0] * P
                    loss_cont = loss_cont + F.l1_loss(exit_pt / P, entry_pt / P)
                    
                    vec_exit = macro_beziers[mask_pair, r, c, 3] - macro_beziers[mask_pair, r, c, 2]
                    vec_entry = macro_beziers[mask_pair, r + 1, c, 1] - macro_beziers[mask_pair, r + 1, c, 0]
                    cos_sim = F.cosine_similarity(vec_exit, vec_entry + 1e-6, dim=-1)
                    loss_tan_cont = loss_tan_cont + (1.0 - cos_sim).mean()
                    cont_pairs += 1
                    
        if cont_pairs > 0:
            loss_cont = loss_cont / float(cont_pairs)
            loss_tan_cont = loss_tan_cont / float(cont_pairs)
            
        # Total Weighted Loss
        total_loss = (
            self.lambda_cls * loss_cls +
            self.lambda_ctrl * loss_ctrl +
            self.lambda_sample * loss_sample +
            self.lambda_tan * loss_tan +
            self.lambda_cont * loss_cont +
            self.lambda_tan_cont * loss_tan_cont
        )
        
        return {
            "loss": total_loss,
            "loss_cls": loss_cls,
            "loss_ctrl": loss_ctrl,
            "loss_sample": loss_sample,
            "loss_tan": loss_tan,
            "loss_cont": loss_cont,
            "loss_tan_cont": loss_tan_cont
        }
