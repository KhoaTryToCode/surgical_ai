import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianMatcher3D(nn.Module):
    """
    Bipartite Hungarian Matcher operating at the Instance Level.
    Pairs N predicted query instances to ground-truth landmark curves by minimizing
    a combined cost of Classification Focal Cost + Bidirectional 3D Position Cost.
    """
    def __init__(self, cost_cls: float = 2.0, cost_pos: float = 5.0):
        super().__init__()
        self.cost_cls = cost_cls
        self.cost_pos = cost_pos

    @torch.no_grad()
    def forward(self, pred_cls: torch.Tensor, pred_polylines: torch.Tensor, 
                target_cls: torch.Tensor, target_polylines: torch.Tensor, valid_mask: torch.Tensor):
        """
        pred_cls: (B, N, num_classes+1)
        pred_polylines: (B, N, K, 3)
        target_cls: (B, max_gt)
        target_polylines: (B, max_gt, K, 3)
        valid_mask: (B, max_gt) boolean tensor indicating active GT curves
        Returns:
          indices: List of tuples (pred_idx, gt_idx) for each batch item
        """
        B, N, K, _ = pred_polylines.shape
        indices = []

        for b in range(B):
            valid = valid_mask[b]
            gt_c = target_cls[b][valid] # (num_valid,)
            gt_p = target_polylines[b][valid] # (num_valid, K, 3)
            num_valid = len(gt_c)

            if num_valid == 0:
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue

            # 1. Classification Cost (Softmax probability of GT class)
            prob = F.softmax(pred_cls[b], dim=-1) # (N, num_classes+1)
            cost_cls = -prob[:, gt_c] # (N, num_valid)

            # 2. Bidirectional 3D Position Cost
            p_pred = pred_polylines[b] # (N, K, 3)
            # Forward ordering (1..K)
            diff_fwd = p_pred.unsqueeze(1) - gt_p.unsqueeze(0) # (N, num_valid, K, 3)
            cost_fwd = torch.mean(torch.abs(diff_fwd), dim=(-2, -1)) # (N, num_valid)

            # Reverse ordering (K..1)
            gt_p_rev = torch.flip(gt_p, dims=[1])
            diff_rev = p_pred.unsqueeze(1) - gt_p_rev.unsqueeze(0)
            cost_rev = torch.mean(torch.abs(diff_rev), dim=(-2, -1))

            cost_pos = torch.minimum(cost_fwd, cost_rev) # (N, num_valid)

            # Total Matching Cost
            total_cost = self.cost_cls * cost_cls + self.cost_pos * cost_pos
            total_cost_np = total_cost.cpu().numpy()

            # Hungarian Bipartite Assignment
            pred_idx, gt_idx = linear_sum_assignment(total_cost_np)
            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(gt_idx, dtype=torch.long)
            ))

        return indices

class Vector3DLossSuite(nn.Module):
    """
    Complete Loss Suite for EXP_05 3D Vector Space Transformer.
    Computes:
      - Focal Classification Loss
      - Bidirectional Smooth L1 Position Loss (L_pos)
      - Cosine Tangent Edge Alignment Loss (L_tan)
      - 1D Discrete Laplacian Curvature Loss (L_curv)
      - Auxiliary 2D Mask BCE + Dice Loss (L_mask)
    """
    def __init__(self, lambda_cls: float = 2.0, lambda_pos: float = 5.0, 
                 lambda_tan: float = 2.0, lambda_curv: float = 1.0, lambda_mask: float = 5.0):
        super().__init__()
        self.matcher = HungarianMatcher3D(cost_cls=lambda_cls, cost_pos=lambda_pos)
        self.lambda_cls = lambda_cls
        self.lambda_pos = lambda_pos
        self.lambda_tan = lambda_tan
        self.lambda_curv = lambda_curv
        self.lambda_mask = lambda_mask

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.sum() + target.sum()
        return 1.0 - (2.0 * intersection + eps) / (union + eps)

    def forward(self, outputs_cls: list, outputs_polylines: list, outputs_masks: list, targets: dict):
        """
        Calculates Deep Supervision Loss across all decoder layers L.
        """
        target_cls = targets["target_classes"]         # (B, N)
        target_polylines = targets["target_polylines"] # (B, N, K, 3)
        target_masks = targets["target_masks"]         # (B, N, H, W)
        valid_mask = targets["valid_mask"]             # (B, N)

        num_layers = len(outputs_cls)
        total_loss = 0.0
        loss_dict = {"l_cls": 0.0, "l_pos": 0.0, "l_tan": 0.0, "l_curv": 0.0, "l_mask": 0.0}

        for l in range(num_layers):
            pred_cls_l = outputs_cls[l]             # (B, N, num_classes+1)
            pred_poly_l = outputs_polylines[l]       # (B, N, K, 3)
            pred_mask_l = outputs_masks[l]           # (B, N, H, W)

            # 1. Run Hungarian Matcher for layer l
            matched_indices = self.matcher(pred_cls_l, pred_poly_l, target_cls, target_polylines, valid_mask)

            # Compute Classification Loss
            l_cls_layer = F.cross_entropy(pred_cls_l.view(-1, pred_cls_l.size(-1)), target_cls.view(-1), reduction='mean')
            
            l_pos_layer = 0.0
            l_tan_layer = 0.0
            l_curv_layer = 0.0
            l_mask_layer = 0.0
            num_matched_total = 0

            B, N, K, _ = pred_poly_l.shape

            for b in range(B):
                pred_idx, gt_idx = matched_indices[b]
                if len(pred_idx) == 0:
                    continue

                num_matched_total += len(pred_idx)
                p_matched = pred_poly_l[b, pred_idx] # (M, K, 3)
                gt_matched = target_polylines[b, gt_idx] # (M, K, 3)

                # 2. Bidirectional Smooth L1 Position Loss
                fwd_diff = F.smooth_l1_loss(p_matched, gt_matched, reduction='none').mean(dim=(-2, -1)) # (M,)
                gt_rev = torch.flip(gt_matched, dims=[1])
                rev_diff = F.smooth_l1_loss(p_matched, gt_rev, reduction='none').mean(dim=(-2, -1)) # (M,)

                best_dir_is_rev = rev_diff < fwd_diff
                gt_aligned = gt_matched.clone()
                gt_aligned[best_dir_is_rev] = gt_rev[best_dir_is_rev]

                l_pos_layer += torch.minimum(fwd_diff, rev_diff).sum()

                # 3. Cosine Tangent Alignment Loss
                p_tangents = p_matched[:, 1:] - p_matched[:, :-1] # (M, K-1, 3)
                gt_tangents = gt_aligned[:, 1:] - gt_aligned[:, :-1] # (M, K-1, 3)
                
                cos_sim = F.cosine_similarity(p_tangents, gt_tangents, dim=-1) # (M, K-1)
                l_tan_layer += (1.0 - cos_sim).mean(dim=-1).sum()

                # 4. 1D Discrete Laplacian Curvature Loss
                # Discrete 2nd derivative: p_{j+1} - 2*p_j + p_{j-1}
                p_laplacian = p_matched[:, 2:] - 2.0 * p_matched[:, 1:-1] + p_matched[:, :-2]
                gt_laplacian = gt_aligned[:, 2:] - 2.0 * gt_aligned[:, 1:-1] + gt_aligned[:, :-2]
                l_curv_layer += F.mse_loss(p_laplacian, gt_laplacian, reduction='none').mean(dim=(-2, -1)).sum()

                # 5. Auxiliary 2D Mask Loss (BCE + Dice)
                m_matched = pred_mask_l[b, pred_idx] # (M, H, W)
                gt_m_matched = target_masks[b, gt_idx] # (M, H, W)
                
                bce = F.binary_cross_entropy_with_logits(m_matched, gt_m_matched)
                dice = self._dice_loss(m_matched, gt_m_matched)
                l_mask_layer += (bce + dice)

            norm_factor = max(num_matched_total, 1)
            l_pos_layer = l_pos_layer / norm_factor
            l_tan_layer = l_tan_layer / norm_factor
            l_curv_layer = l_curv_layer / norm_factor
            l_mask_layer = l_mask_layer / norm_factor

            # Layer total loss
            layer_loss = (self.lambda_cls * l_cls_layer + 
                          self.lambda_pos * l_pos_layer + 
                          self.lambda_tan * l_tan_layer + 
                          self.lambda_curv * l_curv_layer + 
                          self.lambda_mask * l_mask_layer)

            total_loss += layer_loss
            loss_dict["l_cls"] += l_cls_layer.item()
            loss_dict["l_pos"] += l_pos_layer.item() if isinstance(l_pos_layer, torch.Tensor) else l_pos_layer
            loss_dict["l_tan"] += l_tan_layer.item() if isinstance(l_tan_layer, torch.Tensor) else l_tan_layer
            loss_dict["l_curv"] += l_curv_layer.item() if isinstance(l_curv_layer, torch.Tensor) else l_curv_layer
            loss_dict["l_mask"] += l_mask_layer.item() if isinstance(l_mask_layer, torch.Tensor) else l_mask_layer

        # Average loss across decoder layers
        total_loss = total_loss / num_layers
        for k in loss_dict:
            loss_dict[k] /= num_layers

        return total_loss, loss_dict
