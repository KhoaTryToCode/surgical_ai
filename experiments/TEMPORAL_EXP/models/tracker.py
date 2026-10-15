"""
MinVIS Bipartite Hungarian Query Tracker (Experiment 17).
Exact implementation of NVIDIA MinVIS (NeurIPS 2022) online bipartite matching:
Tracks instance queries across sequential video frames by computing the cosine distance
between per-frame query embeddings and solving Hungarian assignment.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


class MinVISTracker:
    """
    MinVIS Online Temporal Query Tracker.
    Operates strictly in post-processing without passing hidden activations across frames.
    """
    def __init__(self, num_classes: int = 3, topk: int = 10):
        self.num_classes = num_classes
        self.topk = topk

    @staticmethod
    def match_from_embds(tgt_embds: torch.Tensor, cur_embds: torch.Tensor) -> np.ndarray:
        """
        Bipartite matching from query embeddings via cosine similarity.
        Exact implementation from MinVIS (minvis/video_maskformer_model.py).

        Args:
            tgt_embds: (Q, C) previous/target query embeddings
            cur_embds: (Q, C) current query embeddings

        Returns:
            indices: permutation array of length Q aligning cur_embds to tgt_embds
        """
        cur_norm = cur_embds / (cur_embds.norm(dim=1, keepdim=True) + 1e-6)
        tgt_norm = tgt_embds / (tgt_embds.norm(dim=1, keepdim=True) + 1e-6)

        cos_sim = torch.mm(cur_norm, tgt_norm.transpose(0, 1))  # (Q_cur, Q_tgt)
        cost_embd = 1.0 - cos_sim

        cost_np = cost_embd.detach().cpu().numpy()
        # Hungarian assignment: rows are target, cols are current
        _, col_ind = linear_sum_assignment(cost_np.T)
        return col_ind

    def track_sequence(
        self,
        query_embds_seq: torch.Tensor,
        class_logits_seq: torch.Tensor,
        mask_probs_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tracks queries sequentially across a video sequence of length T.

        Args:
            query_embds_seq: (T, Q, C) query embeddings per frame
            class_logits_seq: (T, Q, K+1) class logits per frame
            mask_probs_seq: (T, Q, H, W) sigmoid mask probabilities per frame

        Returns:
            tracked_sem_seg: (T, num_classes, H, W) tracked semantic segmentation
            tracked_indices_seq: list of permutations applied at each timestep
        """
        t, q, c = query_embds_seq.shape
        tracked_logits = [class_logits_seq[0]]
        tracked_masks = [mask_probs_seq[0]]
        tracked_embds = [query_embds_seq[0]]
        permutations = [np.arange(q)]

        for i in range(1, t):
            # Match current frame i to previous tracked frame i-1
            indices = self.match_from_embds(tracked_embds[-1], query_embds_seq[i])
            idx_tensor = torch.from_numpy(indices).to(query_embds_seq.device)

            tracked_logits.append(class_logits_seq[i][idx_tensor, :])
            tracked_masks.append(mask_probs_seq[i][idx_tensor, :, :])
            tracked_embds.append(query_embds_seq[i][idx_tensor, :])
            permutations.append(indices)

        tracked_logits_tensor = torch.stack(tracked_logits, dim=0)  # (T, Q, K+1)
        tracked_masks_tensor = torch.stack(tracked_masks, dim=0)    # (T, Q, H, W)

        # Compute per-frame semantic segmentation from tracked queries
        class_probs = tracked_logits_tensor.softmax(dim=-1)[..., :self.num_classes]
        tracked_sem_seg = torch.einsum('tqc, tqhw -> tchw', class_probs, tracked_masks_tensor)

        return tracked_sem_seg, permutations
