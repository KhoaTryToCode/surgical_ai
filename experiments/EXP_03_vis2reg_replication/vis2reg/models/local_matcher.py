"""
Local Sparse Matcher with Geometry-based Candidate Filtering

Key idea:
- For each source point, only consider K nearest target points (K=64-128)
- Compute softmax only on this local subset (K-way instead of 6000-way)
- Generate sparse soft_match for weighted SVD

This addresses the fundamental issue that dense 6000-way softmax is intractable.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def build_local_candidates_knn(
    pre_coords: torch.Tensor,  # (B, N_pre, 3) in camera frame
    tgt_coords: torch.Tensor,  # (B, N_tgt, 3) in camera frame
    k_local: int = 64,         # Number of nearest neighbors per source
    radius_mm: float = 15.0,   # Maximum distance in mm (camera frame units are meters)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build local candidate sets using K-nearest neighbors + radius constraint.

    Args:
        pre_coords: Source points in camera frame (B, N_pre, 3)
        tgt_coords: Target points in camera frame (B, N_tgt, 3)
        k_local: Maximum number of candidates per source
        radius_mm: Maximum distance threshold in mm

    Returns:
        candidate_indices: (B, N_pre, k_local) - indices of candidate targets for each source
        candidate_mask: (B, N_pre, k_local) - True if candidate is valid (within radius)
    """
    B, N_pre, _ = pre_coords.shape
    N_tgt = tgt_coords.shape[1]

    # Convert radius from mm to meters (camera frame units)
    radius_m = radius_mm / 1000.0

    # Compute pairwise distances: (B, N_pre, N_tgt)
    # dist[b, i, j] = ||pre[b,i] - tgt[b,j]||
    dist_matrix = torch.cdist(pre_coords, tgt_coords, p=2)  # (B, N_pre, N_tgt)

    # For each source, get k_local nearest targets
    k_actual = min(k_local, N_tgt)
    distances, indices = torch.topk(
        dist_matrix,
        k=k_actual,
        dim=-1,           # along target dimension
        largest=False,    # get smallest distances
        sorted=True
    )  # distances: (B, N_pre, k_actual), indices: (B, N_pre, k_actual)

    # Pad if k_actual < k_local
    if k_actual < k_local:
        pad_size = k_local - k_actual
        # Pad with last index (will be masked out)
        indices = F.pad(indices, (0, pad_size), mode='replicate')
        # Pad distances with large values
        distances = F.pad(distances, (0, pad_size), value=float('inf'))

    # Create mask: valid if within radius
    candidate_mask = distances < radius_m  # (B, N_pre, k_local)

    return indices, candidate_mask


def compute_local_softmax(
    features_pre: torch.Tensor,     # (B, N_pre, D)
    features_tgt: torch.Tensor,     # (B, N_tgt, D)
    candidate_indices: torch.Tensor, # (B, N_pre, k_local)
    candidate_mask: torch.Tensor,    # (B, N_pre, k_local)
    temperature: float = 0.1,
    logit_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute local softmax matching scores.

    Args:
        features_pre: Source features (B, N_pre, D)
        features_tgt: Target features (B, N_tgt, D)
        candidate_indices: Candidate target indices (B, N_pre, k_local)
        candidate_mask: Valid candidate mask (B, N_pre, k_local)
        temperature: Softmax temperature

    Returns:
        local_logits: (B, N_pre, k_local) - similarity logits
        local_probs: (B, N_pre, k_local) - softmax probabilities (row-normalized)
    """
    B, _, _ = features_pre.shape

    # Gather candidate features: (B, N_pre, k_local, D)
    # For each source i, gather features of its k_local candidates
    batch_indices = torch.arange(B, device=features_tgt.device)[:, None, None]
    candidate_feats = features_tgt[
        batch_indices,
        candidate_indices
    ]  # (B, N_pre, k_local, D)

    if not torch.is_tensor(logit_scale):
        logit_scale = torch.tensor(logit_scale, device=features_pre.device)

    # Compute similarity: dot product between source and each candidate
    features_pre_expanded = features_pre.unsqueeze(2)  # (B, N_pre, 1, D)
    local_logits = (features_pre_expanded * candidate_feats).sum(dim=-1)  # (B, N_pre, k_local)
    local_logits = local_logits * logit_scale
    local_logits = local_logits / temperature

    # Apply mask: set invalid candidates to -inf
    local_logits = local_logits.masked_fill(~candidate_mask, float('-inf'))

    # Compute softmax over candidates
    local_probs = F.softmax(local_logits, dim=-1)  # (B, N_pre, k_local)

    # Handle all-invalid rows (where all candidates are masked)
    # Set uniform distribution over valid candidates, or zeros if none
    row_has_valid = candidate_mask.any(dim=-1, keepdim=True)  # (B, N_pre, 1)
    num_valid = candidate_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, N_pre, 1)
    uniform_probs = candidate_mask.float() / num_valid  # (B, N_pre, k_local)

    local_probs = torch.where(row_has_valid, local_probs, uniform_probs)

    return local_logits, local_probs


def build_sparse_soft_match(
    local_probs: torch.Tensor,       # (B, N_pre, k_local)
    candidate_indices: torch.Tensor,  # (B, N_pre, k_local)
    candidate_mask: torch.Tensor,     # (B, N_pre, k_local)
    N_tgt: int,
) -> torch.Tensor:
    """
    Build sparse soft match matrix from local probabilities.

    Args:
        local_probs: Local softmax probabilities (B, N_pre, k_local)
        candidate_indices: Candidate target indices (B, N_pre, k_local)
        candidate_mask: Valid candidate mask (B, N_pre, k_local)
        N_tgt: Total number of target points

    Returns:
        soft_match: (B, N_pre, N_tgt) - sparse soft matching matrix
    """
    B, N_pre, k_local = local_probs.shape

    # Initialize sparse matrix
    soft_match = torch.zeros(
        B, N_pre, N_tgt,
        dtype=local_probs.dtype,
        device=local_probs.device
    )

    # Scatter local probabilities to full matrix
    # For each (b, i, k), set soft_match[b, i, candidate_indices[b,i,k]] = local_probs[b,i,k]
    batch_indices = torch.arange(B, device=soft_match.device)[:, None, None].expand(B, N_pre, k_local)
    src_indices = torch.arange(N_pre, device=soft_match.device)[None, :, None].expand(B, N_pre, k_local)

    # Only scatter valid candidates
    valid_mask = candidate_mask
    soft_match[
        batch_indices[valid_mask],
        src_indices[valid_mask],
        candidate_indices[valid_mask]
    ] = local_probs[valid_mask]

    return soft_match


def local_matcher_forward(
    features_pre: torch.Tensor,
    features_tgt: torch.Tensor,
    pre_coords: torch.Tensor,
    tgt_coords: torch.Tensor,
    k_local: int = 64,
    radius_mm: float = 15.0,
    temperature: float = 0.1,
    logit_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Complete forward pass of local matcher.

    Returns:
        soft_match: (B, N_pre, N_tgt) - sparse soft matching matrix
        local_logits: (B, N_pre, k_local) - local similarity logits
        local_probs: (B, N_pre, k_local) - local softmax probabilities
        candidate_indices: (B, N_pre, k_local) - candidate target indices
        candidate_mask: (B, N_pre, k_local) - valid candidate mask
    """
    # Step 1: Build local candidate sets using geometry
    candidate_indices, candidate_mask = build_local_candidates_knn(
        pre_coords, tgt_coords,
        k_local=k_local,
        radius_mm=radius_mm
    )

    # Step 2: Compute local softmax
    local_logits, local_probs = compute_local_softmax(
        features_pre, features_tgt,
        candidate_indices, candidate_mask,
        temperature=temperature,
        logit_scale=logit_scale,
    )

    # Step 3: Build sparse soft match matrix
    N_tgt = tgt_coords.shape[1]
    soft_match = build_sparse_soft_match(
        local_probs, candidate_indices, candidate_mask, N_tgt
    )

    return soft_match, local_logits, local_probs, candidate_indices, candidate_mask
