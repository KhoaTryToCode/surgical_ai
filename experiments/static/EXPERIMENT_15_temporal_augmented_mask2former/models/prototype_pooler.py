import torch
import torch.nn as nn
import torch.nn.functional as F


class LandmarkPrototypePooler(nn.Module):
    """
    Extracts landmark prototype vectors by mask-pooling feature maps with binary landmark masks.
    Handles absent landmarks via learnable empty-class tokens.
    Adds learned landmark ID embeddings and temporal position embeddings so target queries
    can precisely match corresponding landmark anchors across time.
    """
    def __init__(self, num_classes=3, hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Learnable empty tokens for absent landmarks: (num_classes, hidden_dim)
        self.empty_tokens = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)

        # Landmark Class ID embeddings: (num_classes, hidden_dim)
        # Index 0: Ridge, Index 1: Silhouette, Index 2: Ligament
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)

        # Temporal frame position embedding: (2, hidden_dim)
        # Distinguishes t-2 (index 0) from t-1 (index 1)
        self.time_embeddings = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)

        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def pool_single_frame(self, feature_map, mask, time_idx=0):
        """
        Pools features from a single context frame.
        
        Args:
            feature_map: (B, C, H_f, W_f)
            mask: (B, num_classes, H_m, W_m)
            time_idx: 0 for t-2, 1 for t-1
            
        Returns:
            prototypes: (B, num_classes, hidden_dim)
        """
        B, C, H_f, W_f = feature_map.shape
        _, K, H_m, W_m = mask.shape

        # Match spatial resolution
        if (H_m, W_m) != (H_f, W_f):
            mask = F.interpolate(mask.float(), size=(H_f, W_f), mode='bilinear', align_corners=False)

        prototypes = []
        for c in range(self.num_classes):
            m_c = mask[:, c:c+1, :, :]  # (B, 1, H_f, W_f)
            m_sum = m_c.sum(dim=(2, 3), keepdim=True)  # (B, 1, 1, 1)

            # Masked average pooling
            weighted_feat = (feature_map * m_c).sum(dim=(2, 3))  # (B, C)
            proto_c = weighted_feat / (m_sum.squeeze(-1).squeeze(-1) + 1e-6)  # (B, C)

            # Absent landmark fallback to empty token
            is_present = (m_sum.squeeze(-1).squeeze(-1) > 1.0).float()  # (B, 1)
            empty_c = self.empty_tokens[c].unsqueeze(0).expand(B, -1)  # (B, C)
            proto_c = is_present * proto_c + (1.0 - is_present) * empty_c

            # Add class identity and temporal position tags
            proto_c = proto_c + self.class_embeddings[c].unsqueeze(0) + self.time_embeddings[time_idx].unsqueeze(0)
            prototypes.append(proto_c)

        stacked = torch.stack(prototypes, dim=1)  # (B, num_classes, hidden_dim)
        stacked = self.norm(self.proj(stacked))
        return stacked

    def forward(self, past_feature_maps, past_masks):
        """
        Args:
            past_feature_maps: list of feature maps [F_{t-2}, F_{t-1}], each (B, C, H_f, W_f)
            past_masks: list of masks [M_{t-2}, M_{t-1}], each (B, num_classes, H_m, W_m)
            
        Returns:
            K_past: (B, 2 * num_classes, hidden_dim) e.g., (B, 6, 256)
        """
        all_protos = []
        for time_idx, (f_map, m) in enumerate(zip(past_feature_maps, past_masks)):
            proto = self.pool_single_frame(f_map, m, time_idx=min(time_idx, 1))
            all_protos.append(proto)

        K_past = torch.cat(all_protos, dim=1)  # (B, 6, hidden_dim)
        return K_past
