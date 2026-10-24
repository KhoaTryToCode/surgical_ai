import torch
import torch.nn as nn


class GlobalCurveDecoder(nn.Module):
    """
    Decodes pooled Landmark Super-Tokens (B, C, D) into:
    1. Landmark Existence Logits: (B, C, 1) -> probability landmark is visible
    2. Global 6-Control-Point Bézier/Spline: (B, C, K, 2) in [0, 1]^2
    """
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 4,
        num_ctrl_points: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_ctrl_points = num_ctrl_points
        self.embed_dim = embed_dim
        
        # 1. Existence / Visibility Head
        self.exist_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 2. Global Parametric Curve Control Point Head (K points x 2 coords = 2K parameters)
        self.curve_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_ctrl_points * 2),
            nn.Sigmoid()  # Strictly bounded in normalized image canvas [0, 1]
        )
        
        # Initialize weights
        nn.init.normal_(self.exist_head[-1].weight, std=0.01)
        nn.init.constant_(self.exist_head[-1].bias, 0.0)
        
        nn.init.normal_(self.curve_head[-2].weight, std=0.01)
        # Initialize bias to spread control points along canonical diagonals
        init_bias = torch.linspace(0.2, 0.8, num_ctrl_points * 2)
        self.curve_head[-2].bias.data.copy_(init_bias)

    def forward(self, super_tokens: torch.Tensor) -> dict:
        """
        Args:
            super_tokens: (B, C, D) pooled representation for C landmarks
            
        Returns:
            dict containing:
                exist_logits: (B, C) raw existence logits
                exist_probs:  (B, C) sigmoid probabilities
                ctrl_points:  (B, C, K, 2) global control points in [0, 1]^2
        """
        B, C, D = super_tokens.shape
        
        # Existence prediction
        exist_logits = self.exist_head(super_tokens).squeeze(-1)  # (B, C)
        exist_probs = torch.sigmoid(exist_logits)                # (B, C)
        
        # Global Control Points prediction
        raw_curve = self.curve_head(super_tokens)                # (B, C, K*2)
        ctrl_points = raw_curve.view(B, C, self.num_ctrl_points, 2)  # (B, C, K, 2)
        
        return {
            "exist_logits": exist_logits,
            "exist_probs": exist_probs,
            "ctrl_points": ctrl_points
        }
