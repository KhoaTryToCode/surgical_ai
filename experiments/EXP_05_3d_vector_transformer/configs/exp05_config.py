import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class EXP05Config:
    # Experiment metadata
    exp_id: str = "EXP_05_3d_vector_transformer"
    description: str = "Monocular 3D Vector Space Transformer with Masked Attention"
    
    # Model dimensions
    num_instances: int = 10         # Max number of landmark instances (N)
    num_points: int = 20            # Number of sequential polyline vertices per instance (K)
    num_classes: int = 4            # 0: Background, 1: Ridge, 2: Silhouette, 3: Ligament
    embed_dim: int = 256            # Hidden feature dimension (C)
    num_decoder_layers: int = 6     # Number of Transformer Decoder layers (L)
    num_heads: int = 8              # Multi-head attention heads
    feedforward_dim: int = 1024     # FFN inner dimension
    
    # Geometry & Depth unprojection specs
    fov_degrees: float = 60.0       # Laparoscopic field of view (canonical pinhole)
    z_min: float = 0.1              # Canonical depth lower bound
    z_max: float = 1.0              # Canonical depth upper bound
    
    # Backbone specs (Matching Mask2Former EXP_01)
    mask2former_model_name: str = "facebook/mask2former-swin-tiny-ade-semantic"
    
    # Loss weights
    lambda_cls: float = 2.0         # Classification Focal Loss weight
    lambda_pos: float = 5.0         # Bidirectional 3D Smooth L1 Position Loss weight
    lambda_tan: float = 2.0         # Cosine Tangent Edge Alignment Loss weight
    lambda_curv: float = 1.0        # Discrete 1D Laplacian Curvature Loss weight
    lambda_mask: float = 5.0        # Auxiliary 2D Mask BCE + Dice Loss weight
    
    # Training & Execution
    batch_size: int = 4
    num_workers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    mask_stroke_thickness: int = 35 # GT Mask rasterization stroke thickness at 1024x1024
    
    # Data paths resolution (Local macOS fallback vs Kaggle)
    dataset_dir: str = field(default_factory=lambda: (
        "/kaggle/working/L3D" if os.path.exists("/kaggle/working/L3D")
        else (
            "data/laparoscopic_liver" if os.path.exists("data/laparoscopic_liver")
            else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "laparoscopic_liver")
        )
    ))

config = EXP05Config()
