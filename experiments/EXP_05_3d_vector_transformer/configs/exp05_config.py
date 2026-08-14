import os
from dataclasses import dataclass, field
from typing import List

def resolve_dataset_dir() -> str:
    candidate_paths = [
        "/kaggle/working/L3D",
        "/kaggle/input/laparoscopic-liver-landmarks",
        "/kaggle/input/laparoscopic-liver",
        "/kaggle/input/l3d",
        "data/laparoscopic_liver",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "laparoscopic_liver")
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    if os.path.exists("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input"):
            if "images" in dirs or "labels" in dirs:
                return root
    return candidate_paths[0]

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
    
    # Backbone & Feature Pyramid specs (Matching Mask2Former EXP_01)
    mask2former_model_name: str = "facebook/mask2former-swin-tiny-ade-semantic"
    decoder_stride_idx: int = 1     # 1: Stride-8 (128x128 = 16,384 tokens for High Precision A100), 2: Stride-16 (64x64)
    
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
    
    # Data paths resolution (Auto-resolves Kaggle vs Local)
    dataset_dir: str = field(default_factory=resolve_dataset_dir)

config = EXP05Config()
