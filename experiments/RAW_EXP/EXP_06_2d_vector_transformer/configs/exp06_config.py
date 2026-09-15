import os
from dataclasses import dataclass, field

def resolve_dataset_dir() -> str:
    """
    Auto-detects dataset location across Kaggle, Colab, and local macOS environments.
    """
    kaggle_paths = [
        "/kaggle/working/L3D",
        "/kaggle/input/laparoscopic-liver-landmarks",
        "/kaggle/input/laparoscopic-dataset",
        "/kaggle/input/l3d-train"
    ]
    for p in kaggle_paths:
        if os.path.exists(p):
            return p

    colab_path = "/content/L3D"
    if os.path.exists(colab_path):
        return colab_path

    # Local macOS fallback
    local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/laparoscopic_liver"))
    return local_path

@dataclass
class EXP06Config:
    """
    Configuration for EXP_06 Direct 2D Vector Space Transformer.
    Operates natively in normalized 2D image coordinates (u, v) in [0, 1]^2 with Learned Query Embeddings.
    """
    # Query & Polyline specs
    num_instances: int = 10         # Maximum number of query slots
    num_points: int = 20            # Number of ordered vertices per landmark polyline
    num_classes: int = 4            # 1: Falciform, 2: Ridge, 3: Silhouette, 4: Gallbladder (+ Class 0: Background)
    embed_dim: int = 256            # Transformer hidden embedding dimension
    num_heads: int = 8              # Multihead attention heads
    feedforward_dim: int = 1024     # FFN intermediate dimension
    decoder_layers: int = 6         # Number of hierarchical masked decoder layers
    dropout: float = 0.1            # Transformer query dropout regularization
    
    # Backbone specs (Matching Mask2Former Swin-Tiny)
    mask2former_model_name: str = "facebook/mask2former-swin-tiny-ade-semantic"
    decoder_stride_idx: int = 0     # 0: Stride-4 (256x256 = 65,536 spatial tokens for High Precision A100)
    
    # Loss weights (3 Core Objectives: Classification + 2D Coordinates + Dot-Product Mask)
    lambda_cls: float = 2.0         # Multi-Class Focal Loss weight
    lambda_pos: float = 5.0         # Bidirectional 2D L1 Coordinate Loss weight
    lambda_mask: float = 2.0        # Auxiliary 2D Dot-Product Mask BCE + Dice Loss weight
    
    # Training specs
    batch_size: int = 8
    num_workers: int = 2
    learning_rate: float = 1e-4
    backbone_lr_mult: float = 0.1   # Backbone trains at 1e-5 (0.1x) to prevent pre-trained weight drift
    weight_decay: float = 1e-4
    num_epochs: int = 50
    mask_stroke_thickness: int = 35 # GT Mask rasterization stroke thickness at 1024x1024
    
    # Dataset path resolution
    dataset_dir: str = field(default_factory=resolve_dataset_dir)

config = EXP06Config()
