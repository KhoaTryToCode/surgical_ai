import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from dataclasses import dataclass, field


def resolve_dataset_dir() -> str:
    """
    Auto-detects dataset location across Kaggle, Colab, and local macOS environments.
    """
    kaggle_paths = [
        "/kaggle/working/L3D",
        "/kaggle/input/datasets/khoatrytopublish/l3d-train/Train",
        "/kaggle/input/datasets/khoatrytopublish/l3d-train",
        "/kaggle/input/l3d-train/Train",
        "/kaggle/input/l3d-train",
        "/kaggle/input/laparoscopic-liver-landmarks"
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
class EXP10Config:
    """
    Configuration for EXP_10: Super-Token Geometric ViT.
    
    Architecture:
    - Input resolution: 512x512
    - Channels: 4 (RGB + Depth Anything V2) or 3 (RGB-only)
    - Patch size P = 16 (Grid: 32x32 = 1024 patches)
    - ViT Backbone: timm vit_base_patch16_224 (in_chans=4) or modular ViT encoder fallback
    - Global Organ Pose Conditioning via [CLS] token (captures organ flip/retraction for Patient 40)
    - Super-Token Cross-Attention: Soft pooling of 1024 patches into C=4 landmark super-tokens
    - Global Curve Decoder: 6 control points (K=6) in [0, 1]^2 -> 12 parameters per landmark
    - Dual-Domain Supervision: Vector Smooth L1 (JSON) + Differentiable Soft Dice (GT Mask) + Attention BCE
    """
    # ---------- Geometry & Grid Specs ----------
    image_size: int = 512            # Input resolution (512x512)
    patch_size: int = 16             # P = 16 px -> Grid 32x32
    grid_size: int = 32              # 512 / 16 = 32
    num_patches: int = 1024          # 32 x 32 = 1024
    num_classes: int = 4             # 1: Ridge, 2: Silhouette, 3: Falciform Ligament, 4: Gallbladder Boundary
    
    # Class names mapping (1-indexed for landmark queries)
    class_names: tuple = (
        "Anterior Ridge",
        "Liver Silhouette",
        "Falciform Ligament",
        "Gallbladder Boundary"
    )
    
    # ---------- Depth Anything V2 Modality ----------
    use_depth: bool = True           # Ingest precomputed Depth Anything V2 maps as 4th channel
    in_chans: int = 4                # 4 channels: [R, G, B, Depth]
    
    # ---------- Global 6-Point Spline Curve Specs ----------
    num_ctrl_points: int = 6         # K = 6 control points per landmark curve (P0..P5 in [0, 1]^2)
    curve_sample_points: int = 64    # Number of evaluation points along curve for rendering/loss
    stroke_thickness_px: float = 2.0 # Target stroke thickness in rasterized evaluation masks
    
    # ---------- Backbone & Attention Specs ----------
    backbone_name: str = "vit_base_patch16_224"  # ViT-Base: 86M params, 12 heads, 768 dim
    pretrained: bool = True
    embed_dim: int = 768             # Embedding dimension for vit_base
    depth: int = 12                  # Transformer depth
    num_heads: int = 12              # 12 attention heads
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    cross_attn_heads: int = 8        # Multi-head cross-attention for super-token aggregation
    
    # ---------- Loss Weights (Dual-Domain Supervision) ----------
    lambda_attn: float = 2.0         # Patch-level attention heatmap BCE loss weight
    lambda_vector: float = 5.0       # Global 6-control-point Smooth L1 loss weight
    lambda_dice: float = 5.0         # Differentiable soft Dice loss weight against GT mask
    lambda_exist: float = 1.5        # Landmark visibility/existence classification BCE loss weight
    
    # ---------- Focal Loss Hyperparams for Attention ----------
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75        # Favor sparse landmark patch foreground
    
    # ---------- Training Hyperparameters ----------
    batch_size: int = 16
    num_workers: int = 2
    learning_rate: float = 1e-4
    backbone_lr_mult: float = 0.1    # Backbone fine-tuned gently at 1e-5
    weight_decay: float = 1e-4
    num_epochs: int = 80
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    use_amp: bool = True             # Automatic Mixed Precision for CUDA training
    
    # ---------- Thresholds ----------
    existence_thresh: float = 0.35   # Landmark existence probability threshold
    attn_viz_thresh: float = 0.10    # Threshold for highlighting active patches in visualizer
    
    # ---------- Weights & Biases ----------
    wandb_key: str = "83f4544a22543e319c6009abceaac90b634c68a3"
    wandb_project: str = "Surgical_AI_EXP10"
    wandb_entity: str = ""
