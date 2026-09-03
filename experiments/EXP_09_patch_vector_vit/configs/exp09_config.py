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
class EXP09Config:
    """
    Configuration for EXP_09: Patch-Level Bézier Vector Vision Transformer.
    
    Architecture:
    - Input resolution: 512x512
    - Channels: 4 (RGB + Depth Anything V2) or 3 (RGB-only)
    - Patch size P = 16 (Grid: 32x32 = 1024 patches)
    - ViT Backbone: timm vit_tiny_patch16_224 (in_chans=4) or custom modular ViT encoder
    - Per-patch dual heads:
        1. Class head: (C+1) logits [0: Background, 1: Ridge, 2: Silhouette, 3: Ligament, 4: Gallbladder]
        2. Bézier head: 4 control points (P0, P1, P2, P3) in [0, 1]^2 -> 8 parameters
    - Merging: Global coordinate shift + Anti-aliased line rasterization
    """
    # ---------- Geometry & Grid Specs ----------
    image_size: int = 512            # Input resolution (512x512)
    patch_size: int = 16             # P = 16 px -> Grid 32x32
    grid_size: int = 32              # 512 / 16 = 32
    num_patches: int = 1024          # 32 x 32 = 1024
    num_classes: int = 4             # 1: Ridge, 2: Silhouette, 3: Ligament, 4: Gallbladder (0: Background)
    
    # ---------- Depth Anything V2 Modality ----------
    use_depth: bool = True           # Ingest precomputed Depth Anything V2 maps as 4th channel
    in_chans: int = 4                # 4 channels: [R, G, B, Depth] (or 3 for RGB-only)
    
    # ---------- Cubic Spline & Bézier Specs ----------
    spline_step_px: float = 8.0      # Arc-length resampling step size
    bezier_order: int = 3            # Cubic Bézier: 4 control points (P0, P1, P2, P3)
    num_ctrl_points: int = 4         # 4 points x 2 = 8 coords per patch
    num_sampled_points: int = 10     # Points sampled along Bézier for training/rendering
    stroke_thickness: int = 2        # Merging stroke thickness in pixels
    
    # ---------- Backbone ----------
    backbone_name: str = "vit_tiny_patch16_224"
    pretrained: bool = True
    embed_dim: int = 192             # Embedding dimension for vit_tiny
    depth: int = 12                  # Transformer depth
    num_heads: int = 3               # Number of attention heads
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    
    # ---------- Loss Weights ----------
    lambda_cls: float = 2.0          # Patch classification Focal Loss weight
    lambda_ctrl: float = 5.0         # Control point Smooth L1 weight (active patches)
    lambda_sample: float = 5.0       # Sampled curve L1 weight
    lambda_tan: float = 1.0          # Tangent cosine alignment weight
    lambda_cont: float = 0.5         # Adjacent patch endpoint continuity loss weight
    
    # ---------- Focal Loss Hyperparams ----------
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # ---------- Training ----------
    batch_size: int = 16
    num_workers: int = 2
    learning_rate: float = 1e-4
    backbone_lr_mult: float = 0.1    # Backbone fine-tuned at 1e-5
    weight_decay: float = 1e-4
    num_epochs: int = 80
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # ---------- Thresholds ----------
    confidence_thresh: float = 0.5   # Threshold for active patch during inference
    
    # ---------- Weights & Biases ----------
    wandb_key: str = "83f4544a22543e319c6009abceaac90b634c68a3"
    wandb_project: str = "Surgical_AI_Patch_Bézier_ViT"
    wandb_run_name: str = "EXP_09_ViT_Patch_Bézier"
    
    # ---------- Dataset ----------
    dataset_dir: str = field(default_factory=resolve_dataset_dir)


config = EXP09Config()
