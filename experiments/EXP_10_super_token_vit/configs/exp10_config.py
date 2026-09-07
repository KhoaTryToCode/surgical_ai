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
    Configuration for EXP_10: Macro-Patch Geometric Vision Transformer (Way A).
    
    Key Principles:
    - Input resolution: 512x512
    - Channels: 4 (RGB + Depth Anything V2)
    - Micro-Patch size (ViT): P_micro = 16 px -> Grid 32x32 = 1,024 micro-tokens
    - Macro-Patch size (Prediction): P_macro = 64 px -> Grid 8x8 = 64 macro-tokens
    - Merge factor: 4x4 micro-patches (16 tokens) -> 1 macro-patch
    - Inter-Macro Relational Attention: All-to-all self-attention on 64 macro-tokens for global organ pose
    - Spatial Anchor: Local coordinates in [0, 1] scaled by 64 and shifted by (c*64, r*64)
    - Continuity: Inter-macro endpoint continuity loss + 80% fewer boundary seams
    """
    # ---------- Geometry & Grid Specs ----------
    image_size: int = 512            # Input resolution (512x512)
    micro_patch_size: int = 16       # ViT feature patch size (16 px)
    macro_patch_size: int = 64       # Macro-patch size for prediction (64 px)
    grid_size: int = 8               # 512 / 64 = 8x8 macro-grid
    num_patches: int = 64            # 8 x 8 = 64 macro-tokens
    num_classes: int = 4             # 1: Ridge, 2: Silhouette, 3: Falciform Ligament, 4: Gallbladder (0: Background)
    
    class_names: tuple = (
        "Anterior Ridge",
        "Liver Silhouette",
        "Falciform Ligament",
        "Gallbladder Boundary"
    )
    
    # ---------- Modality ----------
    use_depth: bool = True           # RGB-D 4 channels
    in_chans: int = 4
    
    # ---------- Cubic Bézier Specs ----------
    spline_step_px: float = 8.0      # Polyline arc-length resampling step
    num_ctrl_points: int = 4         # Cubic Bézier: P0, P1, P2, P3 inside each 64x64 macro-patch
    num_sampled_points: int = 10     # Points sampled along curve for loss/continuity
    stroke_thickness: int = 2        # Evaluation stroke thickness in pixels
    
    # ---------- Backbone & Macro Transformer ----------
    backbone_name: str = "vit_base_patch16_224"  # ViT-Base (86M params)
    pretrained: bool = True
    embed_dim: int = 768
    macro_depth: int = 2             # 2 relational self-attention layers on the 64 macro-tokens
    macro_heads: int = 8             # 8 heads for inter-macro relational reasoning
    dropout: float = 0.0
    
    # ---------- Loss Weights ----------
    lambda_cls: float = 2.0          # Macro classification Focal Loss weight
    lambda_ctrl: float = 5.0         # Control point Smooth L1 weight (active patches)
    lambda_sample: float = 5.0       # Sampled curve L1 weight
    lambda_tan: float = 1.0          # Tangent cosine alignment weight
    lambda_cont: float = 1.5         # Adjacent macro-patch endpoint continuity loss weight
    
    # Class weights for focal loss: [0: Background, 1: Ridge, 2: Silhouette, 3: Ligament, 4: Gallbladder]
    class_weights: tuple = (0.20, 1.0, 1.0, 2.5, 3.0)
    focal_gamma: float = 2.0
    
    # ---------- Training Hyperparameters ----------
    batch_size: int = 16
    num_workers: int = 2
    learning_rate: float = 1e-4
    backbone_lr_mult: float = 0.1    # Backbone fine-tuned gently at 1e-5
    weight_decay: float = 1e-4
    num_epochs: int = 80
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    use_amp: bool = True             # AMP on CUDA
    
    # ---------- Thresholds ----------
    confidence_thresh: float = 0.30  # Macro-patch activation threshold
    
    # ---------- Weights & Biases ----------
    wandb_key: str = "83f4544a22543e319c6009abceaac90b634c68a3"
    wandb_project: str = "Surgical_AI_EXP10_Macro"
