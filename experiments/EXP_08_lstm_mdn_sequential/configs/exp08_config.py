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
class EXP08Config:
    """
    Configuration for EXP_08: CNN-LSTM-MDN Sequential Surgical Landmark Detection.
    
    Architecture: ResNet-18 backbone → Bilinear Grid Sample → LSTM → MDN Head
    Training: Teacher Forcing with absolute (u,v) coordinates in [0,1]^2
    """
    # ---------- Polyline & Instance Specs ----------
    num_instances: int = 10         # Maximum landmark instances per image
    num_points: int = 20            # K: Number of ordered vertices per polyline
    num_classes: int = 4            # 1: Ridge, 2: Silhouette, 3: Falciform, 4: Gallbladder
    
    # ---------- Backbone (ResNet-18) ----------
    backbone_name: str = "resnet18"
    backbone_pretrained: bool = True
    backbone_feature_dim: int = 256  # ResNet-18 layer3 output channels
    feature_map_stride: int = 16     # layer3 stride → at 512x512 input: 32x32 feature map
    
    # ---------- LSTM Decoder ----------
    lstm_hidden_dim: int = 256       # LSTM hidden state dimension
    lstm_num_layers: int = 1         # Single-layer LSTM for simplicity
    lstm_dropout: float = 0.0       # No dropout for single-layer LSTM
    
    # ---------- MDN Head ----------
    mdn_num_components: int = 10     # M: Number of Gaussian mixture components
    # Per component: (pi, mu_x, mu_y, sigma_x, sigma_y) = 5 params
    # + 1 end-of-sequence logit per step
    # Total MDN output dim = 5*M + 1 = 51
    
    # ---------- Class Embedding ----------
    class_embed_dim: int = 32        # Learnable class embedding dimension
    
    # ---------- Input / INIT Token ----------
    coord_input_dim: int = 2         # Raw (u, v) coordinate input
    init_token_dim: int = 256        # Learnable <INIT> token dimension (matches LSTM input)
    
    # ---------- Image / Resolution ----------
    image_size: int = 512            # Input image resolution (512x512 for faster training)
    mask_render_size: int = 512      # Mask rasterization resolution
    mask_stroke_thickness: int = 18  # GT mask rasterization stroke thickness at 512x512
    
    # ---------- Loss Weights ----------
    lambda_mdn: float = 1.0         # MDN NLL weight
    lambda_point: float = 5.0       # Expected-point Smooth-L1 weight (dominant early)
    lambda_dir: float = 1.0         # Directional cosine alignment weight
    lambda_mask: float = 2.0        # Rasterized mask Dice weight
    lambda_eos: float = 0.5         # End-of-sequence BCE weight
    
    # ---------- Training ----------
    batch_size: int = 16
    num_workers: int = 2
    learning_rate: float = 1e-4
    backbone_lr_mult: float = 0.1   # Backbone fine-tuned at 1e-5
    weight_decay: float = 1e-4
    num_epochs: int = 80
    warmup_epochs: int = 5          # Linear warmup before cosine annealing
    
    # ---------- Soft Mask Rendering ----------
    soft_mask_sigma: float = 15.0   # Gaussian splatting sigma for differentiable mask rendering (pixels at 512x512)
    
    # ---------- Dataset ----------
    dataset_dir: str = field(default_factory=resolve_dataset_dir)


config = EXP08Config()
