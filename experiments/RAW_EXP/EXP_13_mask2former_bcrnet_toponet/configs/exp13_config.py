"""
EXP_13: Configuration for Mask2Former-BCRNet with TopoNet Loss
================================================================
Centralized configuration dataclass for the master synthesis model:
- Multi-Scale Pixel Decoder + Masked Cross-Attention (Mask2Former)
- Multi-Class Center-Line Topological Constraint (TopoNet clDice)
- 5th-Order Parametric Bézier Curve Refinement (BCRNet)
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


def resolve_dataset_dir(cli_path: Optional[str] = None) -> str:
    """
    Robust dataset path resolution across local macOS and Kaggle CUDA environments.
    Checks CLI override, Kaggle standard paths, and local repo paths.
    """
    candidates = [
        cli_path,
        os.environ.get("L3D_DATASET_DIR"),
        "/kaggle/working/L3D",
        "/kaggle/input/laparoscopic-liver-landmark-detection-l3d/L3D",
        "/kaggle/input/l3d-dataset/L3D",
        "/kaggle/input/laparoscopic-liver/L3D",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/laparoscopic_liver")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/L3D")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data")),
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return cli_path or "/kaggle/working/L3D"


def resolve_checkpoint_dir(cli_path: Optional[str] = None) -> str:
    """Resolve save directory for checkpoints."""
    if cli_path:
        return cli_path
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/checkpoints/EXP_13"
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../checkpoints/EXP_13"))


@dataclass
class EXP13Config:
    # --- Experiment Identity ---
    exp_id: str = "EXP_13_mask2former_bcrnet_toponet"
    description: str = "Master Synthesis: Mask2Former + TopoNet Loss + BCRNet"

    # --- Dataset & Paths ---
    dataset_dir: str = field(default_factory=resolve_dataset_dir)
    save_dir: str = field(default_factory=resolve_checkpoint_dir)
    image_size: int = 512
    in_channels: int = 4  # RGB + Depth Anything V2
    use_depth: bool = True
    spline_step_px: float = 8.0
    stroke_thickness: int = 2
    render_size: int = 128

    # --- Anatomical Classes ---
    # 0: Anterior Ridge, 1: Silhouette, 2: Falciform Ligament
    num_classes: int = 3
    class_names: List[str] = field(default_factory=lambda: ["Ridge", "Silhouette", "Ligament"])

    # --- Mask2Former Engine ---
    backbone: str = "resnet50"  # "resnet50" or "swin-tiny"
    fpn_dim: int = 256
    num_queries_per_class: int = 5  # 5 candidate curve proposals per landmark class -> 15 total
    num_m2f_decoder_layers: int = 6  # Masked cross-attention layers
    m2f_num_heads: int = 8

    # --- Parametric Bézier Curve Head (BCRNet) ---
    bezier_degree: int = 5       # 5th-order Bézier curve
    num_ctrl_pts: int = 6        # K = degree + 1 = 6 control points
    num_sample_pts: int = 26     # 25 uniform points along t in [0, 1] + 1 midpoint
    hcr_stages: int = 3          # 3 coarse-to-fine refinement stages
    bounded_tanh_scale: float = 0.35  # Bounded coordinate offset scale

    # --- Topological Constraints (TopoNet) ---
    skel_num_iter: int = 20      # Iterations of soft morphological erosion/dilation
    lambda_toponet: float = 1.0  # Centerline clDice loss weight

    # --- Loss Formulation & Weights ---
    lambda_m2f_ce: float = 1.0     # Mask2Former query classification CE
    lambda_m2f_mask: float = 2.0   # Mask2Former pixel mask BCE + Dice
    lambda_crv: float = 3.0        # Bidirectional Bézier min-matching Chamfer loss
    lambda_tangent: float = 0.5    # Cosine tangent vector alignment loss
    lambda_contain: float = 0.5    # Differentiable mask containment loss

    # --- Training Schedule & Optimization ---
    epochs: int = 60
    batch_size: int = 4
    num_workers: int = 2
    lr: float = 6e-5
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    device: str = "cuda"

    # --- WandB Logging ---
    wandb_project: str = "surgical-landmark-detection"
    wandb_entity: Optional[str] = None
    wandb_key: str = "83f4544a22543e319c6009abceaac90b634c68a3"
    wandb_id: str = "exp13_m2f_bcrnet_toponet_10423057"
