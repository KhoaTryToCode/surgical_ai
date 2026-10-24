"""
EXP_11: SurgicalCurveFormer Configuration
==========================================
Architecture: BCRNet Core (ACPI + HCR) + EXP_10 Existence Gating + Soft Rasterizer Dice
Target: >69.57% DSC on L3D test set (BCRNet current SOTA)
"""
import os
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
        "/kaggle/input/laparoscopic-liver-landmarks",
    ]
    for p in kaggle_paths:
        if os.path.exists(p):
            return p
    colab_path = "/content/L3D"
    if os.path.exists(colab_path):
        return colab_path
    # Local macOS fallback
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../data/laparoscopic_liver")
    )


@dataclass
class EXP11Config:
    """
    SurgicalCurveFormer (EXP_11) — Full Configuration.

    Architectural Dimensions Reference (for math correctness verification):
    -----------------------------------------------------------------------
    Input:           (B, 4, 512, 512)    [RGB + AdelaiDepth]
    SAM features:    f_s from frozen SAM-ViT-B
    CNN features:    f_e from ResNet-50 (RGB-D, 4-channel)
    FPN scales:
      f1: (B, C_fpn, 256, 256)  stride 4  — finest
      f2: (B, C_fpn, 128, 128)  stride 8
      f3: (B, C_fpn, 64,  64)   stride 16
      f4: (B, C_fpn, 32,  32)   stride 32 — ACPI operates here

    ACPI (per class m ∈ {0,1,2}):
      Input:  f4  (B, C_fpn, 32, 32)
      Output: Δb  (B, 12, 32, 32)  [6 control point offsets × 2 coords]
              s   (B, 1, 32, 32)   [confidence score]
      Control points:
        b_i^j = (sigma(Δb_ix^j + logit(c_ix)), sigma(Δb_iy^j + logit(c_iy)))
      Top-K selection: K=10 per class

    HCR Reference Points (per proposal):
      P_s: (M, K, N-1, 2)   25 uniform curve samples
      P*:  (M, K,   1, 2)   1 center point B(0.5)
      P:   (M, K,   N, 2)   N=26 total  (M=num_classes, K=10)

    Bézier Curve (degree 5, K=6 control points):
      B(t) = Σ_{j=0}^5 C(5,j) (1-t)^{5-j} t^j P_j,  t ∈ [0,1]
      Bernstein basis matrix: M_bern ∈ R^{N_sample × 6}
    """

    # ─── Input / Geometry ────────────────────────────────────────────────────
    image_size: int = 512                     # Image H = W = 512
    in_chans: int = 4                         # 3 (RGB) + 1 (AdelaiDepth)
    num_landmark_classes: int = 3             # Ridge(0), Silhouette(1), Ligament(2)

    # ─── FPN ─────────────────────────────────────────────────────────────────
    fpn_channels: int = 256                   # All FPN levels share C=256 channels
    fpn_strides: tuple = (4, 8, 16, 32)      # f1..f4 spatial strides

    # ─── ACPI — Adaptive Curve Proposal Initialization ───────────────────────
    # BCRNet: operates on f4 (32×32 feature map)
    acpi_feature_level: int = 3              # 0-indexed; level 3 = f4 (stride 32)
    bezier_degree: int = 5                   # 5th-order → 6 control points
    bezier_ctrl_pts: int = 6                 # K_ctrl = degree + 1
    acpi_proposal_pool: int = 256            # Dense pixel proposals per class on f4
    acpi_top_k: int = 10                     # Keep top-K per class after NMS
    acpi_conf_threshold: float = 0.3         # Inference confidence threshold

    # ─── HCR — Hierarchical Curve Refinement ─────────────────────────────────
    # BCRNet: 3 stages, coarse→fine
    # Stage 0: {f3, f4}   Stage 1: {f2, f3}   Stage 2: {f1, f2}
    hcr_num_stages: int = 3
    hcr_feature_pairs: tuple = ((2, 3), (1, 2), (0, 1))  # (coarse_idx, fine_idx) 0-indexed
    hcr_ref_points_n: int = 26               # N ref points: 25 uniform + 1 center
    hcr_deform_heads: int = 8               # Deformable cross-attention heads
    hcr_deform_sampling_pts: int = 4        # Sampling points per head per level
    hcr_embed_dim: int = 256               # Query/key/value dimension in HCR
    hcr_ffn_dim: int = 512                 # FFN hidden dim in HCR transformer layers
    hcr_dropout: float = 0.1

    # ─── Super-Token Existence Gate (EXP_10) ─────────────────────────────────
    vit_backbone: str = "vit_base_patch16_224"  # ViT-B, D=768
    vit_patch_size: int = 16
    vit_embed_dim: int = 768
    vit_pretrained: bool = True
    exist_cross_attn_heads: int = 8
    exist_hidden_dim: int = 512

    # ─── Soft Rasterizer (EXP_10) ────────────────────────────────────────────
    raster_render_size: int = 128            # Renders at 128×128, Dice at this scale
    raster_num_samples: int = 64            # Dense Bézier trajectory samples
    raster_sigma_px: float = 2.0           # Gaussian falloff radius (px at 512 resolution)

    # ─── Loss Weights ────────────────────────────────────────────────────────
    # BCRNet annealing: lambda_d = 1 - sigma((epoch - 10) / 2)
    # Early (epoch << 10):  focus on L_s + L_ind  (dense pixel guidance)
    # Late  (epoch >> 10):  focus on L_cs + L_crv  (curve refinement)
    lambda_s: float = 10.0                  # Deep CNN decoder multi-level Dice
    lambda_ind: float = 1.0                 # ACPI proposal induction BCE
    lambda_cs: float = 1.0                  # HCR confidence focal loss (per stage)
    lambda_crv: float = 1.0                 # HCR Bézier geometry loss (per stage)
    lambda_exist: float = 1.5              # Existence BCE (EXP_10)
    lambda_dice_soft: float = 5.0          # Soft rasterizer Dice (EXP_10)
    focal_gamma: float = 2.0               # Focal loss exponent
    focal_alpha: float = 0.25             # Focal loss positive weight

    # Loss annealing midpoint and steepness (from BCRNet)
    anneal_epoch_center: int = 10          # lambda_d = 1 - sigma((e - center) / 2)
    anneal_slope: float = 2.0

    # ─── Training ─────────────────────────────────────────────────────────────
    batch_size: int = 4                    # BCRNet: single RTX A6000 (48GB)
    num_workers: int = 4
    num_epochs: int = 60                   # BCRNet: 60 epochs on L3D
    finetune_epochs: int = 20             # BCRNet: 20 epochs fine-tune on P2ILF
    optimizer: str = "adam"
    learning_rate: float = 1e-5           # BCRNet: lr = 1e-5
    backbone_lr_mult: float = 0.1         # Backbone fine-tuned at 1e-6
    weight_decay: float = 1e-4           # BCRNet: wd = 1e-4
    use_amp: bool = True                  # Automatic Mixed Precision
    grad_clip_norm: float = 1.0           # Gradient clipping

    # ─── Evaluation Dilation ─────────────────────────────────────────────────
    eval_dilate_px: int = 30              # BCRNet/TopoNet protocol: 30px dilation

    # ─── Checkpointing ────────────────────────────────────────────────────────
    save_dir: str = "checkpoints/EXP_11"
    save_every_n_epochs: int = 5
    validate_every_n_epochs: int = 5

    # ─── Weights & Biases ────────────────────────────────────────────────────
    wandb_key: str = "83f4544a22543e319c6009abceaac90b634c68a3"
    wandb_project: str = "Surgical_AI_EXP11_SurgicalCurveFormer"
    wandb_run_name: str = "EXP_11_SurgicalCurveFormer"

    # ─── Dataset ─────────────────────────────────────────────────────────────
    dataset_dir: str = field(default_factory=resolve_dataset_dir)
    spline_step_px: float = 8.0           # Arc-length resampling for GT polylines
    use_depth: bool = True
    stroke_thickness: int = 2             # Evaluation mask rendering thickness

    # ─── Class Mapping ────────────────────────────────────────────────────────
    # 0=Ridge, 1=Silhouette, 2=Ligament  (0-indexed class IDs for BCR mode)
    CLASS_NAMES: tuple = ("Ridge", "Silhouette", "Ligament")


# Global config singleton
config = EXP11Config()
