"""
EXP_12: SurgicalCurveFormer v2 Configuration
============================================
The Omni-Geometric Master Architecture synthesized from 18 rigorous mathematical proofs.
"""
import os
from dataclasses import dataclass, field


def resolve_dataset_dir() -> str:
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
    # Local fallback
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../data/laparoscopic_liver")
    )


@dataclass
class EXP12Config:
    # --- Experiment Metadata ---
    exp_id: str = "EXP_12_surgical_curve_former_v2"
    exp_name: str = "SurgicalCurveFormer v2: Omni-Geometric Master Synthesis"

    # --- Dataset & Paths ---
    dataset_dir: str = field(default_factory=resolve_dataset_dir)
    save_dir: str = "/kaggle/working/checkpoints/EXP_12"
    image_size: int = 512
    num_classes: int = 3  # Falciform (0), Ridge (1), Silhouette (2)
    use_depth: bool = True

    # --- Architectural Dimensions ---
    bezier_order: int = 5        # 6 control points (Test 01)
    num_ctrl_pts: int = 6
    acpi_top_k: int = 10         # Candidate proposals per class
    acpi_r_max: float = 0.35     # Bounded Tanh radius (Test 09)
    
    hcr_ref_points: int = 26     # 25 uniform + 1 center (Test 10)
    hcr_stages: int = 3
    fpn_dim: int = 256
    snake_width_px: float = 8.0  # ON-Snake lateral cross-section (Test 08)

    # --- Class-Adaptive Cauchy Rasterizer (Tests 02, 14) ---
    rasterizer_grid_size: int = 128
    sigmas_px: list[float] = field(default_factory=lambda: [20.0, 8.0, 16.0])

    # --- Loss Formulation & Weights (Tests 06, 07, 12, 15) ---
    lambda_s: float = 10.0         # Auxiliary CNN deep supervision
    lambda_ind: float = 1.0        # Proposal induction
    ind_pos_weight: float = 15.0   # Imbalance compensation (Test 12)
    
    lambda_cs: float = 1.0         # Proposal confidence
    lambda_crv: float = 2.0        # Bidirectional curve distance (Test 03, 06)
    lambda_ac_cldice: float = 1.0  # Analytical continuous clDice (Test 07)
    lambda_dice: float = 0.5       # Cauchy soft rasterizer Dice
    lambda_exist: float = 1.0      # CLS existence gate (Test 16)
    exist_pos_weight: float = 3.0

    # --- Hold-15 + Cosine Annealing Schedule (Test 11) ---
    hold_epochs: int = 15
    lambda_d_min: float = 0.05

    # --- Optimization ---
    epochs: int = 80
    batch_size: int = 4
    lr: float = 5e-5
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    amp: bool = True
    num_workers: int = 2

    # --- Tracking ---
    wandb: bool = True
    wandb_project: str = "Surgical_AI_EXP12_SurgicalCurveFormerV2"
    wandb_entity: str = "10423057-vietnamese-german-university"
    wandb_run_name: str = "EXP12_OmniGeometric_v2"
