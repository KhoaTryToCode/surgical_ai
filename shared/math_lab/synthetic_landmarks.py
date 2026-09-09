"""
Synthetic Mathematical Surgical Landmark Generators.
Provides continuous analytical ground-truth trajectories representing typical
anatomical structures in laparoscopic liver surgery:
1. Falciform Ligament (S-curve, inflection point)
2. Anterior Ridge (C-curve, asymmetric apex, sharp curvature)
3. Liver Silhouette (Elongated smooth boundary, low curvature)
4. Tortuous Vessel / Deformed Landmark (High-frequency bending)
All outputs are normalized to [0, 1]^2 image space.
"""

import numpy as np
import torch


def generate_falciform_ligament(num_points: int = 500, noise_std: float = 0.0) -> np.ndarray:
    """
    S-shaped curve traversing vertically across the liver.
    Contains an inflection point (d2y/dx2 changes sign).
    """
    t = np.linspace(0, 1, num_points)
    # Cubic trajectory with inflection around t=0.5
    # x(t): subtle lateral sweep
    # y(t): monotonic descent from superior to inferior
    x = 0.5 + 0.18 * np.sin(2.0 * np.pi * t - 0.5) + 0.05 * (t - 0.5)
    y = 0.15 + 0.70 * t + 0.04 * np.cos(np.pi * t)
    
    curve = np.stack([x, y], axis=-1)
    if noise_std > 0:
        curve += np.random.normal(0, noise_std, curve.shape)
    return np.clip(curve, 0.0, 1.0)


def generate_anterior_ridge(num_points: int = 500, noise_std: float = 0.0) -> np.ndarray:
    """
    Asymmetric sharp C-curve representing the anterior inferior liver edge.
    Features a high localized curvature peak (apex of the liver lobe).
    """
    t = np.linspace(0, 1, num_points)
    # Apex occurs around t = 0.65
    x = 0.20 + 0.60 * t
    # Quadratic + exponential peak for sharp localized bend
    y = 0.75 - 0.45 * np.exp(-((t - 0.65) ** 2) / 0.04) + 0.10 * (t - 0.5) ** 2
    
    curve = np.stack([x, y], axis=-1)
    if noise_std > 0:
        curve += np.random.normal(0, noise_std, curve.shape)
    return np.clip(curve, 0.0, 1.0)


def generate_liver_silhouette(num_points: int = 500, noise_std: float = 0.0) -> np.ndarray:
    """
    Wide, shallow parabolic arc representing the diaphragm-liver silhouette boundary.
    Low, nearly constant curvature across large pixel span.
    """
    t = np.linspace(0, 1, num_points)
    x = 0.10 + 0.80 * t
    y = 0.20 + 0.25 * (4.0 * (t - 0.5) ** 2) - 0.05 * t
    
    curve = np.stack([x, y], axis=-1)
    if noise_std > 0:
        curve += np.random.normal(0, noise_std, curve.shape)
    return np.clip(curve, 0.0, 1.0)


def generate_tortuous_landmark(num_points: int = 500, noise_std: float = 0.0) -> np.ndarray:
    """
    High-tortuosity curvilinear structure (e.g. vascular branch or severely deformed ridge).
    Features multiple local inflection points and direction shifts.
    """
    t = np.linspace(0, 1, num_points)
    x = 0.25 + 0.50 * t + 0.08 * np.sin(4.0 * np.pi * t)
    y = 0.20 + 0.60 * t + 0.06 * np.cos(3.0 * np.pi * t)
    
    curve = np.stack([x, y], axis=-1)
    if noise_std > 0:
        curve += np.random.normal(0, noise_std, curve.shape)
    return np.clip(curve, 0.0, 1.0)


def get_all_benchmark_landmarks(num_points: int = 500) -> dict:
    return {
        "Falciform_Ligament_S_Curve": generate_falciform_ligament(num_points),
        "Anterior_Ridge_Sharp_Apex": generate_anterior_ridge(num_points),
        "Liver_Silhouette_Wide_Arc": generate_liver_silhouette(num_points),
        "Tortuous_Deformed_Landmark": generate_tortuous_landmark(num_points),
    }
