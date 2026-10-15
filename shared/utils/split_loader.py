"""
Utility for loading and resolving patient-disjoint 5-Fold Cross-Validation splits.
Ensures 100% reproducibility across local macOS and Kaggle CUDA environments.

Frozen Test Set: Patients 21, 31, 41, 51 (109 frames) - NEVER used in any fold.
Train + Val Pool: 35 patients (1,043 frames) partitioned into 5 patient-disjoint folds.
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

SPLITS_JSON_PATH = Path(__file__).resolve().parent.parent.parent / "_config" / "5fold_patient_splits.json"

EXPECTED_SHA256 = "18d76bcd6899116115d061fdeee6592ab8feafc145aa428e07101fdd14708ea6"


def load_5fold_splits(json_path: Optional[str] = None) -> Dict:
    """
    Loads and validates the canonical 5-fold cross-validation split configuration.
    """
    path = Path(json_path) if json_path else SPLITS_JSON_PATH
    if not path.exists():
        raise FileNotFoundError(f"Canonical 5-fold split configuration not found at '{path}'")

    with open(path, "r") as f:
        data = json.load(f)

    # Sanity checks
    assert data["metadata"]["n_splits"] == 5, "Configuration must define exactly 5 splits."
    assert data["metadata"]["total_pool_patients"] == 35, "Total pool patients must be 35."
    return data


def get_fold_patients(fold_idx: int, json_path: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Returns (train_patients, val_patients) for the specified fold index (0 to 4).
    """
    if not (0 <= fold_idx < 5):
        raise ValueError(f"fold_idx must be between 0 and 4, got {fold_idx}")

    data = load_5fold_splits(json_path)
    fold_key = f"fold_{fold_idx}"
    fold_data = data["folds"][fold_key]

    return fold_data["train_patients"], fold_data["val_patients"]


def get_frozen_test_patients(json_path: Optional[str] = None) -> List[str]:
    """
    Returns the frozen test patients (Patient_21, Patient_31, Patient_41, Patient_51).
    """
    data = load_5fold_splits(json_path)
    return data["metadata"]["frozen_test_patients"]


def resolve_split_file_paths(
    fold_idx: int,
    data_root: str = "data/L3D",
    json_path: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Resolves image file paths for train and validation sets of a given fold
    by scanning the combined pool of images (Train + Val directories).
    
    Args:
        fold_idx: Fold index (0 to 4).
        data_root: Root directory of L3D dataset containing 'Train/images' and 'Val/images'.
        json_path: Optional custom path to 5fold_patient_splits.json.
        
    Returns:
        (train_img_paths, val_img_paths) as sorted lists of absolute or resolved paths.
    """
    train_patients, val_patients = get_fold_patients(fold_idx, json_path)
    val_set = set(val_patients)
    train_set = set(train_patients)

    root = Path(data_root)
    # Collect all image files from Train and Val subdirectories (the pool)
    candidate_dirs = [root / "Train" / "images", root / "Val" / "images"]
    pool_files = []
    for cdir in candidate_dirs:
        if cdir.exists():
            pool_files.extend(glob.glob(str(cdir / "*.jpg")) + glob.glob(str(cdir / "*.png")))

    if not pool_files:
        raise FileNotFoundError(f"No image files found under {candidate_dirs}")

    train_files = []
    val_files = []

    for fpath in pool_files:
        fname = os.path.basename(fpath)
        # Patient ID extraction: 'Patient_XX_...'
        parts = fname.split("_")
        p_id = f"{parts[0]}_{parts[1]}" if len(parts) > 1 else None

        if p_id in val_set:
            val_files.append(str(Path(fpath).resolve()))
        elif p_id in train_set:
            train_files.append(str(Path(fpath).resolve()))

    train_files.sort()
    val_files.sort()

    return train_files, val_files


def resolve_test_file_paths(data_root: str = "data/L3D") -> List[str]:
    """
    Resolves image file paths for the frozen test set.
    """
    root = Path(data_root)
    test_dir = root / "Test" / "images"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test images directory not found at '{test_dir}'")

    test_files = glob.glob(str(test_dir / "*.jpg")) + glob.glob(str(test_dir / "*.png"))
    test_files.sort()
    return [str(Path(f).resolve()) for f in test_files]
