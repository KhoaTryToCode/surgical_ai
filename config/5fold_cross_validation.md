# Canonical 5-Fold Patient-Disjoint Cross-Validation Specification

## Overview
To address patient-level distribution shifts and provide unbiased, representative performance estimates for sparse video landmark segmentation models, the 35 patients in the Train + Val pool (1,043 frames) are partitioned into **5 patient-disjoint folds**.

- **File**: `config/5fold_patient_splits.json` (Locked with `chmod 444` read-only protection)
- **SHA-256**: `18d76bcd6899116115d061fdeee6592ab8feafc145aa428e07101fdd14708ea6`
- **MD5**: `3bb0b24a63ab684ba0c4e643951cb170`
- **Partition Method**: Deterministic Patient-Level K-Fold (`n_splits=5`, `shuffle=True`, `random_state=42`)

---

## 1. Ground Rules & Guarantees
1. **Frozen Test Set (Untouched)**:
   The official test set remains 100% frozen across all folds and is never exposed during training or hyperparameter selection:
   - `Patient_21` (7 frames)
   - `Patient_31` (14 frames)
   - `Patient_41` (84 frames)
   - `Patient_51` (4 frames)
   - **Total Test Frames**: 109 frames (4 patients)
2. **Zero Leakage**:
   Each fold maintains an empty intersection between training and validation patient sets:
   $$\text{Train}_{\text{Patients}} \cap \text{Val}_{\text{Patients}} = \emptyset$$
3. **Exact Coverage**:
   Every pool patient is held out for validation in exactly one fold ($7 \times 5 = 35$ patients).

---

## 2. Fold Partition Table

| Fold | Val Patients (7 patients) | Val Frames | Train Patients (28 patients) | Train Frames |
| :--- | :--- | :--- | :--- | :--- |
| **Fold 0** | `Patient_22` (43), `Patient_25` (46), `Patient_33` (61), `Patient_35` (48), `Patient_39` (5), `Patient_42` (40), `Patient_49` (48) | **291** (27.9%) | Remaining 28 patients | **752** (72.1%) |
| **Fold 1** | `Patient_1` (20), `Patient_8` (43), `Patient_16` (7), `Patient_17` (20), `Patient_20` (36), `Patient_29` (6), `Patient_54` (7) | **139** (13.3%) | Remaining 28 patients | **904** (86.7%) |
| **Fold 2** | `Patient_2` (9), `Patient_4` (18), `Patient_12` (63), `Patient_19` (30), `Patient_30` (4), `Patient_44` (5), `Patient_53` (8) | **137** (13.1%) | Remaining 28 patients | **906** (86.9%) |
| **Fold 3** | `Patient_5` (7), `Patient_18` (6), `Patient_32` (15), `Patient_36` (31), `Patient_38` (99), `Patient_50` (79), `Patient_52` (2) | **239** (22.9%) | Remaining 28 patients | **804** (77.1%) |
| **Fold 4** | `Patient_14` (56), `Patient_15` (19), `Patient_23` (8), `Patient_34` (2), `Patient_40` (101), `Patient_45` (20), `Patient_61` (31) | **237** (22.7%) | Remaining 28 patients | **806** (77.3%) |

---

## 3. Python Integration

Use `shared.utils.split_loader` in any training script:

```python
from shared.utils.split_loader import get_fold_patients, resolve_split_file_paths, resolve_test_file_paths

# 1. Get patient lists for Fold 0
train_pts, val_pts = get_fold_patients(fold_idx=0)

# 2. Automatically resolve image paths from data/L3D pool
train_img_paths, val_img_paths = resolve_split_file_paths(fold_idx=0, data_root="data/L3D")

# 3. Get frozen test image paths
test_img_paths = resolve_test_file_paths(data_root="data/L3D")
```
