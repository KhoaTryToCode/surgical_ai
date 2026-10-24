# EXPERIMENT_11 Results Ledger: Targeted Inversion-Weighted Junction-Steered Mask2Former

## 1. Comparative Performance Baseline

| Metric | EXP_1 (Vanilla M2F) | EXP_5 (Junction-Steered) | EXP_8 (Vis-Gated) | EXP_9 (RGB-D Anchor) | EXP_10 (RGB-D + Junc) | EXP_11 (Inversion-Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Val Macro Dice** | 66.85% | **68.04%** | 66.27% | 66.90% | 67.52% | *Pending* |
| **Val Macro ASSD (px)** | 22.14 | **21.68** | 24.16 | 23.35 | 22.74 | *Pending* |
| **Val Ridge Dice** | 66.12% | **67.43%** | 65.81% | 66.02% | 66.80% | *Pending* |
| **Val Silhouette Dice** | 73.18% | **73.54%** | 72.84% | 72.88% | 73.00% | *Pending* |
| **Val Falciform Dice** | 61.25% | **63.15%** | 60.16% | 61.80% | 62.77% | *Pending* |
| **Patient 40 Macro Dice** | 68.90% | **70.36%** | 68.75% | 69.12% | 69.76% | *Pending* |
| **Patient 40 ASSD (px)** | 21.05 | **19.82** | 21.67 | 20.84 | 20.55 | *Pending* |
| **Test Macro Dice** | 64.91% | **65.82%** | 64.12% | 65.10% | 65.30% | *Pending* |
| **Test Macro ASSD (px)** | 26.45 | **25.21** | 27.05 | 26.15 | 25.90 | *Pending* |

---

## 2. Hard Retracted / Inverted Cases Breakdown (Patient 40 Audit)

| Frame Filename | EXP_5 Dice | EXP_8 Dice | EXP_9 Dice | EXP_10 Dice | EXP_11 Dice | Target Goal / Expected Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Patient_40_08730` | 82.2% | 80.5% | 81.0% | 81.8% | *Pending* | Maintain high performance on moderate traction |
| `Patient_40_08790` | 74.5% | 72.1% | 73.0% | 73.8% | *Pending* | Stable margin delineation |
| `Patient_40_08940` | **0.0%** | **0.0%** | **0.0%** | **0.0%** | *Pending* | **Break the Class-Swap Inversion Trap** (> 50% Dice) |
| `Patient_40_09000` | **10.5%** | **9.8%** | **11.2%** | **10.8%** | *Pending* | **Break the Class-Swap Inversion Trap** (> 50% Dice) |
| `Patient_40_09270` | 78.4% | 76.2% | 77.5% | 78.0% | *Pending* | High precision tracking |
| `Patient_40_09330` | 79.1% | 77.0% | 78.2% | 78.8% | *Pending* | High precision tracking |
| `Patient_40_149190` | 33.2% | 31.5% | 32.8% | 33.0% | *Pending* | Reduce false-positive glare hallucination on absent Falciform |

---

## 3. Training Progress & Milestones
- **Server:** `gpu-a240` (NVIDIA A100 MIG 2g.10gb MPS)
- **Status:** Scaffolding complete; ready for server submission.
