# Stage Contract: Experiment Generation & Sub-Experiment Execution (Layer 2)

Defines decision criteria and workflow steps for scaffolding surgical landmark detection experiments.

---

## 1. Decision Matrix: Primary Experiment vs. Sub-Experiment

| Decision Factor | Create Primary Experiment (`EXP_XX`) | Create Sub-Experiment (`SUB_YY`) |
| :--- | :--- | :--- |
| **Core Objective** | Major research pipeline (e.g. Mask2Former pixel-wise baseline vs Surgical GeMap geometric prompt learning) | Micro-iteration (e.g. original TopoNet paper metric reproduction or hyperparameter search) |
| **Codebase Scope** | Requires distinct model files, scripts, or multi-step pipeline architectures | Reuses parent `models/` and `scripts/`, running metric reproduction or ablation studies |
| **Directory Location** | `experiments/EXP_XX_<name>/` | `experiments/EXP_XX_<name>/sub_experiments/SUB_YY_<name>/` |

---

## 2. Primary Experiments Overview in Workspace

- `experiments/EXP_01_mask2former_pixelwise/`: Pixel-wise landmark segmentation baseline & Mask2Former attention visualizations (includes `SUB_01_toponet_paper_reconstruction`).
- `experiments/EXP_02_surgical_gemap/`: Main novel contribution applying geometric prompt learning (`surgical_gemap.py`) and depth prompts (`depth_anything_v2`) to surgical landmark detection.
- `experiments/EXP_09_patch_vector_vit/`: Patch-Level Bézier Vector Vision Transformer (predicts continuous cubic Bézier control points per ViT patch and merges via Global Coordinate Shift).

---

## 3. Workflow Steps for New Experiments

1. Inspect existing folders in `experiments/` to auto-increment ID (`EXP_03`, etc.).
2. Scaffold directory tree (`models/`, `configs/`, `scripts/`, `results/`, `sub_experiments/`).
3. Build models in `models/` by importing clean reference code from `/repos/` using `sys.path.append()`.
4. Generate `EXP_MANIFEST.md` and `Run_Commands.md`.
