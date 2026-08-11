# Experiment Manifest: EXP_02 — Surgical GeMap (Novel Geometric & Depth Prompt Learning)

---

## 1. Research Context & Core Hypothesis
- **Paper References:** [`papers/2024_gemap_depth_prompt/summary.md`](../../papers/2024_gemap_depth_prompt/summary.md) & [`papers/2024_toponet_liver_landmarks/summary.md`](../../papers/2024_toponet_liver_landmarks/summary.md)
- **Repo References:** [`repos/GeMap/`](../../repos/GeMap/) & [`repos/TopoNet/`](../../repos/TopoNet/)
- **Primary Objective:** Develop a novel surgical landmark detection architecture (`surgical_gemap.py`) that integrates 3D geometric prompt mapping from GeMap with depth prompts (`depth_anything_v2`) and topological vector loss constraints (`vector_losses.py`).
- **Hypothesis:** Fusing depth-driven geometric prompts with topological constraints significantly improves laparoscopic liver landmark localization under severe surgical occlusion and camera deformation.

---

## 2. Architecture & Codebase Design
- **Custom Models (`models/`):**
  - **`surgical_gemap.py`**: Custom Surgical GeMap architecture.
  - **`vector_losses.py`**: Topological vector loss terms.
  - **`befusion.py`**: Multi-modal feature fusion module.
  - **`context_modules.py`**: Spatial context modules.
  - **`depth_anything_v2/`**: Depth prompt encoder.
  - **`cldice/`**: Centerline Dice loss.
  - **`DSCNet/`**: Direction-Sensitive Connectivity Network.
- **Custom Scripts (`scripts/`):** `train_surgical_gemap.py`, `evaluate_all_models.py`, `analyze_results.py`, `setup_kaggle.sh`.

---

## 3. Configuration & Parameters
- **Active Config File:** [`configs/toponet.yaml`](configs/toponet.yaml)
- **Datasets Used:** Laparoscopic liver frames (`data/laparoscopic_liver/`) + Depth maps (`data/depth_maps/`).

---

## 4. Run History Log

| Run ID | Date | Device / Hardware | Config File | Key Results / Metrics | Notes / Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Run_01` | 2026-08-10 | Local (macOS MPS) | `configs/toponet.yaml` | Initial training run | Model pipeline verified |
| `Run_02` | 2026-08-10 | Kaggle (Dual T4 DDP) | `configs/toponet.yaml` | MRE: Z.ZZ mm, Acc: ZZ.Z% | Full Surgical GeMap benchmark run |
