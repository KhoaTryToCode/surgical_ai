# Experiment Manifest: EXP_01 — Mask2Former Pixel-Wise Segmentation Baseline

---

## 1. Research Context & Core Hypothesis
- **Paper Reference:** [`papers/2022_mask2former/summary.md`](../../papers/2022_mask2former/summary.md)
- **Repo Reference:** [`repos/TopoNet/`](../../repos/TopoNet/)
- **Primary Objective:** Establish a pixel-wise semantic landmark segmentation baseline using Mask2Former and evaluate attention map behavior across layers.
- **Hypothesis:** Standard pixel-wise segmentation without geometric or topological prompt constraints degrades on laparoscopic liver landmarks under occlusion.

---

## 2. Architecture & Codebase Design
- **Custom Scripts (`scripts/`):** `ablation_mask2former.py`, `visualize_attention.py`, `evaluate_worst_cases.py`.
- **Sub-Experiments (`sub_experiments/`):**
  - [`SUB_01_toponet_paper_reconstruction`](sub_experiments/SUB_01_toponet_paper_reconstruction/SUB_MANIFEST.md): TopoNet metric reproduction.
  - [`SUB_02_mask2former_rgbd`](sub_experiments/SUB_02_mask2former_rgbd/SUB_MANIFEST.md): Mask2Former RGB-D with Depth Anything V2 4th channel.
- **Shared Utilities Used:** `shared/utils/`.

---

## 3. Configuration & Parameters
- **Active Config File:** [`configs/base.yaml`](configs/base.yaml)

---

## 4. Run History Log

| Run ID | Date | Device / Hardware | Config File | Key Results / Metrics | Notes / Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Run_01` | 2026-08-10 | Local (macOS MPS) | `configs/base.yaml` | MRE: X.XX mm | Baseline Mask2Former ablation run |
| `Run_02` | 2026-08-10 | Local (macOS MPS) | `configs/base.yaml` | 9-layer attention PNG generated | Attention map visualization run |
