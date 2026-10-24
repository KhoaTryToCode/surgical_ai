# Project Changelog & Research Milestones

All notable milestones and architectural progressions in this project are documented below.

---

## [Phase 4: Video Landmark Segmentation & Benchmark Generalization] - October 2026

### Added
- **MinVIS Video Framework (`experiments/temporal`)**: Minimal Video Instance Segmentation adapting Mask2Former to laparoscopic video sequences via Hungarian temporal bipartite matching.
- **Canonical 5-Fold Cross-Validation (`config/5fold_patient_splits.json`)**: Locked patient-disjoint 5-fold split across 35 pool patients (1,043 frames) to benchmark temporal models without patient leakage.
- **Kaggle Git Runner (`experiments/temporal/notebooks/Kaggle_Temporal_Runner.ipynb`)**: Lightweight runner notebook executing directly from git to guarantee zero code drift.
- **Automated Results Aggregator (`experiments/temporal/aggregate.py`)**: Automatic discovery and collation of 5-fold and test evaluation outputs into master Markdown and CSV leaderboards.

---

## [Phase 3: Stratified Temporal Transformer] - October 2026

### Added
- **Pose-Stratified Temporal Transformer (EXP 16)**: Combined pose-stratified training distributions with 3-frame causal prototype cross-attention.
- **New Benchmark SOTA**:
  - **74.31% Micro Dice** (+9.12% over TopoNet SOTA)
  - **71.92% Macro Dice** (+2.35% over BCRNet SOTA)
  - **15.37 px ASSD** (reducing boundary error by >50%)
- **Patient-Level Auditing**: Comprehensive breakdown proving Patient 41 reaches **76.70% Macro Dice** and **11.60 px ASSD**, while identifying severe out-of-distribution traction in Patient 31 (**44.97% Macro Dice**).

---

## [Phase 2: Topological Anchor Steering & Disparity Discovery] - September 2026

### Discovered
- **Benchmark Pose-Inversion Disparity**: Discovered that extreme surgical traction (`Y_ridge < 0.015`) is 1.95x more prevalent in validation (15.6%) than in training (8.0%), with Patient 40 containing 15 flipped frames.
- **Spatial Prior Inertia Collapse**: Audited vanilla Mask2Former Swin-Tiny failure on Patient 40 Frame 008940 (collapsed to 7.56% DSC, 211.77 px ASSD).

### Added
- **Topological Anchor Steering (EXP 13-14)**: Deterministic 4-junction biological anchor derivation (`J_top`, `J_bottom`, `J_lat_r`, `J_lat_l`) with query steering, boosting Macro Dice to **66.86%** and ASSD to **22.64 px**.
- **Pose-Stratified Model (EXP 14)**: Re-stratified patient splits to eliminate representation starvation on inverted poses.

---

## [Phase 1: Baselines & Ablation Studies] - August - September 2026

### Added
- **TopoNet Replication (EXP 1)**: Authentic replication of published MICCAI 2025 TopoNet (59.79% Macro Dice on Val, 65.19% on Test) and systematic component ablations (without `L_per`, without `L_cl`, without BTF).
- **Mask2Former Baselines (EXP 2)**: Verification of Swin-Tiny transformer baseline achieving 65.73% Macro Dice and 28.43 px ASSD with faster runtime than TopoNet's topology losses.
- **Bézier Continuous Landmark Representations (EXP 3-4)**: Parametric curve head formulations evaluated for continuous polyline regression.
