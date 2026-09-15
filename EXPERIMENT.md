# Surgical AI — Systematic Experimentation Protocol & Governance (`EXPERIMENT.md`)

---

## 1. Core Philosophy & Scientific Standards

Every experiment conducted in this workspace is governed by a strict, uniform scientific contract:
1. **Zero Blind Executions:** No script or training job is launched on Kaggle or locally without a pre-defined **Input**, **Output**, **Hypothesis**, and **Quantitative Success Criteria**.
2. **Exact Baseline Fidelity:** All reference baselines (TopoNet, BCRNet, Mask2Former) must directly reflect published literature configurations, utilizing clean, authentic clones in `/repos/` with identical hyperparameters, dataset splits, and loss formulations.
3. **Documentation & Telemetry Parity:** Every training run must immediately record its final epoch metrics, best checkpoint score, and per-class breakdown into the active experiment's `EXP_MANIFEST.md` and the master results ledger.
4. **Structured Progression:** Each micro-iteration changes **exactly one variable** (e.g., +Depth, +DSConv, +clDice) so that empirical gains or regressions can be isolated without confounding factors.

---

## 2. The 4-Part Pre-Flight Protocol

Before initiating any training or ablation run, the following specification must be documented and verified:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       PRE-FLIGHT RUN SPECIFICATION                         │
│                                                                            │
│  1. INPUT SPECIFICATION:                                                   │
│     - Modalities: RGB (3-ch) vs RGB-D (4-ch with AdelaiDepth / DA-V2)      │
│     - Resolution: (e.g., 1024x1024 or 512x512)                             │
│     - Data Split: L3D (921 Train / 122 Val / 109 Test)                     │
│     - Batch Size: Micro-batch size + accumulation steps = Effective batch  │
│     - Solver: Optimizer, learning rate, schedule, weight decay, epochs     │
│                                                                            │
│  2. OUTPUT SPECIFICATION:                                                  │
│     - Output Format: 2D probability map vs continuous Bézier control points│
│     - Evaluation Metrics: Macro DSC, IoU, ASSD (px), per-class metrics:    │
│       * Anterior Ridge DSC                                                 │
│       * Liver Silhouette DSC                                               │
│       * Falciform Ligament DSC                                             │
│     - Artifacts: Best checkpoint (.pth), evaluation CSV, visual overlays   │
│                                                                            │
│  3. CORE HYPOTHESIS:                                                       │
│     - The specific physiological, topological, or architectural mechanism  │
│       being evaluated (what question does this run answer?).               │
│                                                                            │
│  4. EXPECTED RESULT & SUCCESS CRITERIA:                                    │
│     - Benchmark target from literature or preceding ablation stage.        │
│     - Quantitative threshold for passing vs failing.                       │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Workspace Layout & Code Organization

```
Surgical AI/
├── EXPERIMENT.md              # Master experimentation protocol & ledger (this document)
├── _config/                   # Global execution flags and environment setups
├── repos/                     # IMMUTABLE official reference repositories (TopoNet, GeMap, BCRNet)
├── shared/                    # Reusable shared utilities (evaluation metrics, visualization)
├── data/                      # Local dataset storage (L3D, depth maps, visual diagnostics)
├── checkpoints/               # Model weights and trained checkpoints
└── experiments/
    ├── RAW_EXP/               # Archived exploratory experiments (EXP_01 to EXP_13)
    └── EXPERIMENT_1/          # Active: TopoNet Paper Replication & Systematic Ablations
        ├── EXP_MANIFEST.md    # Experiment manifest and run history log
        ├── Run_Commands.md    # Copy-pasteable execution commands for local & Kaggle
        ├── configs/           # Experiment configuration YAMLs
        ├── models/            # Modular PyTorch extensions (importing from /repos/)
        ├── scripts/           # Training and evaluation runners
        └── results/           # Saved metrics JSONs, diagnostic CSVs, and figures
```

---

## 4. Kaggle Environment Execution Contract

To ensure scripts run smoothly across local macOS machines and cloud Kaggle environments:
1. **Dynamic Path Resolution:** Always resolve dataset paths dynamically with fallback logic:
   ```python
   def get_data_path():
       kaggle_path = "/kaggle/input/laparoscopic-liver-landmarks/L3D"
       local_path = "data/L3D"
       return kaggle_path if os.path.exists(kaggle_path) else local_path
   ```
2. **Deterministic Reproducibility:** Every runner must explicitly seed `random`, `numpy`, and `torch` (CUDA & CuDNN deterministic flags enabled).
3. **Graceful Memory Management:** Effective batch size 4 must be achieved via gradient accumulation if GPU VRAM (e.g., 16GB Tesla T4) limits native batch size.
4. **Persistent Artifact Logging:** Ensure evaluation outputs (`.csv` summaries and sample overlay `.png`s) are written to `/kaggle/working/results/` for immediate download.

---

## 5. Master Comparative Results Ledger

Every completed run across all experiments will be permanently recorded here:

| Run ID | Experiment | Architecture / Setting | Split | Epochs | Macro DSC | Macro IoU | ASSD (px) | Ridge DSC | Sil DSC | Falc DSC | Status / Target |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Run_1.1` | EXPERIMENT_1 | Baseline (Vanilla CNN + Concat + L_dice) | Val (122) | 100 | -- | -- | -- | -- | -- | -- | *Target: 56.36% (Table 2)* |
| `Run_1.3` | EXPERIMENT_1 | w/o L_cl (BTF + Snake + L_per only) | Val (122) | 100 | -- | -- | -- | -- | -- | -- | *Target: 58.74% (Table 2)* |
| `Run_1.2` | EXPERIMENT_1 | w/o L_per (BTF + Snake + L_cl only) | Val (122) | 100 | -- | -- | -- | -- | -- | -- | *Target: 58.82% (Table 2)* |
| `Run_1.4` | EXPERIMENT_1 | w/o L_per & L_cl (BTF + Snake + L_dice) | Val (122) | 100 | -- | -- | -- | -- | -- | -- | *Target: 57.48% (Table 2)* |
| `Run_1.5` | EXPERIMENT_1 | w/o BTF (Snake + Concat + L_cl + L_per) | Val (122) | 100 | -- | -- | -- | -- | -- | -- | *Target: 58.75% (Table 2)* |
| `Run_1.0_val` | EXPERIMENT_1 | Full TopoNet (BTF + Snake + L_cl + L_per)| Val (122) | 100 | -- | -- | -- | -- | -- | -- | *Target: 59.79% (Paper), ~66.0% (Canvas Fix)* |
| `Run_1.0_test`| EXPERIMENT_1 | Full TopoNet SOTA Replication | Test (109)| 100 | -- | -- | -- | -- | -- | -- | *Target: 65.19% (Table 1 SOTA)* |
