# Master Workspace Router (Layer 1)

## Purpose & Navigation Protocol
When starting a task, inspect the user's intent below and follow the designated stage contract or reference guide:

---

### 1. Intake & Extract a Surgical AI Research Paper
* **Target Stage:** Literature Processing
* **Stage Contract:** Read `papers/CONTEXT.md`
* **Action:**
  1. Read PDF from `papers/inbox/` or download online reference into `papers/<concise_paper_name>/paper.pdf`.
  2. Parse technical details using `papers/templates/extraction_prompt.md`.
  3. If official code is cited, shallow-clone into `repos/<repo_name>/` using `git clone --depth 1` and `rm -rf repos/<repo_name>/.git`.
  4. Save lossless structured summary to `papers/<concise_paper_name>/summary.md`.

---

### 2. Run / Extend Mask2Former Pixel-Wise Baseline (`EXP_01`)
* **Target Stage:** Primary Experiment 01 — Mask2Former Pixel-Wise Baseline
* **Location:** `experiments/EXP_01_mask2former_pixelwise/`
* **Action:**
  1. Inspect `experiments/EXP_01_mask2former_pixelwise/EXP_MANIFEST.md` and `Run_Commands.md`.
  2. Run ablation scripts in `experiments/EXP_01_mask2former_pixelwise/scripts/ablation_mask2former.py`.
  3. Generate attention visualizations (`visualize_attention.py`).
  4. For TopoNet paper metric verification, consult sub-experiment:
     - `experiments/EXP_01_mask2former_pixelwise/sub_experiments/SUB_01_toponet_paper_reconstruction/`

---

### 3. Run / Extend Surgical GeMap Landmark Detection (`EXP_02`)
* **Target Stage:** Primary Experiment 02 — Surgical GeMap (Novel Geometric & Depth Prompt Learning)
* **Location:** `experiments/EXP_02_surgical_gemap/`
* **Action:**
  1. Inspect `experiments/EXP_02_surgical_gemap/EXP_MANIFEST.md`.
  2. Load model architecture from `experiments/EXP_02_surgical_gemap/models/` (`surgical_gemap.py`, `vector_losses.py`, `depth_anything_v2`).
  3. Execute training script: `experiments/EXP_02_surgical_gemap/scripts/train_surgical_gemap.py`.
  4. Run cross-model evaluation: `experiments/EXP_02_surgical_gemap/scripts/evaluate_all_models.py`.

---

### 4. Lookup Execution Commands, Environments, or Dataset Registries
* **Target Stage:** Reference Specs Lookup
* **Action:**
  - Shell/Kaggle CLI flags: Read `_config/execution_commands.md`
  - Surgical dataset & depth map registries: Read `_config/environment_setup.md`
  - PyTorch coding & import rules: Read `_config/code_conventions.md`
