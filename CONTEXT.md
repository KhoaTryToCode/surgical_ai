# Surgical AI — Master Project Context & Engineering Protocols

> **Audience & Purpose:**  
> This document is the single source of truth for all human developers and AI pair-programming agents working in the `Surgical AI` workspace. It documents the historical evolution of the codebase, current experiment protocols, compute resource hierarchies (10 Kaggle accounts vs. HPC server), strict server operating rules, and exact blueprints for onboarding future models (e.g., Mask2Former, BCRNet, Surgical GeMap) without ambiguity.

---

## 1. Project Evolution & Codebase Organization

### 1.1 Historical Context: The `RAW_EXP/` Consolidation
Early in this project, rapid iterative prototyping produced over a dozen disjointed experiments (`EXP_01` through `EXP_13`), covering pixel-wise segmentation, Bézier curves, LSTM sequential decoders, SuperToken ViTs, and early MapTR/GeMap adaptations. 

To prevent directory bloat and maintain a clean production architecture:
1. **Archive Consolidation:** All legacy explorations were systematically archived into [`experiments/RAW_EXP/`](experiments/RAW_EXP/). These serve strictly as historical reference code.
2. **Standardized Active Experiments:** Production replications and new architectural contributions now live in clean, isolated top-level experiment directories:
   - [`experiments/EXPERIMENT_1/`](experiments/EXPERIMENT_1/): TopoNet (MICCAI 2025) Paper Replication & Systematic Ablation Study.
   - `experiments/EXPERIMENT_2/`: Surgical GeMap (Novel Geometric & Depth-Driven Landmark Prompt Learning).
3. **Immutable Cloned Repositories:** External research repositories ([`repos/TopoNet/`](repos/TopoNet/), [`repos/GeMap/`](repos/GeMap/)) are clean, immutable upstream clones. **Never edit code inside `repos/`**. Any extensions or custom layers must live inside `experiments/<EXP_ID>/models/` and import from `repos/` via `sys.path`.

---

## 2. Compute Resource Hierarchy: Kaggle First, Server Second

We possess two distinct compute platforms. To maximize speed, minimize queue times, and conserve HPC server compute, follow this **strict priority hierarchy**:

```
[New Training / Ablation Task]
       │
       ▼
[Can it run on Kaggle?] ──► YES (Preferred) ──► [Primary: 10 Kaggle Accounts]
       │                                         - Shard into parallel independent notebooks
       │                                         - Free Tesla T4 / P100 GPUs (16 GB)
       │                                         - 10x parallel throughput
       ▼ NO (Computationally complex)
[Secondary: Remote HPC Server (gpu-a240)]
       - Requires native C++ pybind compilation (e.g. Betti Matching 3D)
       - High VRAM requirements exceeding 16 GB
       - Multi-day sequential Slurm runs
```

### 2.1 Primary Compute: 10 Parallel Kaggle Accounts
- **Capacity:** We have **10 independent Kaggle accounts**.
- **Rule:** **Always prioritize Kaggle first.** If an experiment can be run on Kaggle, run it on Kaggle.
- **Parallel Sharding Strategy:**
  - Break multi-configuration ablation studies into dedicated, self-contained Jupyter notebooks (e.g., in `experiments/EXPERIMENT_1/notebooks/`, runs 1 through 6 run simultaneously across independent accounts).
  - Use Kaggle's free GPU accelerators (Tesla T4 16GB or P100 16GB).
  - Each account provides a 12-hour session limit and 30 GPU hours/week. By sharding across 10 accounts, we achieve up to **10x parallel throughput**.
- **Kaggle Optimizations:**
  - **Direct Precomputed Depth:** Never run on-the-fly ViT-B depth inference inside training loops. Load precomputed depth maps directly from the Kaggle dataset (`khoale05/l3d-depth`), dropping per-epoch time from 25 min down to ~3 min.
  - **Memory Safety:** Micro-batch size 1 with gradient accumulation 4 (`1 x 4 = 4`), Automatic Mixed Precision (`torch.amp.autocast('cuda')`), and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### 2.2 Secondary Compute: Remote Slurm HPC Cluster (`gpu-a240`)
- **When to Use the Server:**
  - Experiments requiring native C++ extensions that cannot easily build in Kaggle kernels (e.g., Betti Matching 3D pybind modules).
  - High-VRAM autograd graphs that exceed Kaggle's 16GB limit.
  - Long multi-stage trainings exceeding Kaggle's 12-hour cutoff.

---

## 3. Remote Server Architecture & Strict Rules (`gpu-a240`)

### 3.1 Server Connection & Host Info
- **Host / IP:** `100.82.42.48` (`gpu-a240`) via NetBird VPN.
- **SSH User:** `khoalq` (authenticated via `~/.ssh/id_ed25519`).
- **Cluster Scheduler:** Slurm (`sbatch`, `squeue`, `scontrol`, `scancel`).

### 3.2 Physical Hardware Layout (2x NVIDIA A100-PCIE-40GB)
The server contains **two physical A100 40GB cards**, partitioned as follows:

| Physical GPU | Multi-Instance GPU (MIG) | Slurm GRES Identifier | Hardware Capacity | Usage Context |
| :--- | :---: | :--- | :---: | :--- |
| **GPU 0** | **Disabled** (Full Card) | `#SBATCH --gres=gpu:a100:1` | **40 GB VRAM**, 108 SMs, 1,555 GB/s | Dedicated card. Starts immediately when idle. Best for fast 20GB/40GB runs (~3 min/epoch). |
| **GPU 1** | **Enabled** (Partitioned) | `#SBATCH --gres=gpu:a100_3g.20gb:1` | **19.6 GB VRAM**, 42 SMs, 660 GB/s | **Only 1 slice exists on the machine.** Often occupied by lab colleagues. |
| **GPU 1** | **Enabled** (Partitioned) | `#SBATCH --gres=gpu:a100_2g.10gb:1` | **9.75 GB VRAM**, 28 SMs, 440 GB/s | **Two 10GB slices exist.** Usually free. Runs our 10GB conservative profile (~9 min/epoch). |

### 3.3 Strict Server CAN DO vs. CANNOT DO Rules

#### Storage & Filesystem:
- **CAN DO:** Store all data, environments, checkpoints, and code exclusively under `/data/khoalq/`:
  - Repository: `/data/khoalq/surgical_ai/`
  - Miniconda: `/data/khoalq/miniconda3/envs/surgical_ai/`
  - Datasets: `/data/khoalq/data/L3D/`
  - Checkpoints: `/data/khoalq/checkpoints/`
  - Slurm Logs: `/data/khoalq/logs/`
- **CANNOT DO:** **NEVER write, install packages, or store datasets in `/home/khoalq/`**. The home directory has a strict root disk quota; filling it will freeze the user's terminal and corrupt shell sessions.

#### Resource Inspection:
- **CAN DO:** Verify actual resource availability using live allocation queries:
  ```bash
  scontrol show node gpu-a240 | grep -E 'CfgTRES|AllocTRES'
  squeue
  ```
  Check `AllocTRES`: whatever is listed there is actively in use; whatever is omitted from `AllocTRES` is 100% free.
- **CANNOT DO:** **NEVER trust `sinfo` alone to see if a GPU is free**. `sinfo` displays the *static configured hardware*, not the dynamic real-time availability.

#### Slurm Script Standards:
- **CAN DO:** Always include robustness flags in `.sbatch` scripts:
  ```bash
  #!/usr/bin/env bash
  #SBATCH --partition=gpu
  #SBATCH --account=lab
  #SBATCH --qos=normal
  set -e
  set -o pipefail
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONUNBUFFERED=1
  ```
- **CANNOT DO:** **Never omit `set -o pipefail`**. Without it, piping python output to `tee` (`python ... | tee log.txt`) will mask python crashes, causing Slurm to falsely report `COMPLETED 0:0` when the job actually threw an OOM error.

#### Agent Execution Authorization:
- **CANNOT DO:** **AI agents must NEVER run `sbatch` or kill (`scancel`) user jobs automatically** unless the user explicitly commands it. The user must remain the master operator who executes cluster submissions. Always provide the verified commands for the user to run.

---

## 4. Active Experiment Protocols & Mathematical Standards

### 4.1 Required Artifacts Saved per Experiment Run
Every completed experiment must save the following standardized artifact bundle inside `experiments/<EXP_ID>/results/<RUN_NAME>/` (or `/data/khoalq/checkpoints/<RUN_NAME>/` on server):

1. **`best_model.pth`:** Model checkpoint with the highest validation Macro Dice score. Saved continuously during training (every 5 epochs) so no progress is lost if interrupted.
2. **`summary_metrics.json`:** Comprehensive final evaluation metrics:
   - Macro Mean Dice (`macro_dice`)
   - Mean IoU (`macro_iou`)
   - Mean Average Symmetric Surface Distance (`macro_assd` in px)
   - Class-wise Dice scores: `ridge_dice`, `sil_dice`, `falc_dice`
   - Inference Latency and FPS
   - Patient 40 validation subset metrics
3. **`validation_per_frame_results.csv`:** Frame-by-frame metric breakdown across all 122 validation frames.
4. **`test_per_frame_results.csv`:** Frame-by-frame metric breakdown across all 109 test frames.
5. **`patient_40_diagnostics/`:** Visual error analysis montages for all Patient 40 frames:
   - 4-panel image: [1] Original RGB, [2] Ground Truth Landmarks, [3] Model Predictions, [4] Error Map (False Positives in Red, False Negatives in Blue).

### 4.2 Documentation Parity Rule
Every code addition or experiment iteration MUST update two files simultaneously:
- **`EXP_MANIFEST.md`:** Documents the technical hypothesis, ablation table, hyperparameter configuration, and test verification ledger.
- **`Run_Commands.md`:** Step-by-step CLI commands and Kaggle/Slurm submission instructions.

### 4.3 Algorithmic Vigilance & Mathematical Parity
When implementing or modifying surgical models in this repository, follow these established scientific discoveries:

1. **Patient 32 4K Canvas Dynamic Sizing:**
   - Patient 32 frames in L3D Val are recorded at 4K resolution (2160 x 3840), whereas others are 1080p.
   - Always read `imageHeight` and `imageWidth` dynamically from annotation JSON metadata. Never hardcode canvas dimensions, or Patient 32 annotations will be drawn onto a 1080p canvas and clipped.
2. **Continuous Topological Losses (Centerline Dice / clDice):**
   - Soft skeletonization applies continuous min/max pooling across 40 iterations:
     `Erode(I) = -MaxPool2D(-I), Dilate(I) = MaxPool2D(I), Open(I) = Dilate(Erode(I))`
   - **Ground-Truth CPU Caching:** Ground-truth masks are static. Compute their skeletons once and cache them in CPU RAM. Eliminates 50% of skeletonization calls across 50 epochs with 0.000% mathematical difference.
   - **Chunked Gradient Checkpointing:** To avoid unrolling 40 iterations in GPU autograd at once (which takes >5.5 GB VRAM), checkpoint in chunks of 5 steps (`chunk_size=5`). Limits peak backward memory to <0.7 GB.
   - **Resolution Profiles:**
     - **Profile A (100% Strict Paper Parity):** Native 1024x1024 clDice. Requires 40GB GPU at `batch_size=2` (~15 min/epoch).
     - **Profile B (Scaled Topology Optimization):** Soft Dice is computed at native 1024x1024, while clDice is evaluated on continuous 512x512 probability maps. Peak VRAM drops to **7.51 GB (10GB GPU compatible)** or **14.68 GB (20GB GPU profile at batch size 2, ~3 min/epoch)**.

---

## 5. Agent Onboarding Guide: Plugging in a New Model

When a new AI agent takes over to implement a new architecture (e.g., Mask2Former pixel-wise, BCRNet Bézier landmark detection, or Surgical GeMap):

### Step-by-Step Blueprint:
1. **Assign a New Experiment Folder:** Create `experiments/EXP_XX_<model_name>/` (e.g., `experiments/EXP_02_surgical_gemap/`).
2. **Scaffold Directory Structure:**
   ```
   experiments/EXP_XX_<name>/
   ├── EXP_MANIFEST.md
   ├── Run_Commands.md
   ├── models/           # Modular PyTorch model definitions
   ├── utils/            # Custom dataset loaders, losses, and metrics
   ├── scripts/          # train_<name>.py, evaluate_<name>.py, sbatch runners
   └── results/          # Output checkpoints, JSON summaries, and visual diagnostics
   ```
3. **Dataset Access:** Always resolve paths using relative paths or environment registries:
   - Local Mac: `data/L3D/{Train,Val,Test}`
   - Kaggle: `/kaggle/input/datasets/khoatrytopublish/l3d-{train,val,test}/` and `/kaggle/input/l3d-depth/`
   - Remote HPC: `/data/khoalq/data/L3D/{Train,Val,Test}`
4. **Benchmark Verification:** Before launching heavy runs, write a `--smoke_test` flag in `train_<name>.py` that runs 1 forward/backward pass and 1 validation frame locally on CPU/MPS to verify autograd shapes and metrics.
5. **Execution Choice:**
   - If the model is standard PyTorch without custom C++: **Write Kaggle notebook runners first** and distribute across the 10 accounts.
   - If the model requires custom CUDA/C++ extensions or heavy VRAM: **Write a Slurm `.sbatch` script** targeting `/data/khoalq/`.
