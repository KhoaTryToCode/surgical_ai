# Lossless Paper Extraction Prompt Template

Extract and summarize the provided research paper thoroughly. Ensure zero technical details are lost.

> [!IMPORTANT]
> **STRICT FACTUAL GROUNDING & ZERO HALLUCINATION RULE:**
> Extract ONLY facts, formulas, and parameters explicitly stated in the paper or official codebase. If a hyperparameter, loss weight, dataset split, or dimension is NOT specified, explicitly state `[Not Specified in Paper]` or request clarification. Never fabricate or guess unstated parameters.

---

## 1. Core Metadata
- **Title:**
- **Authors & Institution:**
- **Publication Year / Venue:**
- **Official Code Repository Link:**
- **Primary Contribution (1-2 sentences):**

---

## 2. Theoretical Architecture & Mathematical Formulations
- **Core Novelty / Concept:**
- **Mathematical Equations:** (LaTeX formulations for TopoNet topological continuity loss, clDice, geometric prompt loss)
- **Loss Function(s):** (Exact mathematical formulation and weightings)
- **Tensor Dimensions & Data Flow:** (Input dimensions, intermediate feature maps, output dimensions)

---

## 3. Training Setup & Hyperparameters
- **Optimizer & Schedule:** (e.g., AdamW, learning rate, warmup steps)
- **Hyperparameter Specifics:** (Batch size, epoch/step count, weight decay, dropout rate)
- **Datasets Used & Preprocessing Rules:** (Surgical video resolution, landmark coordinate definitions)

---

## 4. Benchmark Results & Key Claims
- **Baseline Comparison:**
- **Key Metrics Achieved:** (Mean Radial Error MRE, Landmark Detection Accuracy %, Inference FPS)
- **Ablation Study Takeaways:**

---

## 5. Implementation & Codebase Mapping
- **Cloned Repo Reference:** Path inside `/repos/<repo_name>/`.
- **Core Files in Reference Repo:** List specific files/classes in `/repos/` that implement the core mathematical claims.
- **Experiment Adaptation Strategy:** How can we import or modularly subclass this architecture inside `experiments/<EXP_ID>/models/`?
