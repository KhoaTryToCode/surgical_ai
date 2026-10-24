# Experimentation Protocol & Governance

This document outlines the scientific standards, pre-flight verification protocols, evaluation metrics, and reproducibility guidelines governing all models and experiments in this repository.

---

## 1. Core Philosophy & Scientific Standards

Every experiment conducted in this repository adheres to a strict scientific contract:

1. **Zero Blind Executions:** No training job is launched without a pre-defined hypothesis, exact input/output specification, and quantitative success threshold.
2. **Exact Baseline Fidelity:** All reference baselines (TopoNet, BCRNet, Mask2Former) strictly reflect published literature configurations, utilizing identical hyperparameters, dataset splits, stroke thicknesses, and loss formulations.
3. **Canonical Split Protection:** The frozen official test set (109 frames across 4 unseen patients: Patient 21, 31, 41, 51) is never accessed during training, architectural searches, or hyperparameter selection. Cross-validation uses the locked 5-fold patient-disjoint split (`config/5fold_patient_splits.json`).
4. **Structured Progression:** Each micro-iteration alters **exactly one variable** (e.g., query mechanism, spatial anchor loss, temporal window length) so that empirical gains or regressions can be isolated without confounding factors.

---

## 2. The 4-Part Pre-Flight Specification

Before launching any model training or evaluation run, the following specification is verified:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       PRE-FLIGHT RUN SPECIFICATION                         │
│                                                                            │
│  1. INPUT SPECIFICATION:                                                   │
│     - Modalities: Laparoscopic RGB (3-ch)                                  │
│     - Resolution: 1024x1024 (native canvas)                                │
│     - Split Definition: Canonical L3D (921 Train / 122 Val / 109 Test)     │
│       or 5-Fold Patient-Disjoint Cross-Validation (1,043 pool frames)      │
│     - Optimization: AdamW, Cosine Annealing, explicit gradient accumulation│
│                                                                            │
│  2. OUTPUT SPECIFICATION:                                                  │
│     - Format: 3-class mutually exclusive landmark segmentation mask:       │
│       * Class 1: Anterior Liver Ridge                                      │
│       * Class 2: Liver Silhouette Contour                                  │
│       * Class 3: Falciform Ligament                                        │
│     - Target Canvas Stroke Width: 35 px (matching clinical standard)       │
│                                                                            │
│  3. METRIC DEFINITIONS:                                                    │
│     - Macro Dice: Mean of per-class Dice scores across valid landmarks     │
│     - Micro Dice: Global voxel-level intersection over union sum           │
│     - ASSD (Average Symmetric Surface Distance): Boundary error in pixels  │
│     - Per-Patient Breakdown: Separate reporting for challenging patients   │
│                                                                            │
│  4. SUCCESS THRESHOLDS:                                                    │
│     - Superiority over static baseline on extreme traction frames          │
│     - Convergence verification via tracking loss curves                    │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Evaluation Metrics & Formulation

### Metric Computation Standards

All metrics are evaluated in the native image canvas space ($1024 \times 1024$) with class mutual exclusivity enforced via argmax over prediction logits:

1. **Dice Similarity Coefficient (DSC):**
   $$\text{DSC}_c = \frac{2 |P_c \cap G_c|}{|P_c| + |G_c| + \epsilon}$$
   where $P_c$ is the predicted binary mask for class $c$, $G_c$ is ground truth, and $\epsilon = 10^{-6}$.
   - **Macro Dice:** Unweighted mean over present classes: $\frac{1}{C}\sum_{c=1}^C \text{DSC}_c$
   - **Micro Dice:** Aggregated pixel-wise true positive ratio across all classes.

2. **Average Symmetric Surface Distance (ASSD):**
   Given boundary point sets $S(P_c)$ and $S(G_c)$:
   $$\text{ASSD}_c = \frac{1}{|S(P_c)| + |S(G_c)|} \left( \sum_{x \in S(P_c)} \min_{y \in S(G_c)} \|x - y\|_2 + \sum_{y \in S(G_c)} \min_{x \in S(P_c)} \|y - x\|_2 \right)$$
   Reported in canvas pixels.

3. **Per-Class Tracking:**
   Because the falciform ligament is absent in some surgical fields of view, classes without ground truth presence in a frame are handled cleanly according to standard benchmark protocols.

---

## 4. Multi-Platform Execution Standards

To maintain identical behavior across local workstations (macOS Apple Silicon / MPS) and cloud accelerators (Kaggle dual Tesla T4 / CUDA):

1. **Deterministic Seeding:**
   ```python
   def seed_everything(seed=42):
       random.seed(seed)
       os.environ['PYTHONHASHSEED'] = str(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       if torch.cuda.is_available():
           torch.cuda.manual_seed_all(seed)
           torch.backends.cudnn.deterministic = True
   ```

2. **Dynamic DataLoader Concurrency:**
   On macOS/MPS, `num_workers=0` is automatically enforced to prevent multiprocessing semaphore deadlocks, while Linux/CUDA environments leverage `num_workers=4` for pipeline throughput.

3. **Zero Implementation Drift:**
   Cloud runs execute directly from the git repository using git clone / pull runners, ensuring that metrics, models, and data pipelines are strictly identical between local development and cloud verification.
