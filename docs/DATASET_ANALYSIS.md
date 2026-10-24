# Benchmark Dataset Analysis & Geometric Disparity Discovery

This document details the surgical characteristics of the Laparoscopic Liver Landmark (**L3D**) benchmark, the empirical discovery of benchmark pose-inversion disparity, and the geometric challenges of intraoperative soft-tissue tracking.

---

## 1. Benchmark Cohort Overview

Our empirical investigations focus on the canonical Laparoscopic Liver Landmark benchmark (**L3D**) and its high-resolution extension (**L3D-2K**). The L3D dataset captures intraoperative video frames during laparoscopic hepatectomy across multiple clinical centers.

### Canonical Partitioning

The standard L3D benchmark comprises **1,152 frames** distributed across 39 surgical patient procedures:

| Split | Frames | Patient Procedures | Composition & Notes |
|:---|:---:|:---:|:---|
| **Training Set** | 921 | 33 Patients | Dominated by canonical diaphragmatic orientations |
| **Validation Set** | 122 | 3 Patients | Highly concentrated on **Patient 40 (101/122 frames, 82.8%)** |
| **Frozen Test Set** | 109 | 4 Patients | Unseen subjects: Patient 41 (84), Patient 31 (14), Patient 21 (7), Patient 51 (4) |
| **Total** | **1,152** | **39 Patients** | Canonical clinical benchmark |

### Landmark Class Definitions

1. **Class 1 — Anterior Liver Ridge (Green):** The inferior curvilinear border dividing the diaphragmatic and visceral hepatic surfaces.
2. **Class 2 — Liver Silhouette Contour (Blue):** The perimeter outline of the exposed liver lobes against abdominal cavity tissue.
3. **Class 3 — Falciform Ligament (Red):** The peritoneal fold attaching the liver to the anterior abdominal wall.

---

## 2. Discovery of Benchmark Pose-Inversion Disparity

Through systematic geometric auditing of anatomical landmark coordinates and mechanical grasper actions, we identified a critical **geometric coverage disparity** between the benchmark splits:

### Quantitative Disparity

* **Training Set (Canonical Dominated):**
  * 92.0% of training frames (847 frames) exhibit standard diaphragmatic views where the anterior liver ridge rests naturally in the inferior canvas.
  * Extreme surgical elevation—where mechanical grasper traction hoists the anterior liver ridge into the upper margin ($Y_{\text{ridge}} < 0.015$)—accounts for only **74 frames (8.0%)**.

* **Validation Set (Inversion Dominated):**
  * In contrast, the validation cohort contains **19 inverted frames (15.6%)**—a **1.95× higher inversion rate** than the training set.
  * **Patient 40** accounts for 15 of these extreme traction frames, exposing models to soft-tissue configurations virtually absent from training distributions.

```
Training Split (8.0% Inverted)           Validation Split (15.6% Inverted)
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│ Diaphragm (Top)                 │      │ Diaphragm (Top)                 │
│                                 │      │ Anterior Ridge (Hoisted by Grasp│
│ Anterior Ridge (Inferior Canvas)│      │                                 │
└─────────────────────────────────┘      └─────────────────────────────────┘
```

---

## 3. Catastrophic Spatial Prior Inertia in Static Baselines

When evaluated on Patient 40, standard query-based models (such as **Mask2Former Swin-Tiny**) suffer from severe **spatial prior inertia**:

* The network assumes anatomical landmarks follow canonical spatial positions learned from the 92% canonical training frames.
* Under severe traction, the model hallucinates phantom liver contours along the bottom edge, completely missing the hoisted organ boundary.
* On Frame 008940 of Patient 40:
  * Baseline Macro Dice collapses to **7.56%**
  * Average Symmetric Surface Distance (ASSD) surges to **211.77 px**

---

## 4. Pose Re-Stratification Formulation

To test whether representation starvation caused this collapse, we formulated a 1-to-1 patient re-partition:
* **Patient 40** moved into the training pool
* **Patient 38** moved into the validation pool
* **Official Test Set** remained **100% frozen and untouched**

### Stratified Distribution:
* **Stratified Train:** 923 frames (83 inverted, **8.99% rate**)
* **Stratified Val:** 120 frames (10 inverted, **8.33% rate**)
* **Official Test:** 109 frames (untouched)

### Outcome:
While pose re-stratification exposed the network to inverted poses during training (accelerating validation convergence to 65.50% DSC), error auditing revealed persistent failure on unobserved patient morphologies (e.g., Patient 32, 38). Static 2D models can interpolate within observed geometric manifolds, but lack the inductive bias needed to extrapolate non-rigid deformation across unobserved subjects without temporal continuity.

---

## 5. Frozen Test Set Breakdown: Patient 41 vs. Patient 31

The official 109-frame test set further confirms that geometric deformation is the primary determinant of model accuracy:

| Test Patient | Frames | Surgical Character | Macro Dice | ASSD (px) |
|:---|:---:|:---|:---:|:---:|
| **Patient 41** | 84 | Canonical diaphragmatic views with graspers | **76.70%** | **11.60 px** |
| **Patient 21** | 7 | Canonical orientation, stable camera | **73.68%** | **23.57 px** |
| **Patient 51** | 4 | Restricted operative field of view | **62.88%** | **18.79 px** |
| **Patient 31** | 14 | Severe mechanical elevation & traction | **44.97%** | **32.89 px** |

The performance on Patient 41 demonstrates that the architecture has mastered canonical laparoscopic landmark tracking (11.6 px boundary accuracy). The degradation on Patient 31 confirms that residual errors stem from out-of-distribution non-rigid tissue deformation rather than network capacity.
