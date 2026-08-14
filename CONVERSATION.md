# CONVERSATION.md — Mathematical Specs for Monocular 3D Surgical AI (EXP5)

## 1. Unprojecting 2D Pixels + Relative Depth to Canonical 3D Space

Given:
- Pixel coordinates: u in [0, W-1], v in [0, H-1]
- Relative depth map: d(u, v) in (0, 1] from Depth Anything V2
- Assumed standard laparoscopic camera field of view: FOV = 60 degrees

### Normalized Image Coordinates:
u_norm = (u - W / 2) / (W / 2)
v_norm = (v - H / 2) / (H / 2)

### Canonical Focal Length:
f_canon = 1.0 / tan(FOV / 2) = 1.0 / tan(30 degrees) approx 1.732

### Canonical 3D Pinhole Unprojection:
X_canon = (u_norm * Z_canon) / f_canon
Y_canon = (v_norm * Z_canon) / f_canon
Z_canon = Z_min + d(u, v) * (Z_max - Z_min)

---

## 2. Hierarchical Query Formulation (MapTR / BeMapTR 3D Adaptation)

Total Queries Q = N * K

For instance i in {1, ..., N} and point index j in {1, ..., K}:
Q_{i, j} = q_{instance, i} + q_{point, j}

Where:
- q_{instance, i} in R^C identifies landmark structure i (e.g. Falciform ligament, Liver edge, Vessel)
- q_{point, j} in R^C identifies sequential vertex j along the polyline path (from start 1 to end K)

---

## 3. Training Loss Formulations

Total Loss:
L_total = lambda_cls * L_cls + lambda_pos * L_pos + lambda_dir * L_dir + lambda_len * L_len

### 3.1 Hungarian Bipartite Matching Cost:
Cost(i, sigma(i)) = lambda_cls_match * L_cls_cost + lambda_pos_match * L_pos_cost

### 3.2 Bidirectional Smooth L1 Position Loss:
L_pos = min(
    sum_{j=1}^K SmoothL1( pred_p_{i, j} - gt_p_{sigma(i), j} ),
    sum_{j=1}^K SmoothL1( pred_p_{i, j} - gt_p_{sigma(i), K - j + 1} )
)

### 3.3 Cosine Edge Direction Loss:
pred_edge_j = pred_p_{i, j+1} - pred_p_{i, j}
gt_edge_j = gt_p_{sigma(i), j+1} - gt_p_{sigma(i), j}

L_dir = (1 / (K - 1)) * sum_{j=1}^{K-1} ( 1 - ( pred_edge_j . gt_edge_j ) / ( ||pred_edge_j|| * ||gt_edge_j|| + eps ) )
