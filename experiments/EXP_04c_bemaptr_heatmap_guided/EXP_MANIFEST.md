# EXP_04c: Heatmap-Guided Vector Prompt Learning (Surgical-BeMapTR v4)

## Abstract & Hypothesis
In EXP_04b, Branch B (Auxiliary Pixel Edge Segmentation) achieved 100% pixel-perfect localization of surgical landmarks. However, Branch A (Vector Transformer Queries) operated in parallel, sampling un-guided feature maps and predicting false-positive lines over non-landmark abdominal fat.

**Hypothesis:** Modulating FPN feature maps physically during the forward pass ($P_2^{\text{guided}} = P_2 \cdot (1.0 + \text{sigmoid}(\text{aux\_edge\_logits}))\big$) and refining reference points via relative offsets ($\text{ref\_pts}_l = \text{sigmoid}(\text{logit}(\text{ref\_pts}_{l-1}) + \Delta(x,y)_l)$) will force Deformable Attention queries to sample 100% of their features from true landmark edges, eliminating false-positive lines and snapping vector splines directly onto anatomical landmarks with zero spatial translation shift.

---

## Model Architecture (`SurgicalBeMapTRGuided`)
- **Backbone:** Swin-Tiny + Multi-Scale FPN ($P_2, P_3, P_4, P_5$)
- **Branch B (Edge Heatmap Head):** 4-channel $256 \times 256$ dense edge prediction head.
- **Forward Pass Feature Modulation:** $P_2^{\text{guided}} = P_2 \cdot (1.0 + \mathbf{M}_{\text{edge}})$.
- **Branch A (Guided Vector Decoder):** 6-layer Deformable Cross-Attention decoder sampling from $P_2^{\text{guided}}$.
- **Relative Reference Anchoring:** Blends Spatial Centroid Voting (70%) with Relative Logit Offset Anchoring (30%).
- **Curve Restoration:** Piecewise Cubic Bézier Matrix ($P = B \cdot C$, $10 \to 20$ points).

---

## Output Checkpoints
- `checkpoints_bemaptr_guided/best_surgical_bemaptr_guided.pth`
- `checkpoints_bemaptr_guided/latest_surgical_bemaptr_guided.pth`
