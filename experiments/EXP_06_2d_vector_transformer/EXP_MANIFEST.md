# EXP_06: Direct 2D Vector Space Transformer

## Identity & Core Philosophy
EXP_06 is an assumption-free, native **Direct 2D Vector Space Transformer** for Laparoscopic Liver Landmark Segmentation. It operates directly in normalized 2D image coordinates $(u, v) \in [0, 1]^2$ using **Learned Query Embeddings** (DETR/Mask2Former paradigm), eliminating depth estimation errors, camera calibration assumptions, and unprojection round-tripping.

## Architectural Key Components
1. **Backbone (`models/backbone_2d.py`):**
   - Mask2Former Swin-Tiny feature extractor generating multi-scale pyramid (Stride 4, 8, 16, 32).
   - 2D Sinusoidal Positional Encoding $PE_{2D}(u, v)$ (128 channels).
2. **Hierarchical 2D Masked Decoder (`models/transformer_decoder_2d.py`):**
   - 10 Learned Query Embeddings (`query_polylines = nn.Parameter(10, 20, 2)`).
   - 6 Hierarchical Decoder Layers with Intra-Curve Self-Attention, Masked Cross-Attention (Stride-4 $256 \times 256$), Dot-Product Mask Head, and 2D Displacement Point Heads $(\Delta u, \Delta v) \in [-0.2, 0.2]^2$.
3. **Loss Suite (`models/vector_losses_2d.py`):**
   - 2D Bipartite Hungarian Matcher (Focal + Bidirectional 2D L1).
   - Multi-Class Focal Loss ($L_{\text{cls}}$, $\gamma=2.0, \alpha=0.25$).
   - Bidirectional 2D L1 Distance ($L_{\text{pos}}$).
   - 2D Cosine Tangent Orientation ($L_{\text{tan}}$).
   - 1D Discrete Laplacian Curvature Regularizer ($L_{\text{curv}}$).
   - Auxiliary 2D Mask BCE + Dice Loss ($L_{\text{mask}}$).

## Directory Structure
- `configs/exp06_config.py`: Global hyperparameters and path resolution.
- `models/backbone_2d.py`: Swin-Tiny feature extractor and 2D PE.
- `models/transformer_decoder_2d.py`: 6-layer Hierarchical 2D Masked Decoder.
- `models/vector_losses_2d.py`: Complete 2D Loss Suite and Hungarian Matcher.
- `models/surgical_2d_vector_transformer.py`: Top-level PyTorch module.
- `utils/dataset_2d.py`: Clean 2D surgical dataset reader.
- `scripts/train_2d_vector_transformer.py`: Training script with optimal instance metrics and visual overlays.
- `scripts/evaluate_2d.py`: Evaluation script (mAP, Chamfer Distance, Dice).
- `scripts/visualize_pipeline_smoke_test_2d.py`: 9-panel visual smoke test.
- `Run_Commands.md`: Step-by-step execution instructions for Colab and Kaggle.
