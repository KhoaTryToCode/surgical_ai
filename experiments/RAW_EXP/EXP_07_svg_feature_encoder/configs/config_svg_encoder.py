import os

class SVGEncoderConfig:
    """
    Configuration for EXP_07: SVG / Vector-Aware Bézier Spline Transformer.
    """
    exp_id = "EXP_07_svg_feature_encoder"
    exp_name = "SVG_Bezier_Spline_Transformer"
    
    # Model architecture — Encoder
    embed_dim = 256
    num_heads = 8
    mask2former_model_name = "facebook/mask2former-swin-tiny-ade-semantic"
    
    # SVG Vector Encoder parameters
    saliency_channels = 1
    vector_field_channels = 2   # (Tx, Ty) unit tangent vectors
    normal_field_channels = 2   # (Nx, Ny) unit normal vectors
    curvature_channels = 1      # kappa (local bending energy)
    
    # Bézier Primitive Extraction (encoder-side)
    num_bezier_proposals = 5
    num_points_per_bezier = 30
    
    # Model architecture — Decoder
    num_queries = 10            # Number of Bézier query curves
    num_decoder_layers = 6      # Iterative refinement layers
    num_sample_t = 20           # Points sampled along each Bézier for probing
    num_classes = 4             # {Ridge=1, Silhouette=2, Falciform=3, Gallbladder=4}
    
    # Dataset
    dataset_dir = "/kaggle/working/L3D"
    num_instances = 10          # Max GT landmarks per image
    
    # Training hyperparameters
    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-4
    backbone_lr_mult = 0.1      # Backbone gets 0.1× base LR
    weight_decay = 1e-4
    num_workers = 2
    gradient_clip_norm = 1.0
    
    # Loss weights
    lambda_curve = 10.0         # L_curve: Ordered Arc-Length L1 distance (primary trajectory fitting)
    lambda_len = 2.0            # L_len: Curve span length match (prevents short noodle shortcut)
    lambda_cls = 2.0            # L_cls: Focal classification loss
    lambda_endpoint = 3.0       # L_endpoint: Endpoint anchoring
    lambda_smooth = 0.05        # L_smooth: Gentle curvature regularization
    lambda_aux_saliency = 2.0   # L_aux: Direct supervision on encoder saliency map
    
    # Validation metrics
    stroke_thickness_dice = 20  # Pixel stroke width for Dice/IoU rasterization
    
    # Visualization
    anchor_viz_enabled = True
    viz_interval = 10           # Log diagnostic every N steps
    
    # Output settings
    output_dir = "experiments/EXP_07_svg_feature_encoder/outputs"

config = SVGEncoderConfig()
