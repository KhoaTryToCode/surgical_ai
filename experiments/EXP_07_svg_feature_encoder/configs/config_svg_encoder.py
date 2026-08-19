import os

class SVGEncoderConfig:
    """
    Configuration for EXP_07: SVG / Vector-Aware Feature Map vs Standard Swin Encoder.
    """
    exp_id = "EXP_07_svg_feature_encoder"
    exp_name = "SVG_Vector_Aware_Feature_Map_Comparison"
    
    # Model architecture
    embed_dim = 256
    num_heads = 8
    mask2former_model_name = "facebook/mask2former-swin-tiny-ade-semantic"
    
    # SVG Vector Encoder parameters
    saliency_channels = 1
    vector_field_channels = 2 # (Tx, Ty) unit tangent vectors
    normal_field_channels = 2 # (Nx, Ny) unit normal vectors
    curvature_channels = 1    # kappa (local bending energy)
    
    # Bézier Primitive Extraction
    num_bezier_proposals = 5
    num_points_per_bezier = 30
    
    # Output settings
    output_dir = "experiments/EXP_07_svg_feature_encoder/outputs"

config = SVGEncoderConfig()
