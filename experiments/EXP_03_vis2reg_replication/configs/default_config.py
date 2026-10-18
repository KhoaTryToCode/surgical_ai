from types import SimpleNamespace

def get_default_config():
    """
    Returns the standard configuration for Vis2Reg replication.
    Values follow the MICCAI 2026 paper specifications.
    """
    cfg = SimpleNamespace(
        model={
            'geometric_encoder': {
                'input_dim': None,
                'feature_dim': 256,
                'layers': 4,
                'attr_dim': 0,
                'hidden_dim': 256,
                'heads': 4,
                'coarse_points': 256,
                'knn_k': 20,
                'pool_k': 32,
                'norm_groups': 8,
            },
            'visual_encoder': {
                'enabled': False,  # Geometry-only registration mode (as defined in paper §2.2)
                'backbone': 'resnet50',
                'pretrained': True,
                'feature_dim': 256,
                'num_frames': 1,
            },
            'fusion': {
                'enabled': False,
            },
            'deformation_field': {
                'hidden_dim': 256,
                'num_layers': 5,
                'pe_frequencies': 8,
                'skip_connections': [2, 4],
            },
            'pose_net': {
                'hidden_dim': 256,
                'iters': 3,
                'predict_scale': True,
            },
            'rigid': {
                'use_matching_solver': True,
                'use_camera_frame': True,
                'match_temperature': 0.1,
                'match_coarse_points': 512,
                'sinkhorn_iters': 20,
                'dustbin_score': 1.0,
                'use_ransac_solver': False,  # True for discrete PROSAC/RANSAC
                'use_icp_refine': False,     # True for ICP refinement
            },
            'renderer': None,  # Point renderer configured dynamically for Stage 2
        },
        data={
            'dataset': 'p2i_lreg',
            'max_points': 6000,
            'real_ocv2renderer': True,
        },
        training={
            'batch_size': 1,
            'lr': 3e-4,
            'weight_decay': 1e-4,
            'synthetic_epochs': 20,
            'real_epochs': 60,
            'rigid_warmup_epochs': 20,
            'lambda_3d': 0.5,
            'lambda_vis': 1.2,
            'lambda_sil': 1.0,
            'lambda_def': 0.1,
            'lambda_smooth': 0.1,
            'lambda_topo': 0.3,
        }
    )
    return cfg
