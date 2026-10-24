"""
Central Model Registry for Temporal Video Experiments.
Maps model identifiers ('minvis', 'cutie', 'sam2') to their architecture class,
tracker class, and training input mode.
"""

from typing import Dict, Any, Type, Optional, List
import torch.nn as nn

from .minvis_mask2former import MinVISMask2Former
from .tracker import MinVISTracker

MODEL_REGISTRY = {
    'minvis': {
        'model_class': MinVISMask2Former,
        'default_model_kwargs': {
            'base_checkpoint': 'facebook/mask2former-swin-tiny-ade-semantic',
            'num_classes': 3,
            'hidden_dim': 256
        },
        'tracker_class': MinVISTracker,
        'default_tracker_kwargs': {
            'num_classes': 3
        },
        'input_mode': 'single_frame',  # MinVIS trains on single frames
        'checkpoint_prefix': 'best_minvis',
        'description': 'MinVIS (NeurIPS 2022) query-based online bipartite tracking'
    },
    'cutie': {
        'model_class': None,  # Slot for EXP 18
        'default_model_kwargs': {},
        'tracker_class': None,  # Memory bank internal
        'default_tracker_kwargs': {},
        'input_mode': 'temporal_clip',
        'checkpoint_prefix': 'best_cutie',
        'description': 'Cutie (CVPR 2024) high-resolution object transformer & memory bank'
    },
    'sam2': {
        'model_class': None,  # Slot for EXP 19
        'default_model_kwargs': {},
        'tracker_class': None,  # Streaming memory attention internal
        'default_tracker_kwargs': {},
        'input_mode': 'temporal_clip',
        'checkpoint_prefix': 'best_sam2',
        'description': 'SAM 2 (Meta AI 2024) unified video foundation transformer'
    }
}


def get_model_entry(model_name: str) -> Dict[str, Any]:
    name = model_name.lower().strip()
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise KeyError(f"Model '{name}' not found in registry. Available models: {available}")
    
    entry = MODEL_REGISTRY[name]
    if entry['model_class'] is None:
        raise NotImplementedError(f"Model '{name}' is registered as a future experiment slot ({entry['description']}) but not yet implemented.")
    return entry


def list_available_models() -> List[str]:
    return [name for name, cfg in MODEL_REGISTRY.items() if cfg['model_class'] is not None]
