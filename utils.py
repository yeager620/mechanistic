"""
Utility functions for phi-2 mechanistic interpretability with TDA.
"""
import psutil
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_memory_info() -> Dict[str, float]:
    """Get current memory usage information."""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent': memory.percent
    }

def log_memory_usage(stage: str):
    """Log memory usage at different stages."""
    mem_info = get_memory_info()
    logger.info(f"{stage} - Memory: {mem_info['used_gb']:.1f}GB used, "
                f"{mem_info['available_gb']:.1f}GB available ({mem_info['percent']:.1f}%)")

def validate_phi2_architecture(model) -> Dict[str, Any]:
    """Validate phi-2 model architecture and return key info."""
    config = model.config
    
    # Basic config validation
    arch_info = {
        'model_class': model.__class__.__name__,
        'hidden_size': config.hidden_size,
        'num_layers': config.num_hidden_layers,
        'num_heads': getattr(config, 'n_head', getattr(config, 'num_attention_heads', 'unknown')),
        'vocab_size': config.vocab_size,
        'max_position_embeddings': getattr(config, 'max_position_embeddings', 'unknown')
    }
    
    # Check first layer structure
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        first_layer = model.transformer.h[0]
        arch_info['first_layer_type'] = type(first_layer).__name__
        arch_info['has_mixer'] = hasattr(first_layer, 'mixer')
        
        if hasattr(first_layer, 'mixer'):
            mixer = first_layer.mixer
            arch_info['mixer_type'] = type(mixer).__name__
            arch_info['has_Wqkv'] = hasattr(mixer, 'Wqkv')
            arch_info['has_separate_qkv'] = all(hasattr(mixer, attr) for attr in ['q_proj', 'k_proj', 'v_proj'])
            arch_info['has_out_proj'] = hasattr(mixer, 'out_proj')
        
        arch_info['has_mlp'] = hasattr(first_layer, 'mlp')
        if hasattr(first_layer, 'mlp'):
            mlp = first_layer.mlp
            arch_info['mlp_type'] = type(mlp).__name__
            arch_info['has_fc1_fc2'] = hasattr(mlp, 'fc1') and hasattr(mlp, 'fc2')
    
    # Check embedding structure
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'embd'):
        embd = model.transformer.embd
        arch_info['has_wte'] = hasattr(embd, 'wte')
        arch_info['has_wpe'] = hasattr(embd, 'wpe')
    
    return arch_info

def extract_weight_safely(module, attr_name: str, layer_idx: Optional[int] = None) -> Optional[np.ndarray]:
    """Safely extract weight from module attribute."""
    try:
        if hasattr(module, attr_name):
            weight = getattr(module, attr_name)
            if hasattr(weight, 'weight'):
                return weight.weight.detach().cpu().float().numpy()
            else:
                return weight.detach().cpu().float().numpy()
    except Exception as e:
        layer_info = f" (layer {layer_idx})" if layer_idx is not None else ""
        logger.warning(f"Failed to extract {attr_name}{layer_info}: {e}")
    return None

def extract_bias_safely(module, attr_name: str, layer_idx: Optional[int] = None) -> Optional[np.ndarray]:
    """Safely extract bias vector from module attribute."""
    try:
        if hasattr(module, attr_name):
            layer_module = getattr(module, attr_name)
            if hasattr(layer_module, 'bias') and layer_module.bias is not None:
                return layer_module.bias.detach().cpu().float().numpy()
    except Exception as e:
        layer_info = f" (layer {layer_idx})" if layer_idx is not None else ""
        logger.warning(f"Failed to extract bias for {attr_name}{layer_info}: {e}")
    return None

def normalize_rows(matrix: np.ndarray, method: str = 'l2') -> np.ndarray:
    """Normalize rows of matrix."""
    if method == 'l2':
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return matrix / norms
    elif method == 'unit':
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def subsample_points(points: np.ndarray, max_points: int = 10000, method: str = 'random') -> np.ndarray:
    """Subsample points for manageable TDA computation."""
    if len(points) <= max_points:
        return points
    
    if method == 'random':
        indices = np.random.choice(len(points), max_points, replace=False)
        return points[indices]
    elif method == 'uniform':
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        return points[indices]
    else:
        raise ValueError(f"Unknown subsampling method: {method}")

def validate_point_cloud(points: np.ndarray, name: str = "point_cloud") -> bool:
    """Validate point cloud for TDA computation."""
    if points.ndim != 2:
        logger.error(f"{name} must be 2D array, got {points.ndim}D")
        return False
    
    if points.shape[0] < 3:
        logger.error(f"{name} must have at least 3 points, got {points.shape[0]}")
        return False
    
    if points.shape[1] < 2:
        logger.error(f"{name} must have at least 2 dimensions, got {points.shape[1]}")
        return False
    
    if np.any(~np.isfinite(points)):
        logger.error(f"{name} contains non-finite values")
        return False
    
    logger.info(f"{name} validation passed: {points.shape[0]} points, {points.shape[1]} dimensions")
    return True

def get_device_info() -> Dict[str, Any]:
    """Get device information for model loading."""
    device_info = {
        'has_cuda': torch.cuda.is_available(),
        'has_mps': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'recommended_device': 'cpu'
    }
    
    if device_info['has_cuda']:
        device_info['recommended_device'] = 'cuda'
        device_info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif device_info['has_mps']:
        device_info['recommended_device'] = 'mps'
    
    return device_info