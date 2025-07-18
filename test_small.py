"""
Test the pipeline with a smaller model to verify functionality.
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig
from utils import logger, validate_phi2_architecture
import h5py

def test_with_small_model():
    """Test the pipeline with a smaller model."""
    logger.info("Testing pipeline with smaller model...")
    
    # Use TinyLlama as a test case
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    try:
        # Load small model
        logger.info(f"Loading test model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        config = model.config
        logger.info(f"Model loaded: {config.hidden_size}d, {config.num_hidden_layers} layers")
        
        # Test weight extraction from first layer
        first_layer = model.model.layers[0]
        logger.info(f"First layer type: {type(first_layer)}")
        logger.info(f"First layer attributes: {[attr for attr in dir(first_layer) if not attr.startswith('_')]}")
        
        # Test attention weights
        if hasattr(first_layer, 'self_attn'):
            attn = first_layer.self_attn
            logger.info(f"Attention attributes: {[attr for attr in dir(attn) if not attr.startswith('_')]}")
            
            if hasattr(attn, 'q_proj'):
                q_weight = attn.q_proj.weight.detach().cpu().float().numpy()
                logger.info(f"Q weight shape: {q_weight.shape}")
                
                # Test point cloud preparation
                from utils import normalize_rows, validate_point_cloud
                normalized = normalize_rows(q_weight)
                logger.info(f"Normalized Q weight shape: {normalized.shape}")
                
                if validate_point_cloud(normalized, "test_Q"):
                    logger.info("✓ Point cloud validation passed!")
                else:
                    logger.error("✗ Point cloud validation failed!")
                
                # Test basic TDA computation
                if normalized.shape[0] > 100:
                    # Subsample for quick test
                    test_points = normalized[:100]
                    logger.info(f"Testing TDA with {test_points.shape[0]} points")
                    
                    try:
                        from ripser import ripser
                        result = ripser(test_points, maxdim=1)
                        logger.info(f"TDA computation successful: {len(result['dgms'])} diagrams")
                        logger.info(f"H0 bars: {len(result['dgms'][0])}")
                        logger.info(f"H1 bars: {len(result['dgms'][1])}")
                        logger.info("✓ TDA computation passed!")
                    except Exception as e:
                        logger.error(f"TDA computation failed: {e}")
                        return False
                
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_with_small_model()
    if success:
        logger.info("✓ Small model test passed!")
    else:
        logger.error("✗ Small model test failed!")