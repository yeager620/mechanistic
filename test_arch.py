"""
Quick test to verify phi-2 architecture inspection.
"""
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from utils import validate_phi2_architecture, get_device_info, logger

def test_phi2_architecture():
    """Test phi-2 architecture validation without full download."""
    logger.info("Testing phi-2 architecture validation...")
    
    # Get device info
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")
    
    # Try to load just the config first
    try:
        config = AutoConfig.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        logger.info(f"Config loaded successfully: {config}")
        logger.info(f"Hidden size: {config.hidden_size}")
        logger.info(f"Num layers: {config.num_hidden_layers}")
        logger.info(f"Num heads: {getattr(config, 'n_head', getattr(config, 'num_attention_heads', 32))}")
        return True
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return False

if __name__ == "__main__":
    success = test_phi2_architecture()
    if success:
        logger.info("✓ Architecture test passed!")
    else:
        logger.error("✗ Architecture test failed!")