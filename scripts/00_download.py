"""
Download and extract phi-2 model weights into TDA-ready format.
"""
import torch
import numpy as np
import h5py
import pathlib
from transformers import AutoModelForCausalLM, AutoConfig
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    log_memory_usage, validate_phi2_architecture, extract_weight_safely, 
    extract_bias_safely, get_device_info, logger
)

def load_phi2_model(model_id: str = "microsoft/phi-2") -> tuple:
    """Load phi-2 model with memory-efficient settings."""
    log_memory_usage("Before model loading")
    
    # Get device info
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")
    
    # Load model with memory-efficient settings
    logger.info(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Reduce memory footprint
        device_map="auto",           # Automatic device placement
        low_cpu_mem_usage=True,
        trust_remote_code=True       # Required for phi-2
    )
    
    config = model.config
    log_memory_usage("After model loading")
    
    # Validate architecture
    arch_info = validate_phi2_architecture(model)
    logger.info(f"Architecture info: {arch_info}")
    
    # Extract key dimensions
    d = config.hidden_size
    L = config.num_hidden_layers
    H = getattr(config, 'n_head', getattr(config, 'num_attention_heads', 32))
    
    logger.info(f"Model dimensions - Hidden: {d}, Layers: {L}, Heads: {H}")
    
    return model, config, arch_info

def extract_phi2_weights(model, arch_info: dict) -> dict:
    """Extract phi-2 weights into stratified format."""
    logger.info("Starting weight extraction...")
    layer_dict = {}
    
    # Extract embeddings
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_weight = model.model.embed_tokens.weight.detach().cpu().float().numpy()
            layer_dict["embed"] = embed_weight
            logger.info(f"Extracted embeddings: {embed_weight.shape}")
        else:
            logger.warning("Could not find embedding weights")
    except Exception as e:
        logger.error(f"Failed to extract embeddings: {e}")
    
    # Extract transformer layer weights
    total_layers = len(model.model.layers)
    logger.info(f"Extracting weights from {total_layers} layers...")
    
    for layer_idx, block in enumerate(model.model.layers):
        try:
            logger.info(f"Processing layer {layer_idx}...")
            
            # Attention weights
            if hasattr(block, 'self_attn'):
                attn = block.self_attn
                
                # Handle separate QKV projections (phi-2 style)
                q_weight = extract_weight_safely(attn, 'q_proj', layer_idx)
                k_weight = extract_weight_safely(attn, 'k_proj', layer_idx)
                v_weight = extract_weight_safely(attn, 'v_proj', layer_idx)
                
                if q_weight is not None:
                    layer_dict[f"Q_{layer_idx}"] = q_weight
                    logger.info(f"Layer {layer_idx} q_proj: {q_weight.shape}")
                if k_weight is not None:
                    layer_dict[f"K_{layer_idx}"] = k_weight
                    logger.info(f"Layer {layer_idx} k_proj: {k_weight.shape}")
                if v_weight is not None:
                    layer_dict[f"V_{layer_idx}"] = v_weight
                    logger.info(f"Layer {layer_idx} v_proj: {v_weight.shape}")
                
                # Output projection (phi-2 uses 'dense' instead of 'out_proj')
                o_weight = extract_weight_safely(attn, 'dense', layer_idx)
                if o_weight is not None:
                    layer_dict[f"O_{layer_idx}"] = o_weight
                    logger.info(f"Layer {layer_idx} dense: {o_weight.shape}")
                else:
                    # Fallback to out_proj if dense doesn't exist
                    o_weight = extract_weight_safely(attn, 'out_proj', layer_idx)
                    if o_weight is not None:
                        layer_dict[f"O_{layer_idx}"] = o_weight
                        logger.info(f"Layer {layer_idx} out_proj: {o_weight.shape}")
            
            # MLP weights
            if hasattr(block, 'mlp'):
                mlp = block.mlp
                
                # Phi-2 style MLP
                if hasattr(mlp, 'fc1') and hasattr(mlp, 'fc2'):
                    fc1_weight = extract_weight_safely(mlp, 'fc1', layer_idx)
                    fc2_weight = extract_weight_safely(mlp, 'fc2', layer_idx)
                    fc1_bias = extract_bias_safely(mlp, 'fc1', layer_idx)
                    fc2_bias = extract_bias_safely(mlp, 'fc2', layer_idx)
                    
                    if fc1_weight is not None:
                        layer_dict[f"FF1_{layer_idx}"] = fc1_weight
                        logger.info(f"Layer {layer_idx} fc1: {fc1_weight.shape}")
                    if fc2_weight is not None:
                        layer_dict[f"FF2_{layer_idx}"] = fc2_weight
                        logger.info(f"Layer {layer_idx} fc2: {fc2_weight.shape}")
                    if fc1_bias is not None:
                        layer_dict[f"FF1_bias_{layer_idx}"] = fc1_bias
                        logger.info(f"Layer {layer_idx} fc1_bias: {fc1_bias.shape}")
                    if fc2_bias is not None:
                        layer_dict[f"FF2_bias_{layer_idx}"] = fc2_bias
                        logger.info(f"Layer {layer_idx} fc2_bias: {fc2_bias.shape}")
                else:
                    logger.warning(f"Layer {layer_idx} MLP has no fc1/fc2 structure")
            
            # Log progress
            if (layer_idx + 1) % 5 == 0:
                log_memory_usage(f"After layer {layer_idx}")
            
        except Exception as e:
            logger.error(f"Error processing layer {layer_idx}: {e}")
            logger.error(f"Layer {layer_idx} available attributes: {dir(block)}")
            raise
    
    logger.info(f"Extracted {len(layer_dict)} weight matrices")
    return layer_dict

def save_weights_to_hdf5(layer_dict: dict, output_path: str, model_id: str, config) -> None:
    """Save extracted weights to HDF5 file with metadata."""
    logger.info(f"Saving weights to {output_path}")
    
    path = pathlib.Path(output_path)
    with h5py.File(path, "w") as f:
        # Store metadata
        f.attrs["model_id"] = model_id
        f.attrs["hidden_size"] = config.hidden_size
        f.attrs["num_layers"] = config.num_hidden_layers
        f.attrs["num_heads"] = getattr(config, 'n_head', getattr(config, 'num_attention_heads', 32))
        f.attrs["vocab_size"] = config.vocab_size
        f.attrs["extraction_dtype"] = "float32"
        f.attrs["total_parameters"] = sum(arr.size for arr in layer_dict.values())
        
        # Store weights
        for name, arr in layer_dict.items():
            if arr is not None:
                f.create_dataset(name, data=arr, compression="gzip")
                logger.info(f"Saved {name}: {arr.shape}")
            else:
                logger.warning(f"Skipped {name}: None")
        
        logger.info(f"Saved {len(layer_dict)} datasets")
        file_size_mb = path.stat().st_size / (1024**2)
        logger.info(f"File size: {file_size_mb:.1f} MB")

def main():
    """Main function to download and extract phi-2 weights."""
    MODEL_ID = "microsoft/phi-2"
    OUTPUT_PATH = "phi2_weights.h5"
    
    try:
        # Load model
        model, config, arch_info = load_phi2_model(MODEL_ID)
        
        # Extract weights
        layer_dict = extract_phi2_weights(model, arch_info)
        
        # Save to HDF5
        save_weights_to_hdf5(layer_dict, OUTPUT_PATH, MODEL_ID, config)
        
        logger.info("✓ Weight extraction completed successfully!")
        
        # Summary
        total_params = sum(arr.size for arr in layer_dict.values())
        logger.info(f"Total parameters extracted: {total_params:,}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()