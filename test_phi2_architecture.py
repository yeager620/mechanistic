#!/usr/bin/env python3
"""
Test script to verify the actual Phi-2 model architecture
"""
from transformers import AutoModelForCausalLM, AutoConfig
import torch

def test_phi2_architecture():
    """Test the actual Phi-2 model architecture"""
    print("Loading Phi-2 model...")
    
    # Load config first
    config = AutoConfig.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    print(f"Config num_hidden_layers: {config.num_hidden_layers}")
    print(f"Config hidden_size: {config.hidden_size}")
    print(f"Config intermediate_size: {getattr(config, 'intermediate_size', 'Not found')}")
    print(f"Config vocab_size: {config.vocab_size}")
    print(f"Config num_attention_heads: {getattr(config, 'num_attention_heads', getattr(config, 'n_head', 'Not found'))}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Check actual model structure
    print(f"\nActual model.model.layers length: {len(model.model.layers)}")
    
    # Check first layer structure
    first_layer = model.model.layers[0]
    print(f"First layer type: {type(first_layer)}")
    print(f"First layer attributes: {[attr for attr in dir(first_layer) if not attr.startswith('_')]}")
    
    # Check MLP structure
    if hasattr(first_layer, 'mlp'):
        mlp = first_layer.mlp
        print(f"MLP type: {type(mlp)}")
        print(f"MLP attributes: {[attr for attr in dir(mlp) if not attr.startswith('_') and not callable(getattr(mlp, attr))]}")
        
        if hasattr(mlp, 'fc1'):
            print(f"FC1 weight shape: {mlp.fc1.weight.shape}")
            print(f"FC1 bias shape: {mlp.fc1.bias.shape if mlp.fc1.bias is not None else 'None'}")
        if hasattr(mlp, 'fc2'):
            print(f"FC2 weight shape: {mlp.fc2.weight.shape}")
            print(f"FC2 bias shape: {mlp.fc2.bias.shape if mlp.fc2.bias is not None else 'None'}")
    
    # Check all layer indices
    print(f"\nAll layer indices: {list(range(len(model.model.layers)))}")
    
    # Check last layer
    last_layer = model.model.layers[-1]
    print(f"Last layer index: {len(model.model.layers) - 1}")
    if hasattr(last_layer, 'mlp'):
        mlp = last_layer.mlp
        if hasattr(mlp, 'fc1'):
            print(f"Last layer FC1 weight shape: {mlp.fc1.weight.shape}")

if __name__ == "__main__":
    test_phi2_architecture()