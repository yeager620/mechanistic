#!/usr/bin/env python3
"""
MLP Reconstruction Module for Phi-2 Model
Reconstructs all 24 trained MLPs from extracted weights and biases.
"""
import torch
import torch.nn as nn
import numpy as np
import h5py
import json
import pathlib
from typing import Dict, List, Optional, Tuple
import logging
from utils import log_memory_usage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phi2MLP(nn.Module):
    """Reconstructed Phi-2 MLP with original weights and biases."""
    
    def __init__(self, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.fc1 = nn.Linear(2560, 10240, bias=True)
        self.fc2 = nn.Linear(10240, 2560, bias=True)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x -> fc1 -> GELU -> fc2."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
    def load_weights_and_biases(self, weights_dict: Dict[str, np.ndarray]):
        """Load weights and biases from extracted data."""
        layer_idx = self.layer_idx
        
        # Load fc1 weights and bias
        if f"FF1_{layer_idx}" in weights_dict:
            fc1_weight = torch.from_numpy(weights_dict[f"FF1_{layer_idx}"])
            self.fc1.weight.data = fc1_weight
            logger.info(f"Loaded FC1 weight for layer {layer_idx}: {fc1_weight.shape}")
            
        if f"FF1_bias_{layer_idx}" in weights_dict:
            fc1_bias = torch.from_numpy(weights_dict[f"FF1_bias_{layer_idx}"])
            self.fc1.bias.data = fc1_bias
            logger.info(f"Loaded FC1 bias for layer {layer_idx}: {fc1_bias.shape}")
            
        # Load fc2 weights and bias
        if f"FF2_{layer_idx}" in weights_dict:
            fc2_weight = torch.from_numpy(weights_dict[f"FF2_{layer_idx}"])
            self.fc2.weight.data = fc2_weight
            logger.info(f"Loaded FC2 weight for layer {layer_idx}: {fc2_weight.shape}")
            
        if f"FF2_bias_{layer_idx}" in weights_dict:
            fc2_bias = torch.from_numpy(weights_dict[f"FF2_bias_{layer_idx}"])
            self.fc2.bias.data = fc2_bias
            logger.info(f"Loaded FC2 bias for layer {layer_idx}: {fc2_bias.shape}")

def load_weights_from_hdf5(weights_path: str) -> Dict[str, np.ndarray]:
    """Load all weights and biases from HDF5 file."""
    logger.info(f"Loading weights from {weights_path}")
    weights_dict = {}
    
    with h5py.File(weights_path, 'r') as f:
        # Get metadata
        num_layers = f.attrs.get('num_layers', 32)
        logger.info(f"Model has {num_layers} layers")
        
        # Load all weight matrices and bias vectors
        for key in f.keys():
            if key.startswith(('FF1_', 'FF2_')):
                weights_dict[key] = f[key][:]
                logger.info(f"Loaded {key}: {weights_dict[key].shape}")
    
    return weights_dict

def create_mlp_module(layer_idx: int, weights_dict: Dict[str, np.ndarray]) -> Phi2MLP:
    """Create and initialize a single MLP module."""
    mlp = Phi2MLP(layer_idx)
    mlp.load_weights_and_biases(weights_dict)
    return mlp

def validate_mlp_reconstruction(mlp: Phi2MLP, layer_idx: int, test_input: torch.Tensor) -> Dict[str, float]:
    """Validate MLP reconstruction with test input."""
    mlp.eval()
    
    with torch.no_grad():
        output = mlp(test_input)
        
    # Basic validation metrics
    validation_metrics = {
        'layer_idx': layer_idx,
        'input_shape': list(test_input.shape),
        'output_shape': list(output.shape),
        'output_mean': output.mean().item(),
        'output_std': output.std().item(),
        'output_min': output.min().item(),
        'output_max': output.max().item(),
        'has_nan': torch.isnan(output).any().item(),
        'has_inf': torch.isinf(output).any().item()
    }
    
    return validation_metrics

def extract_and_reconstruct_mlps(weights_path: str, output_dir: str, num_layers: int = 32) -> Dict[str, any]:
    """Main function to reconstruct all MLPs from weights."""
    logger.info("Starting MLP reconstruction process")
    log_memory_usage("Before MLP reconstruction")
    
    # Create output directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    individual_mlps_path = output_path / "individual_mlps"
    individual_mlps_path.mkdir(exist_ok=True)
    
    # Load weights and biases
    weights_dict = load_weights_from_hdf5(weights_path)
    
    # Create test input for validation
    test_input = torch.randn(1, 2560)  # Single sample with 2560 features
    
    # Reconstruct all MLPs
    reconstructed_mlps = {}
    validation_results = []
    
    for layer_idx in range(num_layers):
        logger.info(f"Reconstructing MLP for layer {layer_idx}")
        
        # Create MLP module
        mlp = create_mlp_module(layer_idx, weights_dict)
        
        # Validate reconstruction
        validation_metrics = validate_mlp_reconstruction(mlp, layer_idx, test_input)
        validation_results.append(validation_metrics)
        
        # Save individual MLP
        individual_mlp_path = individual_mlps_path / f"mlp_layer_{layer_idx}.pth"
        torch.save(mlp.state_dict(), individual_mlp_path)
        logger.info(f"Saved MLP {layer_idx} to {individual_mlp_path}")
        
        # Store in collection
        reconstructed_mlps[f"layer_{layer_idx}"] = mlp
    
    # Save complete collection
    complete_collection_path = output_path / "phi2_mlps_complete.pth"
    complete_collection = {
        'mlps': {f"layer_{i}": reconstructed_mlps[f"layer_{i}"].state_dict() for i in range(num_layers)},
        'metadata': {
            'num_layers': num_layers,
            'input_dim': 2048,
            'hidden_dim': 8192,
            'output_dim': 2048,
            'activation': 'GELU'
        }
    }
    torch.save(complete_collection, complete_collection_path)
    logger.info(f"Saved complete collection to {complete_collection_path}")
    
    # Save metadata
    metadata = {
        'num_layers': num_layers,
        'architecture': {
            'input_dim': 2560,
            'hidden_dim': 10240,
            'output_dim': 2560,
            'activation': 'GELU',
            'formula': 'MLP(x) = W2(GELU(W1*x + b1)) + b2'
        },
        'parameters_per_mlp': {
            'fc1_weight': [10240, 2560],
            'fc1_bias': [10240],
            'fc2_weight': [2560, 10240],
            'fc2_bias': [2560]
        },
        'total_parameters_per_mlp': 2560 * 10240 + 10240 + 10240 * 2560 + 2560
    }
    
    metadata_path = output_path / "mlp_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save validation report
    validation_report = {
        'summary': {
            'total_mlps': len(validation_results),
            'successful_reconstructions': len([r for r in validation_results if not r['has_nan'] and not r['has_inf']]),
            'failed_reconstructions': len([r for r in validation_results if r['has_nan'] or r['has_inf']])
        },
        'per_layer_results': validation_results
    }
    
    validation_report_path = output_path / "validation_report.json"
    with open(validation_report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    # Create usage README
    readme_content = """# Phi-2 MLP Reconstructions

This directory contains reconstructed Multi-Layer Perceptrons (MLPs) from Microsoft's Phi-2 model.

## Contents

- `individual_mlps/`: Individual MLP state dictionaries (24 files)
- `phi2_mlps_complete.pth`: All 24 MLPs in a single file
- `mlp_metadata.json`: Architecture specifications
- `validation_report.json`: Reconstruction validation metrics

## Usage

```python
import torch
from mlp_reconstructor import Phi2MLP

# Load individual MLP
mlp = Phi2MLP(layer_idx=0)
mlp.load_state_dict(torch.load('individual_mlps/mlp_layer_0.pth'))

# Load complete collection
complete_collection = torch.load('phi2_mlps_complete.pth')
mlp_states = complete_collection['mlps']

# Use MLP
test_input = torch.randn(1, 2560)
with torch.no_grad():
    output = mlp(test_input)
```

## Architecture

Each MLP follows the pattern: `MLP(x) = W2(GELU(W1*x + b1)) + b2`

- Input dimension: 2560
- Hidden dimension: 10240  
- Output dimension: 2560
- Activation: GELU
"""
    
    readme_path = output_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    log_memory_usage("After MLP reconstruction")
    
    return {
        'reconstructed_mlps': reconstructed_mlps,
        'validation_results': validation_results,
        'output_directory': str(output_path),
        'metadata': metadata
    }

def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reconstruct Phi-2 MLPs from weights")
    parser.add_argument('--input', default='phi2_weights.h5', help='Input weights file')
    parser.add_argument('--output', default='mlp_reconstructions', help='Output directory')
    parser.add_argument('--num-layers', type=int, default=32, help='Number of layers to reconstruct')
    parser.add_argument('--validate', action='store_true', help='Run validation only')
    
    args = parser.parse_args()
    
    if args.validate:
        from validate_mlp_reconstruction import validate_mlp_reconstruction
        logger.info("Running validation against original model")
        validation_report = validate_mlp_reconstruction(
            args.input, args.output, num_layers=args.num_layers
        )
        summary = validation_report['summary']
        print(f"Validation completed! Pass rate: {summary['pass_rate']:.2%}")
        return
    
    results = extract_and_reconstruct_mlps(args.input, args.output, args.num_layers)
    
    logger.info("MLP reconstruction completed successfully!")
    logger.info(f"Output directory: {results['output_directory']}")
    logger.info(f"Reconstructed {len(results['reconstructed_mlps'])} MLPs")
    logger.info(f"Validation summary: {results['validation_results'][0] if results['validation_results'] else 'No validation performed'}")

if __name__ == "__main__":
    main()