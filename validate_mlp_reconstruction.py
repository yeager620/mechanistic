#!/usr/bin/env python3
"""
Validation script for MLP reconstruction.
Compares reconstructed MLPs against original Phi-2 model.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import pathlib
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM
from mlp_reconstructor import Phi2MLP, load_weights_from_hdf5
from utils import log_memory_usage, logger
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_original_phi2_model(model_id: str = "microsoft/phi-2") -> nn.Module:
    """Load the original Phi-2 model for comparison."""
    logger.info(f"Loading original model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Use float32 for exact comparison
        device_map="cpu",           # Keep on CPU for validation
        trust_remote_code=True
    )
    model.eval()
    return model

def extract_original_mlp_outputs(model: nn.Module, test_inputs: torch.Tensor, num_layers: int = 24) -> List[torch.Tensor]:
    """Extract outputs from original model MLPs."""
    logger.info("Extracting original MLP outputs")
    original_outputs = []
    
    # Hook to capture MLP outputs
    def get_mlp_output_hook(layer_idx):
        def hook(module, input, output):
            original_outputs.append((layer_idx, output.detach().clone()))
        return hook
    
    # Register hooks
    hooks = []
    for layer_idx in range(num_layers):
        if hasattr(model.model.layers[layer_idx], 'mlp'):
            hook = model.model.layers[layer_idx].mlp.register_forward_hook(get_mlp_output_hook(layer_idx))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        model(test_inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Sort outputs by layer index
    original_outputs.sort(key=lambda x: x[0])
    return [output for _, output in original_outputs]

def validate_single_mlp(
    original_output: torch.Tensor,
    reconstructed_mlp: Phi2MLP,
    test_input: torch.Tensor,
    layer_idx: int,
    tolerance: float = 1e-5
) -> Dict[str, any]:
    """Validate a single reconstructed MLP against original."""
    reconstructed_mlp.eval()
    
    with torch.no_grad():
        reconstructed_output = reconstructed_mlp(test_input)
    
    # Ensure same shape
    if original_output.shape != reconstructed_output.shape:
        logger.error(f"Shape mismatch for layer {layer_idx}: "
                    f"original {original_output.shape} vs reconstructed {reconstructed_output.shape}")
        return {
            'layer_idx': layer_idx,
            'validation_passed': False,
            'error': 'Shape mismatch'
        }
    
    # Compute differences
    diff = torch.abs(original_output - reconstructed_output)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    relative_error = torch.mean(diff / (torch.abs(original_output) + 1e-8)).item()
    
    # Check if validation passes
    validation_passed = max_diff < tolerance
    
    return {
        'layer_idx': layer_idx,
        'validation_passed': validation_passed,
        'max_absolute_difference': max_diff,
        'mean_absolute_difference': mean_diff,
        'relative_error': relative_error,
        'tolerance': tolerance,
        'original_output_stats': {
            'mean': torch.mean(original_output).item(),
            'std': torch.std(original_output).item(),
            'min': torch.min(original_output).item(),
            'max': torch.max(original_output).item()
        },
        'reconstructed_output_stats': {
            'mean': torch.mean(reconstructed_output).item(),
            'std': torch.std(reconstructed_output).item(),
            'min': torch.min(reconstructed_output).item(),
            'max': torch.max(reconstructed_output).item()
        }
    }

def validate_mlp_reconstruction(
    weights_path: str,
    reconstruction_dir: str,
    model_id: str = "microsoft/phi-2",
    num_layers: int = 32,
    num_test_samples: int = 10,
    tolerance: float = 1e-5
) -> Dict[str, any]:
    """Comprehensive validation of MLP reconstruction."""
    logger.info("Starting comprehensive MLP reconstruction validation")
    log_memory_usage("Before validation")
    
    # Load original model
    original_model = load_original_phi2_model(model_id)
    
    # Load weights and create reconstructed MLPs
    weights_dict = load_weights_from_hdf5(weights_path)
    reconstructed_mlps = {}
    
    for layer_idx in range(num_layers):
        mlp = Phi2MLP(layer_idx)
        mlp.load_weights_and_biases(weights_dict)
        reconstructed_mlps[layer_idx] = mlp
    
    # Generate test inputs
    test_inputs = torch.randn(num_test_samples, 2560)
    
    # Extract original MLP outputs
    original_outputs = extract_original_mlp_outputs(original_model, test_inputs, num_layers)
    
    # Validate each layer
    validation_results = []
    passed_count = 0
    
    for layer_idx in range(num_layers):
        logger.info(f"Validating layer {layer_idx}")
        
        # Get the input to this MLP (we'll use the test input for simplicity)
        # In practice, this would be the actual intermediate representation
        mlp_input = test_inputs  # Simplified for validation
        
        validation_result = validate_single_mlp(
            original_outputs[layer_idx],
            reconstructed_mlps[layer_idx],
            mlp_input,
            layer_idx,
            tolerance
        )
        
        validation_results.append(validation_result)
        
        if validation_result['validation_passed']:
            passed_count += 1
            logger.info(f"✓ Layer {layer_idx} validation PASSED (max diff: {validation_result['max_absolute_difference']:.2e})")
        else:
            logger.warning(f"✗ Layer {layer_idx} validation FAILED (max diff: {validation_result['max_absolute_difference']:.2e})")
    
    # Summary statistics
    max_diffs = [r['max_absolute_difference'] for r in validation_results if 'max_absolute_difference' in r]
    mean_diffs = [r['mean_absolute_difference'] for r in validation_results if 'mean_absolute_difference' in r]
    
    summary = {
        'total_layers': num_layers,
        'passed_layers': passed_count,
        'failed_layers': num_layers - passed_count,
        'pass_rate': passed_count / num_layers,
        'tolerance': tolerance,
        'overall_max_difference': max(max_diffs) if max_diffs else float('inf'),
        'overall_mean_difference': np.mean(mean_diffs) if mean_diffs else float('inf'),
        'test_configuration': {
            'num_test_samples': num_test_samples,
            'model_id': model_id,
            'weights_path': weights_path,
            'reconstruction_dir': reconstruction_dir
        }
    }
    
    # Save validation report
    reconstruction_path = pathlib.Path(reconstruction_dir)
    validation_report = {
        'summary': summary,
        'per_layer_results': validation_results,
        'timestamp': str(torch.cuda.current_stream().query()) if torch.cuda.is_available() else 'N/A'
    }
    
    validation_report_path = reconstruction_path / "detailed_validation_report.json"
    with open(validation_report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    logger.info(f"Validation completed! Pass rate: {summary['pass_rate']:.2%}")
    logger.info(f"Overall max difference: {summary['overall_max_difference']:.2e}")
    logger.info(f"Detailed report saved to: {validation_report_path}")
    
    log_memory_usage("After validation")
    
    return validation_report

def main():
    """Main entry point for validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate MLP reconstruction against original model")
    parser.add_argument('--weights', default='phi2_weights.h5', help='Path to weights file')
    parser.add_argument('--reconstruction-dir', default='mlp_reconstructions', help='MLP reconstruction directory')
    parser.add_argument('--model-id', default='microsoft/phi-2', help='Original model ID')
    parser.add_argument('--num-layers', type=int, default=32, help='Number of layers')
    parser.add_argument('--tolerance', type=float, default=1e-5, help='Validation tolerance')
    parser.add_argument('--test-samples', type=int, default=10, help='Number of test samples')
    
    args = parser.parse_args()
    
    validation_report = validate_mlp_reconstruction(
        args.weights,
        args.reconstruction_dir,
        args.model_id,
        args.num_layers,
        args.test_samples,
        args.tolerance
    )
    
    # Print summary
    summary = validation_report['summary']
    print(f"\n{'='*60}")
    print(f"MLP RECONSTRUCTION VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total layers: {summary['total_layers']}")
    print(f"Passed: {summary['passed_layers']}")
    print(f"Failed: {summary['failed_layers']}")
    print(f"Pass rate: {summary['pass_rate']:.2%}")
    print(f"Overall max difference: {summary['overall_max_difference']:.2e}")
    print(f"Tolerance: {summary['tolerance']:.2e}")
    print(f"{'='*60}")
    
    if summary['pass_rate'] == 1.0:
        print("🎉 All MLPs passed validation!")
    elif summary['pass_rate'] >= 0.8:
        print("⚠️  Most MLPs passed validation, but some failed.")
    else:
        print("❌ Many MLPs failed validation - check reconstruction process.")

if __name__ == "__main__":
    main()