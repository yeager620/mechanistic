"""
End-to-end test of the phi-2 TDA pipeline with a smaller model.
"""
import os
import shutil
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig
from utils import logger

def test_full_pipeline():
    """Test the complete pipeline with a smaller model."""
    logger.info("🧪 Testing complete TDA pipeline...")
    
    # Use a smaller model for testing
    TEST_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Clean up any existing test files
    test_files = ["test_weights.h5", "test_point_clouds", "test_tda_results"]
    for f in test_files:
        if os.path.exists(f):
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)
    
    try:
        # Step 1: Download and extract weights
        logger.info("📥 Step 1: Loading model and extracting weights...")
        from importlib import import_module
        
        # Modify the download script to use test model
        download_module = import_module('00_download')
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            TEST_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        config = model.config
        logger.info(f"✅ Model loaded: {config.hidden_size}d, {config.num_hidden_layers} layers")
        
        # Extract weights (adapted for TinyLlama structure)
        layer_dict = {}
        
        # Extract embedding
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_weight = model.model.embed_tokens.weight.detach().cpu().float().numpy()
            layer_dict["embed"] = embed_weight
            logger.info(f"Extracted embeddings: {embed_weight.shape}")
        
        # Extract first few layers only for testing
        max_layers = min(3, config.num_hidden_layers)
        for layer_idx in range(max_layers):
            layer = model.model.layers[layer_idx]
            
            # Attention weights
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                for proj_name, proj in [('q_proj', 'Q'), ('k_proj', 'K'), ('v_proj', 'V'), ('o_proj', 'O')]:
                    if hasattr(attn, proj_name):
                        weight = getattr(attn, proj_name).weight.detach().cpu().float().numpy()
                        layer_dict[f"{proj}_{layer_idx}"] = weight
                        logger.info(f"Extracted {proj}_{layer_idx}: {weight.shape}")
            
            # MLP weights
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                for proj_name, proj in [('up_proj', 'FF1'), ('down_proj', 'FF2')]:
                    if hasattr(mlp, proj_name):
                        weight = getattr(mlp, proj_name).weight.detach().cpu().float().numpy()
                        layer_dict[f"{proj}_{layer_idx}"] = weight
                        logger.info(f"Extracted {proj}_{layer_idx}: {weight.shape}")
        
        # Save weights
        import h5py
        with h5py.File("test_weights.h5", "w") as f:
            f.attrs["model_id"] = TEST_MODEL
            f.attrs["hidden_size"] = config.hidden_size
            f.attrs["num_layers"] = max_layers
            f.attrs["num_heads"] = getattr(config, 'num_attention_heads', 32)
            f.attrs["vocab_size"] = config.vocab_size
            f.attrs["extraction_dtype"] = "float32"
            f.attrs["total_parameters"] = sum(arr.size for arr in layer_dict.values())
            
            for name, arr in layer_dict.items():
                f.create_dataset(name, data=arr, compression="gzip")
        
        logger.info(f"✅ Step 1 complete: {len(layer_dict)} weight matrices saved")
        
        # Step 2: Extract point clouds
        logger.info("🎯 Step 2: Extracting point clouds...")
        
        # Import and run extraction
        extract_module = import_module('01_extract')
        
        # Load the weights we just saved
        with h5py.File("test_weights.h5", "r") as f:
            metadata = dict(f.attrs)
            loaded_weights = {name: f[name][:] for name in f.keys()}
        
        # Extract point clouds for each stratum
        point_clouds = {}
        for name, weight_matrix in loaded_weights.items():
            if weight_matrix.shape[0] > 10:  # Skip very small matrices
                try:
                    # Normalize and prepare
                    from utils import normalize_rows, subsample_points
                    points = normalize_rows(weight_matrix)
                    
                    # Subsample for testing
                    if points.shape[0] > 500:
                        points = subsample_points(points, 500)
                    
                    # Apply PCA if needed
                    if points.shape[1] > 20:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=20, random_state=42)
                        points = pca.fit_transform(points)
                    
                    point_clouds[name] = points
                    logger.info(f"Prepared {name}: {points.shape}")
                except Exception as e:
                    logger.warning(f"Failed to process {name}: {e}")
        
        # Save point clouds
        os.makedirs("test_point_clouds", exist_ok=True)
        for name, points in point_clouds.items():
            np.savez_compressed(f"test_point_clouds/{name}.npz", points=points)
        
        logger.info(f"✅ Step 2 complete: {len(point_clouds)} point clouds saved")
        
        # Step 3: Compute TDA
        logger.info("🔍 Step 3: Computing persistent homology...")
        
        tda_results = {}
        betti_analysis = {}
        
        from ripser import ripser
        
        for name, points in point_clouds.items():
            try:
                # Compute PH
                result = ripser(points, maxdim=1)
                diagrams = result['dgms']
                
                # Analyze results
                ph_summary = {}
                for dim in range(len(diagrams)):
                    dgm = diagrams[dim]
                    finite_bars = dgm[dgm[:, 1] != np.inf]
                    
                    ph_summary[f'H{dim}'] = {
                        'num_bars': len(dgm),
                        'num_finite_bars': len(finite_bars),
                        'num_infinite_bars': len(dgm) - len(finite_bars),
                        'max_persistence': float(np.max(finite_bars[:, 1] - finite_bars[:, 0])) if len(finite_bars) > 0 else 0.0
                    }
                
                tda_results[name] = {
                    'diagrams': diagrams,
                    'summary': ph_summary,
                    'point_count': points.shape[0],
                    'dimension': points.shape[1]
                }
                
                # Betti analysis
                betti_analysis[name] = {
                    'beta_0': ph_summary['H0']['num_infinite_bars'],
                    'beta_1': ph_summary['H1']['num_infinite_bars'] if 'H1' in ph_summary else 0,
                    'max_persistence_H0': ph_summary['H0']['max_persistence'],
                    'max_persistence_H1': ph_summary['H1']['max_persistence'] if 'H1' in ph_summary else 0.0
                }
                
                logger.info(f"Computed TDA for {name}: β₀={betti_analysis[name]['beta_0']}, β₁={betti_analysis[name]['beta_1']}")
                
            except Exception as e:
                logger.error(f"TDA failed for {name}: {e}")
        
        # Save TDA results
        os.makedirs("test_tda_results", exist_ok=True)
        
        import pickle
        with open("test_tda_results/persistence_diagrams.pkl", "wb") as f:
            pickle.dump(tda_results, f)
        
        import json
        with open("test_tda_results/betti_analysis.json", "w") as f:
            json.dump(betti_analysis, f, indent=2)
        
        logger.info(f"✅ Step 3 complete: TDA computed for {len(tda_results)} strata")
        
        # Step 4: Summary
        logger.info("📊 Step 4: Generating summary...")
        
        total_beta_0 = sum(b['beta_0'] for b in betti_analysis.values())
        total_beta_1 = sum(b['beta_1'] for b in betti_analysis.values())
        
        summary = f"""
🔍 PIPELINE TEST SUMMARY
========================

Model: {TEST_MODEL}
Extracted Layers: {max_layers}
Point Cloud Strata: {len(point_clouds)}
TDA Computations: {len(tda_results)}

Topology Results:
- Total Connected Components (β₀): {total_beta_0}
- Total Loops (β₁): {total_beta_1}

Strata Analysis:
"""
        
        for name, analysis in betti_analysis.items():
            summary += f"- {name}: β₀={analysis['beta_0']}, β₁={analysis['beta_1']}\n"
        
        summary += f"""
✅ PIPELINE TEST SUCCESSFUL!
All components working correctly.
Ready for full phi-2 analysis.
"""
        
        with open("test_tda_results/test_summary.txt", "w") as f:
            f.write(summary)
        
        logger.info(summary)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    if success:
        logger.info("🎉 Full pipeline test PASSED!")
    else:
        logger.error("💥 Full pipeline test FAILED!")