"""
HuggingFace-first workflow templates for phi2 analysis.
Memory-efficient analysis pipelines using remote data.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Iterator
from pathlib import Path
from hf_data_loader import HFDataLoader, load_phi2_weights, load_phi2_point_clouds
from utils import logger
import tempfile
import os

class StreamingAnalyzer:
    """Memory-efficient analyzer using HuggingFace remote data."""
    
    def __init__(self, repo_id: str = "totalorganfailure/phi2-weights", 
                 local_cache_size: int = 1000):  # MB
        self.loader = HFDataLoader(repo_id)
        self.cache_size = local_cache_size * 1024 * 1024  # Convert to bytes
        self.temp_dir = tempfile.mkdtemp(prefix="phi2_analysis_")
        logger.info(f"🔄 StreamingAnalyzer initialized with {local_cache_size}MB cache")
    
    def analyze_layer_range(self, start_layer: int, end_layer: int, 
                          components: List[str] = None) -> Dict[str, Any]:
        """Analyze a specific range of layers without loading full dataset."""
        components = components or ['Q', 'K', 'V', 'O', 'FF1', 'FF2']
        
        logger.info(f"📊 Analyzing layers {start_layer}-{end_layer}")
        
        results = {
            'layer_range': (start_layer, end_layer),
            'components': components,
            'statistics': {},
            'matrices_processed': 0
        }
        
        # Stream weight matrices for this layer range
        for matrix_name, matrix in self.loader.stream_weight_matrices(
            layer_range=(start_layer, end_layer), 
            components=components
        ):
            # Compute statistics on-the-fly
            stats = {
                'shape': matrix.shape,
                'mean': float(np.mean(matrix)),
                'std': float(np.std(matrix)),
                'min': float(np.min(matrix)),
                'max': float(np.max(matrix)),
                'norm': float(np.linalg.norm(matrix))
            }
            
            results['statistics'][matrix_name] = stats
            results['matrices_processed'] += 1
            
            # Clear matrix from memory immediately
            del matrix
        
        logger.info(f"✓ Processed {results['matrices_processed']} matrices")
        return results
    
    def compare_attention_heads(self, layer_id: int) -> Dict[str, Any]:
        """Compare Q/K/V/O matrices for a specific layer."""
        logger.info(f"🔍 Comparing attention heads for layer {layer_id}")
        
        # Load only the matrices we need
        matrix_names = [f'Q_{layer_id}', f'K_{layer_id}', f'V_{layer_id}', f'O_{layer_id}']
        matrices = {}
        
        for name in matrix_names:
            matrices[name] = self.loader.load_weight_matrix(name)
        
        # Compute comparative statistics
        comparison = {
            'layer_id': layer_id,
            'matrix_shapes': {name: mat.shape for name, mat in matrices.items()},
            'norms': {name: float(np.linalg.norm(mat)) for name, mat in matrices.items()},
            'similarities': {}
        }
        
        # Compute pairwise similarities
        names = list(matrices.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                mat1_flat = matrices[name1].flatten()
                mat2_flat = matrices[name2].flatten()
                
                # Ensure same length for correlation
                min_len = min(len(mat1_flat), len(mat2_flat))
                correlation = np.corrcoef(mat1_flat[:min_len], mat2_flat[:min_len])[0, 1]
                comparison['similarities'][f'{name1}_vs_{name2}'] = float(correlation)
        
        logger.info(f"✓ Attention head comparison complete")
        return comparison
    
    def streaming_pca_analysis(self, stratum_name: str, 
                              n_components: int = 10) -> Dict[str, Any]:
        """Perform PCA analysis on a point cloud stratum."""
        logger.info(f"📊 Streaming PCA analysis for {stratum_name}")
        
        # Load point cloud
        pc_data = self.loader.load_point_cloud(stratum_name)
        points = pc_data['points']
        
        # Compute PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(points)
        
        results = {
            'stratum_name': stratum_name,
            'original_shape': points.shape,
            'transformed_shape': transformed.shape,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'n_components': n_components
        }
        
        # Clear from memory
        del points, transformed
        
        logger.info(f"✓ PCA analysis complete: {results['explained_variance_ratio'][:3]}")
        return results
    
    def batch_stratum_analysis(self, strata: List[str], 
                              analysis_type: str = "pca") -> Dict[str, Any]:
        """Analyze multiple strata in batches to manage memory."""
        logger.info(f"🔄 Batch analysis of {len(strata)} strata")
        
        results = {
            'analysis_type': analysis_type,
            'strata_analyzed': len(strata),
            'results': {}
        }
        
        for stratum in strata:
            try:
                if analysis_type == "pca":
                    stratum_result = self.streaming_pca_analysis(stratum)
                    results['results'][stratum] = stratum_result
                    
                elif analysis_type == "statistics":
                    pc_data = self.loader.load_point_cloud(stratum)
                    points = pc_data['points']
                    
                    stats = {
                        'shape': points.shape,
                        'mean': np.mean(points, axis=0).tolist(),
                        'std': np.std(points, axis=0).tolist(),
                        'point_norms': {
                            'mean': float(np.mean(np.linalg.norm(points, axis=1))),
                            'std': float(np.std(np.linalg.norm(points, axis=1)))
                        }
                    }
                    results['results'][stratum] = stats
                    del points
                    
            except Exception as e:
                logger.error(f"Failed to analyze {stratum}: {e}")
                results['results'][stratum] = {'error': str(e)}
        
        logger.info(f"✓ Batch analysis complete")
        return results
    
    def cleanup(self):
        """Clean up temporary files and cache."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        self.loader.clear_cache()
        logger.info("🧹 Cleanup complete")

# Workflow templates
def quick_layer_analysis(layer_id: int, repo_id: str = "totalorganfailure/phi2-weights") -> Dict[str, Any]:
    """Quick analysis of a specific layer."""
    analyzer = StreamingAnalyzer(repo_id)
    
    try:
        # Analyze single layer
        layer_stats = analyzer.analyze_layer_range(layer_id, layer_id + 1)
        
        # Compare attention heads
        attention_comparison = analyzer.compare_attention_heads(layer_id)
        
        return {
            'layer_id': layer_id,
            'layer_statistics': layer_stats,
            'attention_comparison': attention_comparison
        }
    
    finally:
        analyzer.cleanup()

def comparative_stratum_analysis(strata: List[str], 
                               repo_id: str = "totalorganfailure/phi2-weights") -> Dict[str, Any]:
    """Compare multiple strata using remote data."""
    analyzer = StreamingAnalyzer(repo_id)
    
    try:
        # PCA analysis
        pca_results = analyzer.batch_stratum_analysis(strata, "pca")
        
        # Statistical analysis
        stats_results = analyzer.batch_stratum_analysis(strata, "statistics")
        
        return {
            'strata': strata,
            'pca_analysis': pca_results,
            'statistical_analysis': stats_results
        }
    
    finally:
        analyzer.cleanup()

def memory_efficient_tda(stratum_name: str, 
                        max_points: int = 1000,
                        repo_id: str = "totalorganfailure/phi2-weights") -> Dict[str, Any]:
    """Run TDA on a subsample to manage memory."""
    from ripser import ripser
    
    loader = HFDataLoader(repo_id)
    
    try:
        # Load point cloud
        pc_data = loader.load_point_cloud(stratum_name)
        points = pc_data['points']
        
        # Subsample if needed
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            logger.info(f"📊 Subsampled to {max_points} points for TDA")
        
        # Run TDA
        logger.info(f"🔄 Computing persistent homology for {stratum_name}")
        diagrams = ripser(points, maxdim=1)['dgms']
        
        # Compute Betti numbers
        beta_0 = len(diagrams[0]) - 1  # Subtract the infinite bar
        beta_1 = len(diagrams[1]) if len(diagrams) > 1 else 0
        
        results = {
            'stratum_name': stratum_name,
            'points_analyzed': len(points),
            'beta_0': beta_0,
            'beta_1': beta_1,
            'max_persistence_H0': float(np.max(diagrams[0][diagrams[0][:, 1] != np.inf, 1] - diagrams[0][diagrams[0][:, 1] != np.inf, 0])) if len(diagrams[0]) > 1 else 0,
            'max_persistence_H1': float(np.max(diagrams[1][:, 1] - diagrams[1][:, 0])) if beta_1 > 0 else 0
        }
        
        logger.info(f"✓ TDA complete: β₀={beta_0}, β₁={beta_1}")
        return results
        
    except Exception as e:
        logger.error(f"TDA failed for {stratum_name}: {e}")
        return {'error': str(e)}
    
    finally:
        loader.clear_cache()

# Configuration templates
def create_hf_config(repo_id: str = "totalorganfailure/phi2-weights") -> Dict[str, Any]:
    """Create configuration for HuggingFace-first workflows."""
    return {
        "data_source": {
            "type": "huggingface",
            "repo_id": repo_id,
            "cache_strategy": "minimal",
            "max_local_cache_mb": 1000
        },
        "analysis_patterns": {
            "layer_wise": {
                "batch_size": 1,
                "components": ["Q", "K", "V", "O", "FF1", "FF2"]
            },
            "stratum_wise": {
                "max_points_per_analysis": 5000,
                "pca_components": 10
            },
            "tda": {
                "max_points": 1000,
                "max_dimension": 1
            }
        },
        "memory_management": {
            "cleanup_after_each_analysis": True,
            "temp_dir_prefix": "phi2_analysis_",
            "max_concurrent_matrices": 4
        }
    }

if __name__ == "__main__":
    # Example usage
    logger.info("🚀 Testing HuggingFace-first workflows")
    
    # Quick layer analysis
    result = quick_layer_analysis(0)
    print(f"Layer 0 analysis: {result['layer_statistics']['matrices_processed']} matrices")
    
    # Comparative analysis
    strata = ["Q_stratum", "K_stratum", "V_stratum"]
    comparison = comparative_stratum_analysis(strata)
    print(f"Stratum comparison: {len(comparison['pca_analysis']['results'])} strata")
    
    # Memory-efficient TDA
    tda_result = memory_efficient_tda("embedding", max_points=500)
    print(f"TDA result: β₀={tda_result.get('beta_0', 'N/A')}, β₁={tda_result.get('beta_1', 'N/A')}")
    
    print("✅ HuggingFace-first workflows ready!")