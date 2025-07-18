"""
HuggingFace-first data loading utilities for phi2 analysis.
Enables remote data access without storing large files locally.
"""
import os
import tempfile
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
import json
from functools import lru_cache
from contextlib import contextmanager
from utils import logger

class HFDataLoader:
    """Remote data loader for HuggingFace datasets."""
    
    def __init__(self, repo_id: str = "totalorganfailure/phi2-weights", cache_dir: Optional[str] = None):
        self.repo_id = repo_id
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "hf_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"🔗 HFDataLoader initialized for {repo_id}")
    
    @lru_cache(maxsize=32)
    def list_available_files(self) -> List[str]:
        """List all available files in the HuggingFace dataset."""
        try:
            files = list_repo_files(self.repo_id, repo_type="dataset")
            logger.info(f"📁 Found {len(files)} files in {self.repo_id}")
            return files
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    @contextmanager
    def get_weights_file(self, filename: str = "phi2_weights.h5"):
        """Context manager for accessing the weights file."""
        try:
            # Download to temporary location
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                repo_type="dataset",
                cache_dir=self.cache_dir
            )
            logger.info(f"📥 Downloaded {filename} to cache")
            
            # Return h5py file handle
            with h5py.File(local_path, 'r') as f:
                yield f
                
        except Exception as e:
            logger.error(f"Failed to load weights file: {e}")
            raise
    
    def load_weight_matrix(self, matrix_name: str, weights_file: str = "phi2_weights.h5") -> np.ndarray:
        """Load a specific weight matrix from HuggingFace."""
        with self.get_weights_file(weights_file) as f:
            if matrix_name not in f:
                available = list(f.keys())
                logger.error(f"Matrix '{matrix_name}' not found. Available: {available[:10]}...")
                raise KeyError(f"Matrix '{matrix_name}' not found")
            
            matrix = f[matrix_name][:]
            logger.info(f"✓ Loaded {matrix_name}: {matrix.shape}")
            return matrix
    
    def load_point_cloud(self, stratum_name: str) -> Dict[str, np.ndarray]:
        """Load a point cloud from HuggingFace."""
        filename = f"point_clouds/{stratum_name}.npz"
        try:
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                repo_type="dataset",
                cache_dir=self.cache_dir
            )
            
            data = np.load(local_path)
            result = {key: data[key] for key in data.keys()}
            logger.info(f"✓ Loaded point cloud {stratum_name}: {result['points'].shape}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load point cloud {stratum_name}: {e}")
            raise
    
    def load_analysis_results(self, filename: str) -> Dict[str, Any]:
        """Load analysis results (JSON files) from HuggingFace."""
        try:
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                repo_type="dataset",
                cache_dir=self.cache_dir
            )
            
            with open(local_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"✓ Loaded analysis results: {filename}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load analysis results {filename}: {e}")
            raise
    
    def load_summary_data(self) -> pd.DataFrame:
        """Load the weight summary CSV from HuggingFace."""
        try:
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="weight_summary.csv",
                repo_type="dataset",
                cache_dir=self.cache_dir
            )
            
            df = pd.read_csv(local_path)
            logger.info(f"✓ Loaded weight summary: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load weight summary: {e}")
            raise
    
    def stream_weight_matrices(self, layer_range: Optional[tuple] = None, 
                             components: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Stream weight matrices on-demand without loading everything."""
        components = components or ['Q', 'K', 'V', 'O', 'FF1', 'FF2']
        
        with self.get_weights_file() as f:
            available_keys = list(f.keys())
            
            # Filter by layer range if specified
            if layer_range:
                start, end = layer_range
                filtered_keys = [k for k in available_keys 
                               if any(k.startswith(f"{comp}_") for comp in components)
                               and any(str(i) in k for i in range(start, end))]
            else:
                filtered_keys = [k for k in available_keys 
                               if any(k.startswith(f"{comp}_") for comp in components)]
            
            logger.info(f"📊 Streaming {len(filtered_keys)} weight matrices...")
            
            for key in filtered_keys:
                yield key, f[key][:]
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        try:
            # Load analysis stats
            stats = self.load_analysis_results("analysis_stats.json")
            
            # Get file list
            files = self.list_available_files()
            
            # Categorize files
            file_categories = {
                "weights": [f for f in files if f.endswith('.h5')],
                "point_clouds": [f for f in files if f.startswith('point_clouds/')],
                "analysis": [f for f in files if f.endswith('.json')],
                "summaries": [f for f in files if f.endswith('.csv')],
                "documentation": [f for f in files if f.endswith('.md')]
            }
            
            info = {
                "repo_id": self.repo_id,
                "total_files": len(files),
                "file_categories": file_categories,
                "analysis_stats": stats
            }
            
            logger.info(f"📊 Dataset info: {info['total_files']} files, {len(info['file_categories'])} categories")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get dataset info: {e}")
            return {"error": str(e)}
    
    def clear_cache(self):
        """Clear the local cache."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info("🧹 Cache cleared")

# Convenience functions for common operations
def load_phi2_weights(matrix_names: List[str], 
                     repo_id: str = "totalorganfailure/phi2-weights") -> Dict[str, np.ndarray]:
    """Load specific weight matrices from HuggingFace."""
    loader = HFDataLoader(repo_id)
    weights = {}
    
    for name in matrix_names:
        weights[name] = loader.load_weight_matrix(name)
    
    return weights

def load_phi2_point_clouds(strata: List[str], 
                          repo_id: str = "totalorganfailure/phi2-weights") -> Dict[str, Dict[str, np.ndarray]]:
    """Load specific point clouds from HuggingFace."""
    loader = HFDataLoader(repo_id)
    point_clouds = {}
    
    for stratum in strata:
        point_clouds[stratum] = loader.load_point_cloud(stratum)
    
    return point_clouds

def get_phi2_analysis_summary(repo_id: str = "totalorganfailure/phi2-weights") -> Dict[str, Any]:
    """Get complete analysis summary from HuggingFace."""
    loader = HFDataLoader(repo_id)
    
    try:
        return {
            "dataset_info": loader.get_dataset_info(),
            "weight_summary": loader.load_summary_data(),
            "available_files": loader.list_available_files()
        }
    except Exception as e:
        logger.error(f"Failed to get analysis summary: {e}")
        return {"error": str(e)}

# Usage examples in docstring
if __name__ == "__main__":
    # Example usage
    loader = HFDataLoader()
    
    # Get dataset overview
    info = loader.get_dataset_info()
    print(f"📊 Dataset: {info['total_files']} files")
    
    # Load specific weight matrices
    with loader.get_weights_file() as f:
        q_0 = f['Q_0'][:]
        k_0 = f['K_0'][:]
        print(f"✓ Q_0: {q_0.shape}, K_0: {k_0.shape}")
    
    # Load point clouds
    embedding_pc = loader.load_point_cloud("embedding")
    print(f"✓ Embedding point cloud: {embedding_pc['points'].shape}")
    
    # Load analysis results
    summary = loader.load_summary_data()
    print(f"✓ Weight summary: {len(summary)} entries")