"""
Extract and prepare point clouds from phi-2 weights for TDA analysis.
"""
import numpy as np
import h5py
import pathlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    log_memory_usage, normalize_rows, subsample_points, 
    validate_point_cloud, logger
)

def load_weights_from_hdf5(input_path: str) -> tuple:
    """Load weights from HDF5 file."""
    logger.info(f"Loading weights from {input_path}")
    
    path = pathlib.Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Weight file not found: {input_path}")
    
    layer_dict = {}
    metadata = {}
    
    with h5py.File(path, "r") as f:
        # Load metadata
        for key in f.attrs.keys():
            metadata[key] = f.attrs[key]
        
        # Load weights
        for name in f.keys():
            layer_dict[name] = f[name][:]
            logger.info(f"Loaded {name}: {layer_dict[name].shape}")
    
    logger.info(f"Loaded {len(layer_dict)} weight matrices")
    logger.info(f"Metadata: {metadata}")
    
    return layer_dict, metadata

def prepare_stratum_point_cloud(
    weight_matrix: np.ndarray, 
    stratum_name: str,
    max_points: int = 10000,
    pca_components: int = 30,
    normalize_method: str = 'l2'
) -> np.ndarray:
    """Prepare point cloud from weight matrix for TDA."""
    logger.info(f"Preparing point cloud for {stratum_name}")
    logger.info(f"Input matrix shape: {weight_matrix.shape}")
    
    # Each row is a point in the ambient space
    points = weight_matrix.copy()
    
    # Normalize rows to remove scale ambiguity
    if normalize_method:
        points = normalize_rows(points, method=normalize_method)
        logger.info(f"Normalized {points.shape[0]} points using {normalize_method}")
    
    # Subsample if too many points
    if points.shape[0] > max_points:
        points = subsample_points(points, max_points, method='random')
        logger.info(f"Subsampled to {points.shape[0]} points")
    
    # Dimensionality reduction for computational efficiency
    if points.shape[1] > pca_components:
        logger.info(f"Applying PCA: {points.shape[1]} -> {pca_components} dimensions")
        pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=42)
        points = pca.fit_transform(points)
        
        # Log explained variance
        explained_var = pca.explained_variance_ratio_.sum()
        logger.info(f"PCA explained variance: {explained_var:.3f}")
    
    # Validate point cloud
    if not validate_point_cloud(points, stratum_name):
        raise ValueError(f"Invalid point cloud for {stratum_name}")
    
    logger.info(f"Final point cloud shape: {points.shape}")
    return points

def extract_all_strata(layer_dict: dict, metadata: dict) -> dict:
    """Extract point clouds for all strata."""
    logger.info("Extracting point clouds for all strata...")
    
    point_clouds = {}
    num_layers = metadata.get('num_layers', 0)
    
    # Process embedding stratum
    if 'embed' in layer_dict:
        try:
            embed_cloud = prepare_stratum_point_cloud(
                layer_dict['embed'], 
                'embedding_stratum',
                max_points=8000,
                pca_components=25
            )
            point_clouds['embedding'] = embed_cloud
        except Exception as e:
            logger.error(f"Failed to process embedding stratum: {e}")
    
    # Process attention strata (Q, K, V, O)
    attention_strata = ['Q', 'K', 'V', 'O']
    for stratum_type in attention_strata:
        try:
            # Collect all layers for this stratum
            stratum_matrices = []
            for layer_idx in range(num_layers):
                key = f"{stratum_type}_{layer_idx}"
                if key in layer_dict:
                    stratum_matrices.append(layer_dict[key])
            
            if stratum_matrices:
                # Concatenate all layers
                combined_matrix = np.vstack(stratum_matrices)
                logger.info(f"Combined {stratum_type} stratum: {combined_matrix.shape}")
                
                # Prepare point cloud
                point_cloud = prepare_stratum_point_cloud(
                    combined_matrix,
                    f"{stratum_type}_stratum",
                    max_points=12000,
                    pca_components=30
                )
                point_clouds[f"{stratum_type}_stratum"] = point_cloud
            else:
                logger.warning(f"No matrices found for {stratum_type} stratum")
        except Exception as e:
            logger.error(f"Failed to process {stratum_type} stratum: {e}")
    
    # Process MLP strata (FF1, FF2)
    mlp_strata = ['FF1', 'FF2']
    for stratum_type in mlp_strata:
        try:
            # Collect all layers for this stratum
            stratum_matrices = []
            for layer_idx in range(num_layers):
                key = f"{stratum_type}_{layer_idx}"
                if key in layer_dict:
                    stratum_matrices.append(layer_dict[key])
            
            if stratum_matrices:
                # Concatenate all layers
                combined_matrix = np.vstack(stratum_matrices)
                logger.info(f"Combined {stratum_type} stratum: {combined_matrix.shape}")
                
                # Prepare point cloud
                point_cloud = prepare_stratum_point_cloud(
                    combined_matrix,
                    f"{stratum_type}_stratum",
                    max_points=15000,
                    pca_components=35
                )
                point_clouds[f"{stratum_type}_stratum"] = point_cloud
            else:
                logger.warning(f"No matrices found for {stratum_type} stratum")
        except Exception as e:
            logger.error(f"Failed to process {stratum_type} stratum: {e}")
    
    logger.info(f"Extracted {len(point_clouds)} point clouds")
    return point_clouds

def save_point_clouds(point_clouds: dict, output_dir: str = "point_clouds") -> None:
    """Save point clouds to individual files."""
    logger.info(f"Saving point clouds to {output_dir}/")
    
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for stratum_name, points in point_clouds.items():
        file_path = output_path / f"{stratum_name}.npz"
        np.savez_compressed(file_path, points=points)
        logger.info(f"Saved {stratum_name}: {points.shape} -> {file_path}")
    
    # Create summary file
    summary_path = output_path / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Point Cloud Summary\n")
        f.write("==================\n\n")
        for stratum_name, points in point_clouds.items():
            f.write(f"{stratum_name}: {points.shape[0]} points, {points.shape[1]} dimensions\n")
        f.write(f"\nTotal strata: {len(point_clouds)}\n")
        f.write(f"Total points: {sum(p.shape[0] for p in point_clouds.values())}\n")
    
    logger.info(f"Saved summary to {summary_path}")

def analyze_intrinsic_dimensions(point_clouds: dict) -> dict:
    """Analyze intrinsic dimensionality of each stratum."""
    logger.info("Analyzing intrinsic dimensions...")
    
    results = {}
    
    for stratum_name, points in point_clouds.items():
        try:
            # Compute SVD for dimensionality analysis
            U, s, Vt = np.linalg.svd(points, full_matrices=False)
            
            # Compute explained variance ratios
            explained_var_ratios = s**2 / np.sum(s**2)
            
            # Find effective dimensionality (95% variance)
            cumsum = np.cumsum(explained_var_ratios)
            effective_dim = np.argmax(cumsum >= 0.95) + 1
            
            results[stratum_name] = {
                'ambient_dim': points.shape[1],
                'effective_dim_95': effective_dim,
                'effective_dim_99': np.argmax(cumsum >= 0.99) + 1,
                'top_5_singular_values': s[:5].tolist(),
                'explained_var_ratio_top_5': explained_var_ratios[:5].tolist()
            }
            
            logger.info(f"{stratum_name}: ambient_dim={points.shape[1]}, "
                       f"effective_dim_95={effective_dim}")
            
        except Exception as e:
            logger.error(f"Failed to analyze {stratum_name}: {e}")
    
    return results

def main():
    """Main function to extract point clouds from phi-2 weights."""
    INPUT_PATH = "phi2_weights.h5"
    OUTPUT_DIR = "point_clouds"
    
    try:
        # Load weights
        layer_dict, metadata = load_weights_from_hdf5(INPUT_PATH)
        
        # Extract point clouds
        point_clouds = extract_all_strata(layer_dict, metadata)
        
        # Analyze intrinsic dimensions
        dim_analysis = analyze_intrinsic_dimensions(point_clouds)
        
        # Save results
        save_point_clouds(point_clouds, OUTPUT_DIR)
        
        # Save dimension analysis (convert numpy types to native Python types)
        import json
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        dim_analysis_serializable = convert_numpy_types(dim_analysis)
        with open(f"{OUTPUT_DIR}/dimension_analysis.json", "w") as f:
            json.dump(dim_analysis_serializable, f, indent=2)
        
        logger.info("✓ Point cloud extraction completed successfully!")
        
        # Summary
        total_points = sum(p.shape[0] for p in point_clouds.values())
        logger.info(f"Total point clouds: {len(point_clouds)}")
        logger.info(f"Total points: {total_points:,}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()