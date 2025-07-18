"""
Explore and visualize the structure of extracted phi-2 weights.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from utils import logger

def explore_hdf5_structure(file_path: str = "phi2_weights.h5"):
    """Explore the structure of the HDF5 weights file."""
    logger.info(f"🔍 Exploring structure of {file_path}")
    
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        return
    
    with h5py.File(file_path, "r") as f:
        # Print metadata
        logger.info("📊 METADATA:")
        for key, value in f.attrs.items():
            logger.info(f"  {key}: {value}")
        
        # Print dataset structure
        logger.info("\n📋 WEIGHT MATRICES:")
        total_params = 0
        weight_info = []
        
        for name in sorted(f.keys()):
            dataset = f[name]
            shape = dataset.shape
            params = np.prod(shape)
            total_params += params
            
            weight_info.append({
                'name': name,
                'shape': f"{shape[0]}×{shape[1]}" if len(shape) == 2 else str(shape),
                'parameters': params,
                'dtype': str(dataset.dtype),
                'size_mb': dataset.nbytes / (1024**2)
            })
            
            logger.info(f"  {name:15} {shape!s:20} {params:>10,} params")
        
        logger.info(f"\n📈 SUMMARY:")
        logger.info(f"  Total datasets: {len(f.keys())}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  File size: {Path(file_path).stat().st_size / (1024**2):.1f} MB")
        
        return weight_info

def visualize_weight_distributions(file_path: str = "phi2_weights.h5", sample_size: int = 10000):
    """Visualize weight value distributions for different strata."""
    logger.info(f"📊 Visualizing weight distributions from {file_path}")
    
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Define strata to visualize
    strata_samples = {
        'embed': 'Embedding',
        'Q_0': 'Query Layer 0', 
        'K_0': 'Key Layer 0',
        'V_0': 'Value Layer 0',
        'FF1_0': 'MLP Up Layer 0',
        'FF2_0': 'MLP Down Layer 0'
    }
    
    with h5py.File(file_path, "r") as f:
        for i, (key, title) in enumerate(strata_samples.items()):
            if i >= len(axes):
                break
                
            if key in f:
                weights = f[key][:]
                
                # Sample for visualization
                if weights.size > sample_size:
                    flat_weights = weights.flatten()
                    sample_idx = np.random.choice(len(flat_weights), sample_size, replace=False)
                    sample_weights = flat_weights[sample_idx]
                else:
                    sample_weights = weights.flatten()
                
                # Plot histogram
                axes[i].hist(sample_weights, bins=50, alpha=0.7, density=True)
                axes[i].set_title(f'{title}\n{weights.shape} | μ={np.mean(sample_weights):.3f}, σ={np.std(sample_weights):.3f}')
                axes[i].set_xlabel('Weight Value')
                axes[i].set_ylabel('Density')
                axes[i].grid(True, alpha=0.3)
                
                logger.info(f"  {key}: shape={weights.shape}, mean={np.mean(sample_weights):.4f}, std={np.std(sample_weights):.4f}")
            else:
                axes[i].text(0.5, 0.5, f'{key}\nNot Found', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{title} - Not Found')
    
    # Hide unused subplots
    for i in range(len(strata_samples), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('weight_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    logger.info("📊 Weight distribution plot saved as 'weight_distributions.png'")

def visualize_weight_matrices(file_path: str = "phi2_weights.h5", max_dim: int = 100):
    """Visualize actual weight matrices as heatmaps."""
    logger.info(f"🔥 Visualizing weight matrices from {file_path}")
    
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    matrices_to_show = ['Q_0', 'K_0', 'V_0', 'O_0', 'FF1_0', 'FF2_0']
    
    with h5py.File(file_path, "r") as f:
        for i, matrix_name in enumerate(matrices_to_show):
            if i >= len(axes):
                break
                
            if matrix_name in f:
                weights = f[matrix_name][:]
                
                # Subsample for visualization if too large
                if weights.shape[0] > max_dim:
                    rows = np.random.choice(weights.shape[0], max_dim, replace=False)
                    weights = weights[rows]
                if weights.shape[1] > max_dim:
                    cols = np.random.choice(weights.shape[1], max_dim, replace=False)
                    weights = weights[:, cols]
                
                # Create heatmap
                im = axes[i].imshow(weights, cmap='RdBu_r', aspect='auto')
                axes[i].set_title(f'{matrix_name}\n{f[matrix_name].shape} → {weights.shape}')
                axes[i].set_xlabel('Hidden Dimension')
                axes[i].set_ylabel('Output Dimension')
                plt.colorbar(im, ax=axes[i], shrink=0.8)
                
                logger.info(f"  {matrix_name}: visualized {weights.shape} subset of {f[matrix_name].shape}")
            else:
                axes[i].text(0.5, 0.5, f'{matrix_name}\nNot Found', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{matrix_name} - Not Found')
    
    plt.tight_layout()
    plt.savefig('weight_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    logger.info("🔥 Weight matrix heatmaps saved as 'weight_matrices.png'")

def create_weight_summary_table(file_path: str = "phi2_weights.h5"):
    """Create a summary table of all weight matrices."""
    logger.info(f"📋 Creating weight summary table from {file_path}")
    
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        return
    
    weight_data = []
    
    with h5py.File(file_path, "r") as f:
        for name in sorted(f.keys()):
            dataset = f[name]
            weights = dataset[:]
            
            # Parse stratum and layer info
            if '_' in name:
                stratum, layer = name.split('_', 1)
                layer_num = layer if layer.isdigit() else 'N/A'
            else:
                stratum = name
                layer_num = 'N/A'
            
            weight_data.append({
                'Matrix': name,
                'Stratum': stratum,
                'Layer': layer_num,
                'Shape': f"{dataset.shape[0]}×{dataset.shape[1]}" if len(dataset.shape) == 2 else str(dataset.shape),
                'Parameters': np.prod(dataset.shape),
                'Mean': np.mean(weights),
                'Std': np.std(weights),
                'Min': np.min(weights),
                'Max': np.max(weights),
                'Sparsity': np.mean(np.abs(weights) < 1e-6) * 100  # % of near-zero weights
            })
    
    df = pd.DataFrame(weight_data)
    
    # Print summary by stratum
    logger.info("\n📊 SUMMARY BY STRATUM:")
    stratum_summary = df.groupby('Stratum').agg({
        'Parameters': 'sum',
        'Mean': 'mean',
        'Std': 'mean',
        'Matrix': 'count'
    }).round(4)
    stratum_summary.columns = ['Total_Params', 'Avg_Mean', 'Avg_Std', 'Num_Matrices']
    logger.info(stratum_summary.to_string())
    
    # Save detailed table
    df.to_csv('weight_summary.csv', index=False)
    logger.info("\n📋 Detailed weight summary saved as 'weight_summary.csv'")
    
    return df

def main():
    """Main function to explore weight structure."""
    logger.info("🔍 WEIGHT STRUCTURE EXPLORATION")
    logger.info("=" * 50)
    
    # Check if weights file exists
    weights_file = "phi2_weights.h5"
    if not Path(weights_file).exists():
        logger.error(f"❌ Weights file not found: {weights_file}")
        logger.info("ℹ️  Please run 'uv run python 00_download.py' first")
        return
    
    try:
        # 1. Explore HDF5 structure
        logger.info("\n1️⃣ EXPLORING HDF5 STRUCTURE:")
        weight_info = explore_hdf5_structure(weights_file)
        
        # 2. Create summary table
        logger.info("\n2️⃣ CREATING SUMMARY TABLE:")
        df = create_weight_summary_table(weights_file)
        
        # 3. Visualize weight distributions
        logger.info("\n3️⃣ VISUALIZING WEIGHT DISTRIBUTIONS:")
        visualize_weight_distributions(weights_file)
        
        # 4. Visualize weight matrices
        logger.info("\n4️⃣ VISUALIZING WEIGHT MATRICES:")
        visualize_weight_matrices(weights_file)
        
        logger.info("\n✅ Weight exploration complete!")
        logger.info("📁 Generated files:")
        logger.info("  - weight_summary.csv")
        logger.info("  - weight_distributions.png")
        logger.info("  - weight_matrices.png")
        
    except Exception as e:
        logger.error(f"❌ Exploration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()