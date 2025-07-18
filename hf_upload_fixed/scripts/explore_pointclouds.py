"""
Explore and visualize point cloud data structure.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
import umap
from utils import logger

def explore_pointcloud_structure(pc_dir: str = "point_clouds"):
    """Explore the structure of point cloud files."""
    logger.info(f"🔍 Exploring point cloud structure in {pc_dir}/")
    
    pc_path = Path(pc_dir)
    if not pc_path.exists():
        logger.error(f"Point cloud directory not found: {pc_dir}")
        logger.info("ℹ️  Please run 'uv run python 01_extract.py' first")
        return
    
    # Find all point cloud files
    pc_files = list(pc_path.glob("*.npz"))
    if not pc_files:
        logger.error(f"No point cloud files found in {pc_dir}")
        return
    
    logger.info(f"📊 Found {len(pc_files)} point cloud files")
    
    pc_info = []
    total_points = 0
    
    for file_path in sorted(pc_files):
        if file_path.name.startswith('summary'):
            continue
            
        try:
            data = np.load(file_path)
            points = data['points']
            
            # Parse stratum info
            stratum_name = file_path.stem
            if '_' in stratum_name:
                stratum_type, layer = stratum_name.split('_', 1)
                layer_num = layer if layer.isdigit() else 'N/A'
            else:
                stratum_type = stratum_name
                layer_num = 'N/A'
            
            pc_info.append({
                'file': file_path.name,
                'stratum': stratum_type,
                'layer': layer_num,
                'points': points.shape[0],
                'dimensions': points.shape[1],
                'mean_norm': np.mean(np.linalg.norm(points, axis=1)),
                'std_norm': np.std(np.linalg.norm(points, axis=1)),
                'file_size_mb': file_path.stat().st_size / (1024**2)
            })
            
            total_points += points.shape[0]
            
            logger.info(f"  {stratum_name:15} {points.shape[0]:>6} points × {points.shape[1]:>3}D | norm μ={np.mean(np.linalg.norm(points, axis=1)):.3f}")
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    
    logger.info(f"\n📈 SUMMARY:")
    logger.info(f"  Total point clouds: {len(pc_info)}")
    logger.info(f"  Total points: {total_points:,}")
    logger.info(f"  Directory size: {sum(info['file_size_mb'] for info in pc_info):.1f} MB")
    
    return pc_info

def visualize_pointcloud_stats(pc_dir: str = "point_clouds"):
    """Visualize point cloud statistics."""
    logger.info(f"📊 Visualizing point cloud statistics from {pc_dir}/")
    
    pc_path = Path(pc_dir)
    if not pc_path.exists():
        logger.error(f"Point cloud directory not found: {pc_dir}")
        return
    
    # Collect data
    pc_data = []
    for file_path in sorted(pc_path.glob("*.npz")):
        if file_path.name.startswith('summary'):
            continue
            
        try:
            data = np.load(file_path)
            points = data['points']
            norms = np.linalg.norm(points, axis=1)
            
            pc_data.append({
                'name': file_path.stem,
                'points': points.shape[0],
                'dimensions': points.shape[1],
                'mean_norm': np.mean(norms),
                'std_norm': np.std(norms),
                'min_norm': np.min(norms),
                'max_norm': np.max(norms)
            })
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
    
    if not pc_data:
        logger.error("No valid point cloud data found")
        return
    
    df = pd.DataFrame(pc_data)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Number of points per stratum
    axes[0,0].bar(range(len(df)), df['points'], color='skyblue', alpha=0.7)
    axes[0,0].set_title('Number of Points per Stratum')
    axes[0,0].set_ylabel('Number of Points')
    axes[0,0].set_xticks(range(len(df)))
    axes[0,0].set_xticklabels(df['name'], rotation=45, ha='right')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Dimensions per stratum
    axes[0,1].bar(range(len(df)), df['dimensions'], color='lightcoral', alpha=0.7)
    axes[0,1].set_title('Dimensions per Stratum')
    axes[0,1].set_ylabel('Number of Dimensions')
    axes[0,1].set_xticks(range(len(df)))
    axes[0,1].set_xticklabels(df['name'], rotation=45, ha='right')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Mean norm per stratum
    axes[1,0].bar(range(len(df)), df['mean_norm'], color='lightgreen', alpha=0.7)
    axes[1,0].set_title('Mean L2 Norm per Stratum')
    axes[1,0].set_ylabel('Mean L2 Norm')
    axes[1,0].set_xticks(range(len(df)))
    axes[1,0].set_xticklabels(df['name'], rotation=45, ha='right')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Norm distribution comparison
    axes[1,1].scatter(df['mean_norm'], df['std_norm'], alpha=0.7, s=df['points']/20, c=df['dimensions'], cmap='viridis')
    axes[1,1].set_xlabel('Mean L2 Norm')
    axes[1,1].set_ylabel('Std L2 Norm')
    axes[1,1].set_title('Norm Statistics (size=points, color=dimensions)')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add colorbar
    scatter = axes[1,1].collections[0]
    plt.colorbar(scatter, ax=axes[1,1], label='Dimensions')
    
    plt.tight_layout()
    plt.savefig('pointcloud_stats.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    logger.info("📊 Point cloud statistics saved as 'pointcloud_stats.png'")
    
    return df

def visualize_sample_pointclouds(pc_dir: str = "point_clouds", sample_strata: list = None):
    """Visualize sample point clouds using PCA and UMAP."""
    logger.info(f"🎯 Visualizing sample point clouds from {pc_dir}/")
    
    pc_path = Path(pc_dir)
    if not pc_path.exists():
        logger.error(f"Point cloud directory not found: {pc_dir}")
        return
    
    # Default strata to visualize
    if sample_strata is None:
        sample_strata = ['embed', 'Q_0', 'K_0', 'V_0', 'FF1_0', 'FF2_0']
    
    available_files = [f.stem for f in pc_path.glob("*.npz") if not f.name.startswith('summary')]
    strata_to_plot = [s for s in sample_strata if s in available_files]
    
    if not strata_to_plot:
        logger.error("No matching strata found for visualization")
        return
    
    logger.info(f"Visualizing: {strata_to_plot}")
    
    # Create subplots
    n_strata = len(strata_to_plot)
    fig, axes = plt.subplots(2, n_strata, figsize=(4*n_strata, 8))
    if n_strata == 1:
        axes = axes.reshape(2, 1)
    
    for i, stratum in enumerate(strata_to_plot):
        file_path = pc_path / f"{stratum}.npz"
        
        try:
            data = np.load(file_path)
            points = data['points']
            
            # Subsample for visualization
            if len(points) > 2000:
                indices = np.random.choice(len(points), 2000, replace=False)
                points_viz = points[indices]
            else:
                points_viz = points
            
            # PCA visualization
            if points_viz.shape[1] > 2:
                pca = PCA(n_components=2, random_state=42)
                points_pca = pca.fit_transform(points_viz)
                explained_var = pca.explained_variance_ratio_.sum()
            else:
                points_pca = points_viz
                explained_var = 1.0
            
            axes[0,i].scatter(points_pca[:, 0], points_pca[:, 1], alpha=0.6, s=1)
            axes[0,i].set_title(f'{stratum} - PCA\n{points.shape[0]} points, {points.shape[1]}D\nExplained var: {explained_var:.2f}')
            axes[0,i].set_xlabel('PC1')
            axes[0,i].set_ylabel('PC2')
            axes[0,i].grid(True, alpha=0.3)
            
            # UMAP visualization
            if points_viz.shape[1] > 2 and len(points_viz) > 15:  # UMAP needs minimum points
                try:
                    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(points_viz)-1))
                    points_umap = reducer.fit_transform(points_viz)
                    
                    axes[1,i].scatter(points_umap[:, 0], points_umap[:, 1], alpha=0.6, s=1)
                    axes[1,i].set_title(f'{stratum} - UMAP\n{len(points_viz)} points visualized')
                    axes[1,i].set_xlabel('UMAP1')
                    axes[1,i].set_ylabel('UMAP2')
                    axes[1,i].grid(True, alpha=0.3)
                except Exception as e:
                    axes[1,i].text(0.5, 0.5, f'UMAP failed:\n{str(e)[:50]}...', 
                                  ha='center', va='center', transform=axes[1,i].transAxes)
                    axes[1,i].set_title(f'{stratum} - UMAP Failed')
            else:
                axes[1,i].scatter(points_viz[:, 0], points_viz[:, 1] if points_viz.shape[1] > 1 else np.zeros(len(points_viz)), 
                                alpha=0.6, s=1)
                axes[1,i].set_title(f'{stratum} - Raw 2D')
                axes[1,i].set_xlabel('Dim 1')
                axes[1,i].set_ylabel('Dim 2')
                axes[1,i].grid(True, alpha=0.3)
            
            logger.info(f"  Visualized {stratum}: {points.shape} → {points_viz.shape}")
            
        except Exception as e:
            logger.error(f"Failed to visualize {stratum}: {e}")
            axes[0,i].text(0.5, 0.5, f'Failed to load\n{stratum}', ha='center', va='center', transform=axes[0,i].transAxes)
            axes[1,i].text(0.5, 0.5, f'Failed to load\n{stratum}', ha='center', va='center', transform=axes[1,i].transAxes)
    
    plt.tight_layout()
    plt.savefig('pointcloud_visualizations.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    logger.info("🎯 Point cloud visualizations saved as 'pointcloud_visualizations.png'")

def inspect_single_pointcloud(pc_dir: str = "point_clouds", stratum: str = "Q_0"):
    """Detailed inspection of a single point cloud."""
    logger.info(f"🔍 Detailed inspection of {stratum} point cloud")
    
    pc_path = Path(pc_dir)
    file_path = pc_path / f"{stratum}.npz"
    
    if not file_path.exists():
        logger.error(f"Point cloud file not found: {file_path}")
        available = [f.stem for f in pc_path.glob("*.npz") if not f.name.startswith('summary')]
        logger.info(f"Available strata: {available}")
        return
    
    try:
        data = np.load(file_path)
        points = data['points']
        
        logger.info(f"📊 {stratum} POINT CLOUD ANALYSIS:")
        logger.info(f"  Shape: {points.shape}")
        logger.info(f"  Data type: {points.dtype}")
        logger.info(f"  Memory usage: {points.nbytes / (1024**2):.2f} MB")
        
        # Statistical analysis
        logger.info(f"\n📈 STATISTICAL PROPERTIES:")
        logger.info(f"  Min value: {np.min(points):.6f}")
        logger.info(f"  Max value: {np.max(points):.6f}")
        logger.info(f"  Mean: {np.mean(points):.6f}")
        logger.info(f"  Std: {np.std(points):.6f}")
        logger.info(f"  Median: {np.median(points):.6f}")
        
        # Norm analysis
        norms = np.linalg.norm(points, axis=1)
        logger.info(f"\n📏 NORM ANALYSIS:")
        logger.info(f"  Mean L2 norm: {np.mean(norms):.6f}")
        logger.info(f"  Std L2 norm: {np.std(norms):.6f}")
        logger.info(f"  Min L2 norm: {np.min(norms):.6f}")
        logger.info(f"  Max L2 norm: {np.max(norms):.6f}")
        
        # Dimensionality analysis
        logger.info(f"\n📐 DIMENSIONALITY ANALYSIS:")
        if points.shape[1] > 1:
            pca = PCA()
            pca.fit(points)
            explained_var = pca.explained_variance_ratio_
            
            logger.info(f"  Top 5 PC explained variance: {explained_var[:5]}")
            logger.info(f"  Cumulative variance (top 10): {np.cumsum(explained_var[:10])[-1]:.4f}")
            logger.info(f"  Effective dimensionality (95%): {np.argmax(np.cumsum(explained_var) >= 0.95) + 1}")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Norm distribution
        axes[0].hist(norms, bins=50, alpha=0.7, density=True)
        axes[0].set_title(f'{stratum} - L2 Norm Distribution')
        axes[0].set_xlabel('L2 Norm')
        axes[0].set_ylabel('Density')
        axes[0].grid(True, alpha=0.3)
        
        # 2. First few dimensions
        if points.shape[1] >= 2:
            axes[1].scatter(points[:, 0], points[:, 1], alpha=0.6, s=1)
            axes[1].set_title(f'{stratum} - First 2 Dimensions')
            axes[1].set_xlabel('Dimension 1')
            axes[1].set_ylabel('Dimension 2')
            axes[1].grid(True, alpha=0.3)
        
        # 3. Explained variance
        if points.shape[1] > 1:
            axes[2].plot(range(1, min(21, len(explained_var)+1)), explained_var[:20], 'o-')
            axes[2].set_title(f'{stratum} - PCA Explained Variance')
            axes[2].set_xlabel('Principal Component')
            axes[2].set_ylabel('Explained Variance Ratio')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{stratum}_detailed_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f"🔍 Detailed analysis saved as '{stratum}_detailed_analysis.png'")
        
    except Exception as e:
        logger.error(f"Failed to inspect {stratum}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to explore point cloud structure."""
    logger.info("🎯 POINT CLOUD EXPLORATION")
    logger.info("=" * 50)
    
    # Check if point clouds exist
    pc_dir = "point_clouds"
    if not Path(pc_dir).exists():
        logger.error(f"❌ Point cloud directory not found: {pc_dir}")
        logger.info("ℹ️  Please run 'uv run python 01_extract.py' first")
        return
    
    try:
        # 1. Explore structure
        logger.info("\n1️⃣ EXPLORING POINT CLOUD STRUCTURE:")
        pc_info = explore_pointcloud_structure(pc_dir)
        
        # 2. Visualize statistics
        logger.info("\n2️⃣ VISUALIZING STATISTICS:")
        df = visualize_pointcloud_stats(pc_dir)
        
        # 3. Visualize sample point clouds
        logger.info("\n3️⃣ VISUALIZING SAMPLE POINT CLOUDS:")
        visualize_sample_pointclouds(pc_dir)
        
        # 4. Detailed inspection of one stratum
        logger.info("\n4️⃣ DETAILED INSPECTION:")
        available_strata = [f.stem for f in Path(pc_dir).glob("*.npz") if not f.name.startswith('summary')]
        if available_strata:
            inspect_single_pointcloud(pc_dir, available_strata[0])
        
        logger.info("\n✅ Point cloud exploration complete!")
        logger.info("📁 Generated files:")
        logger.info("  - pointcloud_stats.png")
        logger.info("  - pointcloud_visualizations.png")
        logger.info("  - [stratum]_detailed_analysis.png")
        
    except Exception as e:
        logger.error(f"❌ Exploration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()