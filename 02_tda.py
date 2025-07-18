"""
Compute persistent homology on phi-2 point clouds using TDA.
"""
import numpy as np
import pathlib
import json
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
from utils import logger, log_memory_usage

def load_point_clouds(input_dir: str = "point_clouds") -> dict:
    """Load point clouds from directory."""
    logger.info(f"Loading point clouds from {input_dir}/")
    
    input_path = pathlib.Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Point cloud directory not found: {input_dir}")
    
    point_clouds = {}
    
    for file_path in input_path.glob("*.npz"):
        if file_path.name == "summary.txt":
            continue
            
        stratum_name = file_path.stem
        data = np.load(file_path)
        points = data['points']
        point_clouds[stratum_name] = points
        logger.info(f"Loaded {stratum_name}: {points.shape}")
    
    logger.info(f"Loaded {len(point_clouds)} point clouds")
    return point_clouds

def compute_persistent_homology(points: np.ndarray, stratum_name: str, maxdim: int = 1) -> dict:
    """Compute persistent homology for a point cloud."""
    logger.info(f"Computing persistent homology for {stratum_name}")
    logger.info(f"Point cloud: {points.shape[0]} points, {points.shape[1]} dimensions")
    
    log_memory_usage(f"Before PH computation ({stratum_name})")
    
    try:
        # Compute persistent homology using Ripser
        result = ripser(points, maxdim=maxdim, distance_matrix=False)
        diagrams = result['dgms']
        
        # Extract information about each homology dimension
        ph_summary = {}
        
        for dim in range(len(diagrams)):
            dgm = diagrams[dim]
            
            if len(dgm) > 0:
                # Remove infinite bars for analysis
                finite_bars = dgm[dgm[:, 1] != np.inf]
                
                ph_summary[f'H{dim}'] = {
                    'num_bars': len(dgm),
                    'num_finite_bars': len(finite_bars),
                    'num_infinite_bars': len(dgm) - len(finite_bars),
                    'max_persistence': float(np.max(finite_bars[:, 1] - finite_bars[:, 0])) if len(finite_bars) > 0 else 0.0,
                    'mean_persistence': float(np.mean(finite_bars[:, 1] - finite_bars[:, 0])) if len(finite_bars) > 0 else 0.0,
                    'birth_range': [float(np.min(dgm[:, 0])), float(np.max(dgm[:, 0]))] if len(dgm) > 0 else [0.0, 0.0],
                    'death_range': [float(np.min(finite_bars[:, 1])), float(np.max(finite_bars[:, 1]))] if len(finite_bars) > 0 else [0.0, 0.0]
                }
            else:
                ph_summary[f'H{dim}'] = {
                    'num_bars': 0,
                    'num_finite_bars': 0,
                    'num_infinite_bars': 0,
                    'max_persistence': 0.0,
                    'mean_persistence': 0.0,
                    'birth_range': [0.0, 0.0],
                    'death_range': [0.0, 0.0]
                }
            
            logger.info(f"H{dim}: {ph_summary[f'H{dim}']['num_bars']} bars "
                       f"({ph_summary[f'H{dim}']['num_finite_bars']} finite, "
                       f"{ph_summary[f'H{dim}']['num_infinite_bars']} infinite)")
        
        log_memory_usage(f"After PH computation ({stratum_name})")
        
        return {
            'diagrams': diagrams,
            'summary': ph_summary,
            'stratum_name': stratum_name,
            'point_count': points.shape[0],
            'dimension': points.shape[1]
        }
    
    except Exception as e:
        logger.error(f"Failed to compute PH for {stratum_name}: {e}")
        raise

def analyze_betti_numbers(ph_results: dict) -> dict:
    """Analyze Betti numbers across strata."""
    logger.info("Analyzing Betti numbers across strata...")
    
    betti_analysis = {}
    
    for stratum_name, result in ph_results.items():
        summary = result['summary']
        betti_analysis[stratum_name] = {
            'beta_0': summary['H0']['num_infinite_bars'],  # Connected components
            'beta_1': summary['H1']['num_infinite_bars'] if 'H1' in summary else 0,  # Loops
            'finite_H0': summary['H0']['num_finite_bars'],
            'finite_H1': summary['H1']['num_finite_bars'] if 'H1' in summary else 0,
            'max_persistence_H0': summary['H0']['max_persistence'],
            'max_persistence_H1': summary['H1']['max_persistence'] if 'H1' in summary else 0.0
        }
    
    return betti_analysis

def save_persistence_diagrams(ph_results: dict, output_dir: str = "tda_results") -> None:
    """Save persistence diagrams as images."""
    logger.info(f"Saving persistence diagrams to {output_dir}/")
    
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for stratum_name, result in ph_results.items():
        try:
            diagrams = result['diagrams']
            
            # Create persistence diagram plot
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_diagrams(diagrams, show=False, ax=ax)
            ax.set_title(f"Persistence Diagram - {stratum_name}")
            
            # Save plot
            plot_path = output_path / f"{stratum_name}_persistence_diagram.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved diagram for {stratum_name}")
            
        except Exception as e:
            logger.error(f"Failed to save diagram for {stratum_name}: {e}")

def create_betti_comparison(betti_analysis: dict, output_dir: str = "tda_results") -> None:
    """Create comparison plots of Betti numbers across strata."""
    logger.info("Creating Betti number comparison plots...")
    
    output_path = pathlib.Path(output_dir)
    
    # Extract data for plotting
    strata = list(betti_analysis.keys())
    beta_0 = [betti_analysis[s]['beta_0'] for s in strata]
    beta_1 = [betti_analysis[s]['beta_1'] for s in strata]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Beta_0 (connected components)
    ax1.bar(range(len(strata)), beta_0, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Stratum')
    ax1.set_ylabel('β₀ (Connected Components)')
    ax1.set_title('Connected Components Across Strata')
    ax1.set_xticks(range(len(strata)))
    ax1.set_xticklabels(strata, rotation=45, ha='right')
    
    # Beta_1 (loops)
    ax2.bar(range(len(strata)), beta_1, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Stratum')
    ax2.set_ylabel('β₁ (Loops)')
    ax2.set_title('Loops Across Strata')
    ax2.set_xticks(range(len(strata)))
    ax2.set_xticklabels(strata, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path / "betti_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved Betti number comparison plot")

def main():
    """Main function to compute TDA on phi-2 point clouds."""
    INPUT_DIR = "point_clouds"
    OUTPUT_DIR = "tda_results"
    
    try:
        # Load point clouds
        point_clouds = load_point_clouds(INPUT_DIR)
        
        # Compute persistent homology for each stratum
        ph_results = {}
        for stratum_name, points in point_clouds.items():
            logger.info(f"Processing {stratum_name}...")
            ph_results[stratum_name] = compute_persistent_homology(points, stratum_name)
        
        # Analyze Betti numbers
        betti_analysis = analyze_betti_numbers(ph_results)
        
        # Create output directory
        output_path = pathlib.Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        
        # Save results
        logger.info("Saving results...")
        
        # Save raw persistence diagrams (as pickle for later use)
        import pickle
        with open(output_path / "persistence_diagrams.pkl", "wb") as f:
            pickle.dump(ph_results, f)
        
        # Save Betti analysis as JSON
        with open(output_path / "betti_analysis.json", "w") as f:
            json.dump(betti_analysis, f, indent=2)
        
        # Save summary report
        with open(output_path / "tda_summary.txt", "w") as f:
            f.write("TDA Analysis Summary\n")
            f.write("===================\n\n")
            
            f.write("Betti Numbers by Stratum:\n")
            f.write("-" * 40 + "\n")
            for stratum_name, analysis in betti_analysis.items():
                f.write(f"{stratum_name}:\n")
                f.write(f"  β₀ (components): {analysis['beta_0']}\n")
                f.write(f"  β₁ (loops): {analysis['beta_1']}\n")
                f.write(f"  Max persistence H₀: {analysis['max_persistence_H0']:.6f}\n")
                f.write(f"  Max persistence H₁: {analysis['max_persistence_H1']:.6f}\n")
                f.write("\n")
            
            f.write(f"Total strata analyzed: {len(betti_analysis)}\n")
            f.write(f"Total β₀: {sum(a['beta_0'] for a in betti_analysis.values())}\n")
            f.write(f"Total β₁: {sum(a['beta_1'] for a in betti_analysis.values())}\n")
        
        # Create visualizations
        save_persistence_diagrams(ph_results, OUTPUT_DIR)
        create_betti_comparison(betti_analysis, OUTPUT_DIR)
        
        logger.info("✓ TDA computation completed successfully!")
        
        # Summary
        total_components = sum(a['beta_0'] for a in betti_analysis.values())
        total_loops = sum(a['beta_1'] for a in betti_analysis.values())
        logger.info(f"Found {total_components} connected components and {total_loops} loops across all strata")
        
    except Exception as e:
        logger.error(f"TDA pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()