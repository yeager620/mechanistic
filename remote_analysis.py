"""
Remote analysis script demonstrating HuggingFace-first workflows.
Run analysis without storing large files locally.
"""
import numpy as np
from hf_data_loader import HFDataLoader
from hf_workflows import StreamingAnalyzer, quick_layer_analysis, comparative_stratum_analysis
from config_remote import use_remote_data, get_current_config
from utils import logger
import argparse
import json
from pathlib import Path
import sys

def demo_remote_workflow():
    """Demonstrate basic remote analysis workflow."""
    logger.info("🚀 REMOTE ANALYSIS DEMO")
    logger.info("=" * 40)
    
    # Switch to remote configuration
    config = use_remote_data("totalorganfailure/phi2-weights")
    logger.info(f"✓ Using remote data: {config.data_source.repo_id}")
    
    # Initialize remote loader
    loader = HFDataLoader(config.data_source.repo_id)
    
    # Get dataset overview
    logger.info("📊 Getting dataset overview...")
    info = loader.get_dataset_info()
    print(f"Dataset: {info['total_files']} files")
    print(f"Categories: {list(info['file_categories'].keys())}")
    
    # Load summary data (small file)
    logger.info("📈 Loading weight summary...")
    summary = loader.load_summary_data()
    print(f"Weight summary: {len(summary)} entries")
    print(f"Available matrices: {summary['Matrix'].head().tolist()}")
    
    # Quick layer analysis (streams only needed data)
    logger.info("🔍 Analyzing layer 0...")
    layer_result = quick_layer_analysis(0, config.data_source.repo_id)
    print(f"Layer 0: {layer_result['layer_statistics']['matrices_processed']} matrices processed")
    
    # Compare a few strata
    logger.info("📊 Comparing key strata...")
    strata = ["Q_stratum", "K_stratum", "embedding"]
    comparison = comparative_stratum_analysis(strata, config.data_source.repo_id)
    
    for stratum in strata:
        pca_result = comparison['pca_analysis']['results'][stratum]
        variance = pca_result['explained_variance_ratio'][:3]
        print(f"{stratum}: Top 3 PC variance = {variance}")
    
    logger.info("✅ Remote workflow demo complete!")

def analyze_specific_layer(layer_id: int, output_path: str = "remote_analysis_results"):
    """Analyze a specific layer using remote data."""
    logger.info(f"🔍 Analyzing layer {layer_id} remotely...")
    
    # Use remote configuration
    config = use_remote_data()
    
    # Run analysis
    result = quick_layer_analysis(layer_id, config.data_source.repo_id)
    
    # Save results
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)
    
    result_file = output_dir / f"layer_{layer_id}_analysis.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"✓ Layer {layer_id} analysis saved to {result_file}")
    return result

def batch_layer_analysis(start_layer: int, end_layer: int, 
                        output_path: str = "remote_analysis_results"):
    """Analyze multiple layers in batch using remote data."""
    logger.info(f"🔄 Batch analyzing layers {start_layer}-{end_layer}...")
    
    config = use_remote_data()
    analyzer = StreamingAnalyzer(config.data_source.repo_id)
    
    try:
        # Analyze layer range
        result = analyzer.analyze_layer_range(start_layer, end_layer)
        
        # Save results
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        result_file = output_dir / f"layers_{start_layer}_to_{end_layer}_analysis.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"✓ Batch analysis saved to {result_file}")
        return result
        
    finally:
        analyzer.cleanup()

def compare_all_strata(output_path: str = "remote_analysis_results"):
    """Compare all available strata using remote data."""
    logger.info("📊 Comparing all strata remotely...")
    
    config = use_remote_data()
    loader = HFDataLoader(config.data_source.repo_id)
    
    # Get available strata
    files = loader.list_available_files()
    strata = [f.split('/')[-1].replace('.npz', '') for f in files 
             if f.startswith('point_clouds/') and f.endswith('.npz')]
    
    logger.info(f"Found {len(strata)} strata: {strata}")
    
    # Run comparative analysis
    result = comparative_stratum_analysis(strata, config.data_source.repo_id)
    
    # Save results
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)
    
    result_file = output_dir / "all_strata_comparison.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"✓ Stratum comparison saved to {result_file}")
    return result

def memory_efficient_analysis(stratum_name: str, max_points: int = 1000,
                             output_path: str = "remote_analysis_results"):
    """Run memory-efficient TDA analysis on remote data."""
    from hf_workflows import memory_efficient_tda
    
    logger.info(f"🧠 Memory-efficient TDA for {stratum_name}...")
    
    config = use_remote_data()
    
    # Run TDA
    result = memory_efficient_tda(stratum_name, max_points, config.data_source.repo_id)
    
    # Save results
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)
    
    result_file = output_dir / f"{stratum_name}_tda_analysis.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"✓ TDA analysis saved to {result_file}")
    return result

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Remote phi2 analysis using HuggingFace")
    parser.add_argument("--demo", action="store_true", help="Run demo workflow")
    parser.add_argument("--layer", type=int, help="Analyze specific layer")
    parser.add_argument("--batch", nargs=2, type=int, metavar=("START", "END"),
                       help="Analyze layer range")
    parser.add_argument("--strata", action="store_true", help="Compare all strata")
    parser.add_argument("--tda", type=str, help="Run TDA on specific stratum")
    parser.add_argument("--max-points", type=int, default=1000, 
                       help="Max points for TDA analysis")
    parser.add_argument("--output", default="remote_analysis_results",
                       help="Output directory")
    parser.add_argument("--repo-id", default="totalorganfailure/phi2-weights",
                       help="HuggingFace repository ID")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_remote_workflow()
    
    elif args.layer is not None:
        analyze_specific_layer(args.layer, args.output)
    
    elif args.batch:
        start, end = args.batch
        batch_layer_analysis(start, end, args.output)
    
    elif args.strata:
        compare_all_strata(args.output)
    
    elif args.tda:
        memory_efficient_analysis(args.tda, args.max_points, args.output)
    
    else:
        print("No analysis specified. Use --help for options.")
        print("Example: python remote_analysis.py --demo")

if __name__ == "__main__":
    main()