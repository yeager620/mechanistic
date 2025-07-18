#!/usr/bin/env python3
"""
Phi-2 TDA Analysis Launcher
Main entry point for all analysis workflows.
"""
import argparse
import sys
import os
import importlib.util
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(description="Phi-2 Mechanistic Interpretability with TDA")
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Local analysis commands
    local_parser = subparsers.add_parser('local', help='Local analysis pipeline')
    local_subparsers = local_parser.add_subparsers(dest='local_command')
    
    # Download weights
    download_parser = local_subparsers.add_parser('download', help='Download and extract model weights')
    download_parser.add_argument('--model', default='microsoft/phi-2', help='Model ID')
    download_parser.add_argument('--output', default='phi2_weights.h5', help='Output file')
    
    # Extract point clouds
    extract_parser = local_subparsers.add_parser('extract', help='Extract point clouds from weights')
    extract_parser.add_argument('--input', default='phi2_weights.h5', help='Input weights file')
    extract_parser.add_argument('--output', default='point_clouds', help='Output directory')
    extract_parser.add_argument('--max-points', type=int, default=15000, help='Max points per stratum')
    
    # Compute TDA
    tda_parser = local_subparsers.add_parser('tda', help='Compute persistent homology')
    tda_parser.add_argument('--input', default='point_clouds', help='Input point clouds directory')
    tda_parser.add_argument('--output', default='tda_results', help='Output directory')
    
    # Reconstruct MLPs
    reconstruct_parser = local_subparsers.add_parser('reconstruct-mlps', help='Reconstruct MLPs from weights')
    reconstruct_parser.add_argument('--input', default='phi2_weights.h5', help='Input weights file')
    reconstruct_parser.add_argument('--output', default='mlp_reconstructions', help='Output directory')
    reconstruct_parser.add_argument('--num-layers', type=int, default=32, help='Number of layers to reconstruct')
    
    # Remote analysis commands
    remote_parser = subparsers.add_parser('remote', help='Remote analysis workflows')
    remote_parser.add_argument('--demo', action='store_true', help='Run demo workflow')
    remote_parser.add_argument('--layer', type=int, help='Analyze specific layer')
    remote_parser.add_argument('--batch', nargs=2, type=int, metavar=('START', 'END'), help='Analyze layer range')
    remote_parser.add_argument('--strata', action='store_true', help='Compare all strata')
    remote_parser.add_argument('--tda', type=str, help='Run TDA on specific stratum')
    remote_parser.add_argument('--max-points', type=int, default=1000, help='Max points for TDA')
    remote_parser.add_argument('--output', default='remote_analysis_results', help='Output directory')
    remote_parser.add_argument('--repo-id', default='totalorganfailure/phi2-weights', help='HuggingFace repository ID')
    
    args = parser.parse_args()
    
    if args.command == 'local':
        if args.local_command == 'download':
            # Load the download script dynamically
            spec = importlib.util.spec_from_file_location("download_script", "scripts/00_download.py")
            download_script = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(download_script)
            sys.argv = ['00_download.py', '--model', args.model, '--output', args.output]
            download_script.main()
        elif args.local_command == 'extract':
            # Load the extract script dynamically
            spec = importlib.util.spec_from_file_location("extract_script", "scripts/01_extract.py")
            extract_script = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(extract_script)
            sys.argv = ['01_extract.py', '--input', args.input, '--output', args.output, '--max-points', str(args.max_points)]
            extract_script.main()
        elif args.local_command == 'tda':
            # Load the TDA script dynamically
            spec = importlib.util.spec_from_file_location("tda_script", "scripts/02_tda.py")
            tda_script = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tda_script)
            sys.argv = ['02_tda.py', '--input', args.input, '--output', args.output]
            tda_script.main()
        elif args.local_command == 'reconstruct-mlps':
            from mlp_reconstructor import extract_and_reconstruct_mlps
            results = extract_and_reconstruct_mlps(args.input, args.output, args.num_layers)
            print(f"MLP reconstruction completed! Output directory: {results['output_directory']}")
        else:
            local_parser.print_help()
    
    elif args.command == 'remote':
        from analysis.remote_analysis import main as remote_main
        # Build sys.argv for remote_analysis
        sys.argv = ['remote_analysis.py']
        if args.demo:
            sys.argv.append('--demo')
        if args.layer is not None:
            sys.argv.extend(['--layer', str(args.layer)])
        if args.batch:
            sys.argv.extend(['--batch', str(args.batch[0]), str(args.batch[1])])
        if args.strata:
            sys.argv.append('--strata')
        if args.tda:
            sys.argv.extend(['--tda', args.tda])
        if args.max_points != 1000:
            sys.argv.extend(['--max-points', str(args.max_points)])
        if args.output != 'remote_analysis_results':
            sys.argv.extend(['--output', args.output])
        if args.repo_id != 'totalorganfailure/phi2-weights':
            sys.argv.extend(['--repo-id', args.repo_id])
        
        remote_main()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()