"""
Upload analysis to HuggingFace with proper structure for data display.
This version ensures data files are the primary content in the repository.
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
from utils import logger

def create_hf_upload_structure():
    """Create HuggingFace upload structure with data files as primary content."""
    logger.info("🚀 Creating HuggingFace upload structure...")
    
    # Create clean upload directory
    upload_dir = Path("hf_upload_fixed")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir()
    
    # Main data files go in root for primary display
    logger.info("📦 Organizing primary data files...")
    
    # Primary data files (these will show in the main table)
    primary_files = [
        ("phi2_analysis_organized/data/raw/phi2_weights.h5", "phi2_weights.h5"),
        ("phi2_analysis_organized/analysis/reports/weight_summary.csv", "weight_summary.csv"),
        ("phi2_analysis_organized/STATS.json", "analysis_stats.json"),
        ("phi2_analysis_organized/INDEX.md", "analysis_index.md"),
    ]
    
    for src, dst in primary_files:
        src_path = Path(src)
        if src_path.exists():
            dst_path = upload_dir / dst
            shutil.copy2(src_path, dst_path)
            logger.info(f"✓ Primary data: {src} → {dst}")
    
    # Point cloud data (also primary)
    pc_dir = Path("phi2_analysis_organized/data/processed/point_clouds")
    if pc_dir.exists():
        pc_upload_dir = upload_dir / "point_clouds"
        shutil.copytree(pc_dir, pc_upload_dir)
        logger.info("✓ Point clouds copied to root level")
    
    # Visualizations go in subdirectory
    logger.info("🎨 Organizing visualizations...")
    viz_dir = upload_dir / "visualizations"
    viz_dir.mkdir()
    
    # Copy visualization files
    viz_source = Path("phi2_analysis_organized/visualizations")
    if viz_source.exists():
        for viz_file in viz_source.rglob("*.png"):
            rel_path = viz_file.relative_to(viz_source)
            dst_path = viz_dir / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(viz_file, dst_path)
            logger.info(f"✓ Visualization: {rel_path}")
    
    # Scripts and documentation in subdirectories
    logger.info("📝 Organizing scripts and documentation...")
    
    # Scripts
    scripts_dir = upload_dir / "scripts"
    scripts_source = Path("phi2_analysis_organized/scripts")
    if scripts_source.exists():
        shutil.copytree(scripts_source, scripts_dir)
        logger.info("✓ Scripts copied")
    
    # Documentation
    docs_dir = upload_dir / "documentation"
    docs_source = Path("phi2_analysis_organized/documentation")
    if docs_source.exists():
        shutil.copytree(docs_source, docs_dir)
        logger.info("✓ Documentation copied")
    
    # Create comprehensive README for the dataset
    create_hf_readme(upload_dir)
    
    return upload_dir

def create_hf_readme(upload_dir):
    """Create a comprehensive README for HuggingFace dataset."""
    logger.info("📄 Creating HuggingFace README...")
    
    readme_content = f"""---
title: Phi-2 Mechanistic Interpretability Analysis
tags:
- mechanistic-interpretability
- topological-data-analysis
- phi-2
- transformers
- persistent-homology
license: mit
size_categories:
- 1B<n<10B
language:
- en
---

# Phi-2 Mechanistic Interpretability with Topological Data Analysis

This dataset contains a complete mechanistic interpretability analysis of Microsoft's Phi-2 model using Topological Data Analysis (TDA). The analysis extracts weight matrices, organizes them into latent-space strata, and applies persistent homology to understand the model's geometric structure.

## Dataset Contents

### Primary Data Files

- **`phi2_weights.h5`** (5.0GB) - Complete weight matrices from Phi-2 model
  - 193 weight matrices organized by layer and component
  - Includes all attention (Q/K/V/O) and MLP (FF1/FF2) weights
  - Embedding and final layer weights
  - Format: HDF5 with hierarchical structure

- **`weight_summary.csv`** - Statistical summary of all weight matrices
  - Mean, std, min, max for each weight matrix
  - Shape information and parameter counts
  - Organized by model component and layer

- **`analysis_stats.json`** - Analysis metadata and statistics
  - Model information (2.7B parameters, 32 layers)
  - Processing statistics and file counts
  - Analysis completion timestamps

- **`point_clouds/`** - Processed point clouds ready for TDA
  - Normalized weight vectors organized by stratum
  - PCA-reduced dimensionality (2560D → ~30D)
  - Seven strata: embedding, Q/K/V/O attention, FF1/FF2 MLP

### Analysis Results

The analysis reveals key topological structures in Phi-2:

#### Latent-Space Strata Organization
- **Embedding Stratum**: Token embeddings (8,000 points × 25D)
- **Attention Strata**: Q/K/V/O projections (12,000 points × 30D each)
- **MLP Strata**: Feed-forward layers (15,000 points × 35D each)

#### Key Findings
- **K_stratum**: Most structured (24.6% explained variance)
- **Q_stratum**: Highly structured (16.6% explained variance)  
- **V_stratum**: Low structure (4.7% explained variance)
- **O_stratum**: Very low structure (3.5% explained variance)
- **FF2_stratum**: Extremely compressed (4.0% explained variance)

## Usage

### Loading Weight Matrices

```python
import h5py
import numpy as np

# Load complete weight dataset
with h5py.File('phi2_weights.h5', 'r') as f:
    # Access specific weight matrices
    layer_0_query = f['Q_0'][:]  # Query weights for layer 0
    layer_0_key = f['K_0'][:]    # Key weights for layer 0
    
    # List all available matrices
    print("Available weight matrices:", list(f.keys()))
    
    # Load embedding weights
    embedding = f['embedding'][:]
```

### Loading Point Clouds

```python
# Load processed point clouds
embedding_pc = np.load('point_clouds/embedding.npz')['points']
q_stratum_pc = np.load('point_clouds/Q_stratum.npz')['points']

# Point cloud metadata
with np.load('point_clouds/embedding.npz') as data:
    print("Points shape:", data['points'].shape)
    print("Original dimension:", data['original_dim'])
    print("Reduced dimension:", data['reduced_dim'])
```

### Analysis Pipeline

The complete analysis pipeline consists of:

1. **Weight Extraction**: `scripts/00_download.py`
2. **Point Cloud Preparation**: `scripts/01_extract.py` 
3. **TDA Computation**: `scripts/02_tda.py`
4. **Visualization**: `scripts/03_viz.ipynb`

## Model Architecture

- **Model**: Microsoft Phi-2
- **Parameters**: 2.7 billion
- **Layers**: 32 transformer layers
- **Hidden Size**: 2560 dimensions
- **Attention Heads**: 32 per layer
- **Vocabulary**: 51,200 tokens

## Methodology

This analysis applies Topological Data Analysis to understand the geometric structure of neural network weights:

1. **Weight Stratification**: Organize weights by functional role (attention vs MLP)
2. **Point Cloud Construction**: Normalize weight vectors and reduce dimensionality
3. **Persistent Homology**: Compute topological features across scales
4. **Interpretation**: Analyze Betti numbers and persistence diagrams

## File Structure

```
phi2-weights/
├── phi2_weights.h5              # Primary weight data (5GB)
├── weight_summary.csv           # Statistical summary
├── analysis_stats.json          # Analysis metadata
├── point_clouds/               # Processed point clouds
│   ├── embedding.npz
│   ├── Q_stratum.npz
│   ├── K_stratum.npz
│   └── ...
├── visualizations/             # Analysis visualizations
│   ├── weights/
│   ├── pointclouds/
│   └── tda/
├── scripts/                    # Analysis pipeline
└── documentation/              # Detailed documentation
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{{phi2_tda_analysis,
  title={{Phi-2 Mechanistic Interpretability with Topological Data Analysis}},
  author={{TDA Analysis}},
  year={{2025}},
  url={{https://huggingface.co/datasets/totalorganfailure/phi2-weights}}
}}
```

## License

This dataset is released under the MIT License. The original Phi-2 model is subject to Microsoft's model license.

## Technical Details

- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Total Dataset Size**: ~5.5GB
- **Point Cloud Dimensions**: 20-35D (PCA reduced from 2560D)
- **TDA Framework**: Ripser for persistent homology computation
- **Visualization**: Matplotlib, UMAP, persistence diagrams

For detailed methodology and analysis scripts, see the `scripts/` directory.
"""
    
    with open(upload_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    logger.info("✅ HuggingFace README created")

def create_upload_script(upload_dir):
    """Create upload script for the fixed structure."""
    script_content = f"""#!/bin/bash
# Upload fixed structure to HuggingFace
echo "Uploading Phi-2 analysis with proper data structure..."

USERNAME="totalorganfailure"
REPO_NAME="phi2-weights"
REPO_ID="$USERNAME/$REPO_NAME"

cd {upload_dir}

echo "Repository: $REPO_ID"
echo "Upload directory: $(pwd)"

# Upload primary data files first (these will show in main table)
echo "Uploading primary data files..."
huggingface-cli upload $REPO_ID phi2_weights.h5 --repo-type dataset
huggingface-cli upload $REPO_ID weight_summary.csv --repo-type dataset
huggingface-cli upload $REPO_ID analysis_stats.json --repo-type dataset
huggingface-cli upload $REPO_ID README.md --repo-type dataset
huggingface-cli upload $REPO_ID analysis_index.md --repo-type dataset

# Upload point clouds
echo "Uploading point clouds..."
huggingface-cli upload $REPO_ID point_clouds/ --repo-type dataset

# Upload visualizations as subdirectory
echo "Uploading visualizations..."
huggingface-cli upload $REPO_ID visualizations/ --repo-type dataset

# Upload scripts and documentation
echo "Uploading scripts and documentation..."
huggingface-cli upload $REPO_ID scripts/ --repo-type dataset
huggingface-cli upload $REPO_ID documentation/ --repo-type dataset

echo "Upload complete!"
echo "View your dataset: https://huggingface.co/datasets/$REPO_ID"
"""
    
    script_path = Path("upload_fixed.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    logger.info(f"✅ Upload script created: {script_path}")
    return script_path

def main():
    """Main function to prepare fixed HuggingFace upload."""
    logger.info("🔧 FIXING HUGGINGFACE UPLOAD STRUCTURE")
    logger.info("=" * 50)
    
    try:
        # Create fixed upload structure
        upload_dir = create_hf_upload_structure()
        
        # Create upload script
        script_path = create_upload_script(upload_dir)
        
        # Summary
        logger.info("\n✅ FIXED UPLOAD STRUCTURE READY!")
        logger.info(f"📁 Upload directory: {upload_dir}")
        logger.info(f"📄 Upload script: {script_path}")
        
        # Count files
        data_files = list(upload_dir.glob("*.h5")) + list(upload_dir.glob("*.csv")) + list(upload_dir.glob("*.json"))
        viz_files = list(upload_dir.glob("visualizations/**/*.png"))
        
        logger.info(f"\n📊 STRUCTURE SUMMARY:")
        logger.info(f"├── Primary data files: {len(data_files)} (will show in main table)")
        logger.info(f"├── Point cloud files: {len(list(upload_dir.glob('point_clouds/*.npz')))}")
        logger.info(f"├── Visualization files: {len(viz_files)} (in subdirectory)")
        logger.info(f"├── Script files: {len(list(upload_dir.glob('scripts/**/*.py')))}")
        logger.info(f"└── Documentation: {len(list(upload_dir.glob('documentation/*')))}")
        
        logger.info(f"\n🚀 NEXT STEPS:")
        logger.info(f"1. Run: ./{script_path}")
        logger.info(f"2. Check: https://huggingface.co/datasets/totalorganfailure/phi2-weights")
        logger.info(f"3. Verify data files show in main table")
        
    except Exception as e:
        logger.error(f"Failed to create fixed upload structure: {e}")
        raise

if __name__ == "__main__":
    main()