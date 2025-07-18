"""
Upload all analysis files to HuggingFace dataset.
"""
import os
import shutil
from pathlib import Path
from utils import logger

def organize_files_for_upload():
    """Organize all analysis files into an upload directory."""
    upload_dir = Path("phi2_analysis_upload")
    upload_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (upload_dir / "raw_data").mkdir(exist_ok=True)
    (upload_dir / "visualizations").mkdir(exist_ok=True)
    (upload_dir / "analysis_results").mkdir(exist_ok=True)
    (upload_dir / "point_clouds").mkdir(exist_ok=True)
    
    files_to_upload = []
    
    # 1. Raw extracted weights
    if Path("phi2_weights.h5").exists():
        shutil.copy("phi2_weights.h5", upload_dir / "raw_data" / "phi2_weights.h5")
        files_to_upload.append("raw_data/phi2_weights.h5")
        logger.info("✓ Added phi2_weights.h5 (5GB)")
    
    # 2. Point cloud data
    pc_dir = Path("point_clouds")
    if pc_dir.exists():
        for file in pc_dir.glob("*"):
            if file.suffix in ['.npz', '.json', '.txt']:
                dest = upload_dir / "point_clouds" / file.name
                shutil.copy(file, dest)
                files_to_upload.append(f"point_clouds/{file.name}")
                logger.info(f"✓ Added {file.name}")
    
    # 3. Visualization files
    viz_files = list(Path(".").glob("*.png"))
    for file in viz_files:
        dest = upload_dir / "visualizations" / file.name
        shutil.copy(file, dest)
        files_to_upload.append(f"visualizations/{file.name}")
        logger.info(f"✓ Added {file.name}")
    
    # 4. Analysis results
    analysis_files = list(Path(".").glob("*.csv"))
    for file in analysis_files:
        dest = upload_dir / "analysis_results" / file.name
        shutil.copy(file, dest)
        files_to_upload.append(f"analysis_results/{file.name}")
        logger.info(f"✓ Added {file.name}")
    
    # 5. Create README for the dataset
    readme_content = f"""# Phi-2 Mechanistic Interpretability Analysis

This dataset contains the complete analysis of Microsoft's Phi-2 model using Topological Data Analysis (TDA) for mechanistic interpretability.

## Contents

### Raw Data (`raw_data/`)
- `phi2_weights.h5` (5GB) - Extracted weight matrices from Phi-2 model
  - 193 weight matrices organized by stratum
  - Q, K, V, O attention projections across 32 layers
  - FF1, FF2 MLP projections across 32 layers  
  - Token embeddings

### Point Clouds (`point_clouds/`)
- `*.npz` - Normalized point clouds ready for TDA analysis
- `dimension_analysis.json` - Intrinsic dimensionality analysis
- `summary.txt` - Point cloud statistics

### Visualizations (`visualizations/`)
- `weight_distributions.png` - Weight value distributions by stratum
- `weight_matrices.png` - Heatmap visualizations of weight matrices
- `pointcloud_stats.png` - Point cloud statistics comparison
- `FF2_stratum_detailed_analysis.png` - Detailed analysis of FF2 stratum

### Analysis Results (`analysis_results/`)
- `weight_summary.csv` - Comprehensive weight statistics

## Key Findings

### Dimensionality Compression
Original Phi-2 weights (2560D) compressed to:
- Embedding: 8,000 points × 25D
- Attention (Q/K/V/O): 12,000 points × 30D each
- MLP (FF1/FF2): 15,000 points × 35D each

### Topological Insights
- **K_stratum**: Highest structure (24.6% explained variance)
- **Q_stratum**: High structure (16.6% explained variance)
- **V_stratum**: Low structure (4.7% explained variance)
- **O_stratum**: Very low structure (3.5% explained variance)
- **FF2_stratum**: Extremely compressed (4.0% explained variance)

## Usage

```python
import h5py
import numpy as np

# Load weight matrices
with h5py.File('raw_data/phi2_weights.h5', 'r') as f:
    q0_weights = f['Q_0'][:]  # Query weights for layer 0
    
# Load point clouds
embedding_pc = np.load('point_clouds/embedding.npz')['points']
```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{phi2_mechanistic_2024,
  title={{Phi-2 Mechanistic Interpretability Analysis}},
  author={{[Your Name]}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/your-username/phi2-mechanistic-analysis}}
}}
```

## License

MIT License - Original Phi-2 model by Microsoft Research

## Total Files: {len(files_to_upload)}
## Total Size: ~6.5GB
"""
    
    with open(upload_dir / "README.md", "w") as f:
        f.write(readme_content)
    files_to_upload.append("README.md")
    
    logger.info(f"✅ Organized {len(files_to_upload)} files for upload")
    return upload_dir, files_to_upload

def create_upload_script():
    """Create a script to upload to HuggingFace."""
    script_content = '''#!/bin/bash
# Upload script for Phi-2 analysis to HuggingFace

echo "🚀 Uploading Phi-2 Mechanistic Interpretability Analysis to HuggingFace"
echo "=================================================="

# Check if logged in
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "❌ Please login first: huggingface-cli login"
    exit 1
fi

# Get username
USERNAME=$(huggingface-cli whoami)
REPO_NAME="phi2-mechanistic-analysis"
REPO_ID="$USERNAME/$REPO_NAME"

echo "📋 Repository: $REPO_ID"
echo "📁 Upload directory: phi2_analysis_upload/"

# Create repository
echo "🔄 Creating repository..."
huggingface-cli repo create $REPO_NAME --type dataset --private false

# Upload files
echo "⬆️  Uploading files..."
cd phi2_analysis_upload

# Upload large files first (with progress)
echo "📤 Uploading large files..."
if [ -f "raw_data/phi2_weights.h5" ]; then
    huggingface-cli upload $REPO_ID raw_data/phi2_weights.h5 --repo-type dataset
fi

# Upload the rest
echo "📤 Uploading remaining files..."
huggingface-cli upload $REPO_ID . --repo-type dataset

echo "✅ Upload complete!"
echo "🔗 View your dataset: https://huggingface.co/datasets/$REPO_ID"
'''
    
    with open("upload_to_hf.sh", "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod("upload_to_hf.sh", 0o755)
    logger.info("✅ Created upload script: upload_to_hf.sh")

def main():
    """Main function to organize and prepare upload."""
    logger.info("🔄 Organizing files for HuggingFace upload...")
    
    try:
        upload_dir, files = organize_files_for_upload()
        create_upload_script()
        
        logger.info(f"✅ Ready for upload!")
        logger.info(f"📁 Upload directory: {upload_dir}")
        logger.info(f"📊 Files prepared: {len(files)}")
        
        # Calculate total size
        total_size = 0
        for file in upload_dir.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
        
        logger.info(f"💾 Total size: {total_size / (1024**3):.1f} GB")
        
        logger.info("\n🚀 NEXT STEPS:")
        logger.info("1. Install HuggingFace CLI: uv add huggingface-hub")
        logger.info("2. Login: uv run huggingface-cli login")
        logger.info("3. Upload: ./upload_to_hf.sh")
        logger.info("   OR manually: cd phi2_analysis_upload && huggingface-cli upload your-username/phi2-mechanistic-analysis . --repo-type dataset")
        
    except Exception as e:
        logger.error(f"Failed to organize files: {e}")
        raise

if __name__ == "__main__":
    main()