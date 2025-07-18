"""
Organize all data and visualization outputs properly.
"""
import os
import shutil
from pathlib import Path
import json
from datetime import datetime
from utils import logger

def create_organized_structure():
    """Create a clean, organized directory structure."""
    
    # Main analysis directory
    analysis_dir = Path("phi2_analysis_organized")
    analysis_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    subdirs = {
        "data": "Raw and processed data files",
        "data/raw": "Original extracted weights",
        "data/processed": "Point clouds and processed data",
        "visualizations": "All plots and charts",
        "visualizations/weights": "Weight-related visualizations",
        "visualizations/pointclouds": "Point cloud visualizations",
        "visualizations/tda": "TDA results visualizations",
        "analysis": "Analysis results and reports",
        "analysis/reports": "Summary reports and statistics",
        "analysis/tda_results": "TDA computation results",
        "notebooks": "Jupyter notebooks",
        "scripts": "Analysis scripts",
        "documentation": "Documentation and README files"
    }
    
    for subdir, description in subdirs.items():
        (analysis_dir / subdir).mkdir(parents=True, exist_ok=True)
        # Create a description file
        with open(analysis_dir / subdir / "_README.txt", "w") as f:
            f.write(f"Directory: {subdir}\n")
            f.write(f"Purpose: {description}\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return analysis_dir

def organize_data_files(analysis_dir):
    """Organize all data files."""
    logger.info("📁 Organizing data files...")
    
    files_moved = []
    
    # Raw data
    raw_files = [
        ("phi2_weights.h5", "data/raw/phi2_weights.h5"),
    ]
    
    for src, dst in raw_files:
        if Path(src).exists():
            dst_path = analysis_dir / dst
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_path)
            files_moved.append(f"✓ {src} → {dst}")
            logger.info(f"Moved {src} to {dst}")
    
    # Processed data (point clouds)
    pc_dir = Path("point_clouds")
    if pc_dir.exists():
        processed_dir = analysis_dir / "data/processed"
        shutil.copytree(pc_dir, processed_dir / "point_clouds", dirs_exist_ok=True)
        files_moved.append("✓ point_clouds/ → data/processed/point_clouds/")
        logger.info("Moved point clouds to processed data")
    
    return files_moved

def organize_visualizations(analysis_dir):
    """Organize all visualization files."""
    logger.info("📊 Organizing visualizations...")
    
    files_moved = []
    
    # Weight visualizations
    weight_viz_files = [
        ("weight_distributions.png", "visualizations/weights/weight_distributions.png"),
        ("weight_matrices.png", "visualizations/weights/weight_matrices.png"),
    ]
    
    # Point cloud visualizations
    pc_viz_files = [
        ("pointcloud_stats.png", "visualizations/pointclouds/pointcloud_stats.png"),
        ("pointcloud_visualizations.png", "visualizations/pointclouds/pointcloud_visualizations.png"),
    ]
    
    # Detailed analysis plots
    detailed_files = []
    for file in Path(".").glob("*_detailed_analysis.png"):
        detailed_files.append((str(file), f"visualizations/pointclouds/{file.name}"))
    
    # TDA visualizations (if they exist)
    tda_viz_files = []
    tda_dir = Path("tda_results")
    if tda_dir.exists():
        for file in tda_dir.glob("*.png"):
            tda_viz_files.append((str(file), f"visualizations/tda/{file.name}"))
    
    all_viz_files = weight_viz_files + pc_viz_files + detailed_files + tda_viz_files
    
    for src, dst in all_viz_files:
        if Path(src).exists():
            dst_path = analysis_dir / dst
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_path)
            files_moved.append(f"✓ {src} → {dst}")
            logger.info(f"Moved {src} to {dst}")
    
    return files_moved

def organize_analysis_results(analysis_dir):
    """Organize analysis results and reports."""
    logger.info("📈 Organizing analysis results...")
    
    files_moved = []
    
    # Reports and statistics
    report_files = [
        ("weight_summary.csv", "analysis/reports/weight_summary.csv"),
    ]
    
    # TDA results
    tda_dir = Path("tda_results")
    if tda_dir.exists():
        tda_result_dir = analysis_dir / "analysis/tda_results"
        shutil.copytree(tda_dir, tda_result_dir, dirs_exist_ok=True)
        files_moved.append("✓ tda_results/ → analysis/tda_results/")
        logger.info("Moved TDA results")
    
    for src, dst in report_files:
        if Path(src).exists():
            dst_path = analysis_dir / dst
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_path)
            files_moved.append(f"✓ {src} → {dst}")
            logger.info(f"Moved {src} to {dst}")
    
    return files_moved

def organize_scripts_and_notebooks(analysis_dir):
    """Organize scripts and notebooks."""
    logger.info("📝 Organizing scripts and notebooks...")
    
    files_moved = []
    
    # Core scripts
    script_files = [
        ("00_download.py", "scripts/00_download.py"),
        ("01_extract.py", "scripts/01_extract.py"),
        ("02_tda.py", "scripts/02_tda.py"),
        ("utils.py", "scripts/utils.py"),
        ("explore_weights.py", "scripts/explore_weights.py"),
        ("explore_pointclouds.py", "scripts/explore_pointclouds.py"),
        ("check_status.py", "scripts/check_status.py"),
        ("upload_analysis.py", "scripts/upload_analysis.py"),
    ]
    
    # Notebooks
    notebook_files = [
        ("03_viz.ipynb", "notebooks/03_viz.ipynb"),
    ]
    
    # Test scripts
    test_files = [
        ("test_arch.py", "scripts/tests/test_arch.py"),
        ("test_small.py", "scripts/tests/test_small.py"),
        ("test_pipeline.py", "scripts/tests/test_pipeline.py"),
    ]
    
    all_files = script_files + notebook_files + test_files
    
    for src, dst in all_files:
        if Path(src).exists():
            dst_path = analysis_dir / dst
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_path)
            files_moved.append(f"✓ {src} → {dst}")
            logger.info(f"Moved {src} to {dst}")
    
    return files_moved

def organize_documentation(analysis_dir):
    """Organize documentation."""
    logger.info("📚 Organizing documentation...")
    
    files_moved = []
    
    doc_files = [
        ("README.md", "documentation/README.md"),
        ("plan.md", "documentation/plan.md"),
        ("pyproject.toml", "documentation/pyproject.toml"),
        ("uv.lock", "documentation/uv.lock"),
    ]
    
    for src, dst in doc_files:
        if Path(src).exists():
            dst_path = analysis_dir / dst
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_path)
            files_moved.append(f"✓ {src} → {dst}")
            logger.info(f"Moved {src} to {dst}")
    
    return files_moved

def create_master_index(analysis_dir):
    """Create a master index file."""
    logger.info("📋 Creating master index...")
    
    index_content = f"""# Phi-2 Mechanistic Interpretability Analysis
## Complete Analysis Index

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Directory Structure

### 📁 Data
- `data/raw/` - Original extracted weights from Phi-2 model
  - `phi2_weights.h5` - 193 weight matrices (5GB)
- `data/processed/` - Processed point clouds ready for TDA
  - `point_clouds/` - Normalized point clouds by stratum

### 📊 Visualizations
- `visualizations/weights/` - Weight matrix visualizations
  - `weight_distributions.png` - Distribution histograms
  - `weight_matrices.png` - Heatmap visualizations
- `visualizations/pointclouds/` - Point cloud visualizations
  - `pointcloud_stats.png` - Statistics comparison
  - `*_detailed_analysis.png` - Detailed stratum analysis
- `visualizations/tda/` - TDA result visualizations
  - Persistence diagrams and Betti number plots

### 📈 Analysis Results
- `analysis/reports/` - Summary reports and statistics
  - `weight_summary.csv` - Comprehensive weight statistics
- `analysis/tda_results/` - TDA computation results
  - `persistence_diagrams.pkl` - Raw persistence diagrams
  - `betti_analysis.json` - Betti number analysis

### 📝 Scripts & Notebooks
- `scripts/` - Analysis pipeline scripts
  - `00_download.py` - Model weight extraction
  - `01_extract.py` - Point cloud preparation
  - `02_tda.py` - TDA computation
  - `utils.py` - Helper functions
  - `explore_*.py` - Exploration scripts
- `notebooks/` - Jupyter notebooks
  - `03_viz.ipynb` - Interactive visualization

### 📚 Documentation
- `documentation/` - Project documentation
  - `README.md` - Main documentation
  - `plan.md` - Original analysis plan

## Key Findings Summary

### Model Architecture
- **Phi-2**: 2.7B parameters, 32 layers, 32 attention heads
- **Hidden dimension**: 2560D
- **Vocabulary**: 51,200 tokens

### Point Cloud Statistics
- **Embedding**: 8,000 points × 25D
- **Attention strata**: 12,000 points × 30D each (Q, K, V, O)
- **MLP strata**: 15,000 points × 35D each (FF1, FF2)

### Topological Structure
- **K_stratum**: Most structured (24.6% explained variance)
- **Q_stratum**: Highly structured (16.6% explained variance)
- **V_stratum**: Low structure (4.7% explained variance)
- **O_stratum**: Very low structure (3.5% explained variance)
- **FF2_stratum**: Extremely compressed (4.0% explained variance)

## Usage

### Quick Access
- **View all visualizations**: `visualizations/`
- **Load point clouds**: `data/processed/point_clouds/`
- **Run analysis**: `scripts/`
- **Interactive exploration**: `notebooks/03_viz.ipynb`

### Data Loading Examples
```python
# Load weight matrices
import h5py
with h5py.File('data/raw/phi2_weights.h5', 'r') as f:
    q0_weights = f['Q_0'][:]

# Load point clouds
import numpy as np
embedding = np.load('data/processed/point_clouds/embedding.npz')['points']
```

## Analysis Pipeline
1. **Extract weights**: `scripts/00_download.py`
2. **Prepare point clouds**: `scripts/01_extract.py`
3. **Compute TDA**: `scripts/02_tda.py`
4. **Visualize results**: `notebooks/03_viz.ipynb`

## Contact & Citation
Analysis completed using TDA for mechanistic interpretability of language models.
"""
    
    with open(analysis_dir / "INDEX.md", "w") as f:
        f.write(index_content)
    
    logger.info("Created master index file")

def create_quick_stats(analysis_dir):
    """Create a quick statistics summary."""
    logger.info("📊 Creating quick statistics...")
    
    stats = {
        "analysis_date": datetime.now().isoformat(),
        "model": "microsoft/phi-2",
        "parameters": "2.7B",
        "layers": 32,
        "hidden_size": 2560,
        "files_organized": 0,
        "total_size_gb": 0,
        "strata_analyzed": 7
    }
    
    # Count files and calculate size
    total_size = 0
    file_count = 0
    
    for file in analysis_dir.rglob("*"):
        if file.is_file() and not file.name.startswith("_"):
            file_count += 1
            total_size += file.stat().st_size
    
    stats["files_organized"] = file_count
    stats["total_size_gb"] = round(total_size / (1024**3), 2)
    
    # Save stats
    with open(analysis_dir / "STATS.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Statistics: {file_count} files, {stats['total_size_gb']}GB total")
    
    return stats

def main():
    """Main function to organize all outputs."""
    logger.info("🗂️  ORGANIZING PHI-2 ANALYSIS OUTPUTS")
    logger.info("=" * 50)
    
    try:
        # Create organized structure
        analysis_dir = create_organized_structure()
        logger.info(f"Created organized structure: {analysis_dir}")
        
        # Organize different types of files
        data_files = organize_data_files(analysis_dir)
        viz_files = organize_visualizations(analysis_dir)
        analysis_files = organize_analysis_results(analysis_dir)
        script_files = organize_scripts_and_notebooks(analysis_dir)
        doc_files = organize_documentation(analysis_dir)
        
        # Create master documentation
        create_master_index(analysis_dir)
        stats = create_quick_stats(analysis_dir)
        
        # Summary
        total_files = len(data_files + viz_files + analysis_files + script_files + doc_files)
        
        logger.info("\n✅ ORGANIZATION COMPLETE!")
        logger.info(f"📁 Organized directory: {analysis_dir}")
        logger.info(f"📊 Total files moved: {total_files}")
        logger.info(f"💾 Total size: {stats['total_size_gb']}GB")
        logger.info(f"🔍 View complete index: {analysis_dir}/INDEX.md")
        
        logger.info("\n📋 ORGANIZED STRUCTURE:")
        logger.info("├── data/")
        logger.info("│   ├── raw/ (original weights)")
        logger.info("│   └── processed/ (point clouds)")
        logger.info("├── visualizations/")
        logger.info("│   ├── weights/")
        logger.info("│   ├── pointclouds/")
        logger.info("│   └── tda/")
        logger.info("├── analysis/")
        logger.info("│   ├── reports/")
        logger.info("│   └── tda_results/")
        logger.info("├── scripts/")
        logger.info("├── notebooks/")
        logger.info("└── documentation/")
        
    except Exception as e:
        logger.error(f"Organization failed: {e}")
        raise

if __name__ == "__main__":
    main()