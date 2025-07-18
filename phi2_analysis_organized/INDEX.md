# Phi-2 Mechanistic Interpretability Analysis
## Complete Analysis Index

Generated: 2025-07-18 05:46:02

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
