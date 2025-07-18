# Phi-2 Mechanistic Interpretability Analysis

This dataset contains the data for analysis of Microsoft's Phi-2 model weights using Topological Data Analysis for mechanistic interpretability.

## Contents

### Raw Data (`raw_data/`)
- `phi2_weights.h5` (5GB) - Extracted weight matrices from Phi-2 model
  - 193 weight matrices organized by stratum
  - Q, K, V, O attention projections across 32 layers
  - FF1, FF2 MLP projections across 32 layers  
  - Token embeddings

### Point Clouds (`point_clouds/`)
- `*.npz` - Normalized point clouds
- `dimension_analysis.json` - Intrinsic dimensionality analysis
- `summary.txt` - Point cloud statistics

### Visualizations (`visualizations/`)
- `weight_distributions.png` - Weight value distributions by stratum
- `weight_matrices.png` - Heatmap visualizations of weight matrices
- `pointcloud_stats.png` - Point cloud statistics comparison
- `FF2_stratum_detailed_analysis.png` - Detailed analysis of FF2 stratum

### Results (`analysis_results/`)
- `weight_summary.csv` - Comprehensive weight statistics
