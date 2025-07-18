---
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
@dataset{phi2_tda_analysis,
  title={Phi-2 Mechanistic Interpretability with Topological Data Analysis},
  author={TDA Analysis},
  year={2025},
  url={https://huggingface.co/datasets/totalorganfailure/phi2-weights}
}
```

## License

This dataset is released under the MIT License. The original Phi-2 model is subject to Microsoft's model license.

## Technical Details

- **Analysis Date**: 2025-07-18
- **Total Dataset Size**: ~5.5GB
- **Point Cloud Dimensions**: 20-35D (PCA reduced from 2560D)
- **TDA Framework**: Ripser for persistent homology computation
- **Visualization**: Matplotlib, UMAP, persistence diagrams

For detailed methodology and analysis scripts, see the `scripts/` directory.
