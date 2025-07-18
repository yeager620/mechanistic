# Phi-2 Mechanistic Interpretability with TDA

Mechanistic interpretability analysis of Microsoft's Phi-2 language model

### Project Structure
```
phi2-tda/
├── phi2_tda.py                 # Main launcher script
├── README.md                   # This file
├── RESEARCH_FINDINGS.txt       # Research summary
├── utils.py                    # Core utility functions
├── scripts/                    # Local analysis pipeline
│   ├── 00_download.py          # Model loading & weight extraction
│   ├── 01_extract.py           # Point cloud preparation
│   └── 02_tda.py               # Persistent homology computation
├── config/                     # Configuration & remote workflows
│   ├── config_remote.py        # Configuration management
│   ├── hf_data_loader.py       # HuggingFace data loading
│   └── hf_workflows.py         # Streaming analysis pipelines
├── analysis/                   # Analysis workflows
│   └── remote_analysis.py      # Remote analysis CLI
├── notebooks/                  # Jupyter notebooks
│   └── 03_viz.ipynb           # Visualization & analysis
└── pyproject.toml              # Project dependencies
```

### Workflows

#### Remote Analysis 
Analyze without storing large files locally:
```bash
# Demo workflow (~100MB cache)
uv run python phi2_tda.py remote --demo

# Analyze specific layer (~200MB cache)
uv run python phi2_tda.py remote --layer 0

# Compare all strata (~500MB cache)
uv run python phi2_tda.py remote --strata

# Memory-efficient TDA (~50MB cache)
uv run python phi2_tda.py remote --tda embedding --max-points 500
```

#### Local Analysis
Full pipeline with local storage:
```bash
# Step 1: Download and extract Phi-2 weights (~5-10 minutes, 5GB)
uv run python phi2_tda.py local download

# Step 2: Prepare point clouds for TDA (~2-3 minutes)
uv run python phi2_tda.py local extract

# Step 3: Compute persistent homology (~5-15 minutes)
uv run python phi2_tda.py local tda

# Step 4: Visualize results
jupyter notebook notebooks/03_viz.ipynb
```

#### MLP Reconstruction
Reconstruct all 32 trained MLPs from weights:
```bash
# Requires downloaded weights from Option 2, Step 1
uv run python phi2_tda.py local reconstruct-mlps --num-layers 32
```

#### Scripts 
```bash
# Remote analysis
uv run python analysis/remote_analysis.py --demo

# Local pipeline
uv run python scripts/00_download.py
uv run python scripts/01_extract.py
uv run python scripts/02_tda.py

# MLP reconstruction
uv run python mlp_reconstructor.py
```

## CLI Reference

```bash
# Remote analysis
uv run python phi2_tda.py remote --demo
uv run python phi2_tda.py remote --layer 5 --output results/
uv run python phi2_tda.py remote --batch 0 10 --output results/
uv run python phi2_tda.py remote --strata --output results/
uv run python phi2_tda.py remote --tda Q_stratum --max-points 1000

# Local analysis
uv run python phi2_tda.py local download --model microsoft/phi-2
uv run python phi2_tda.py local extract --max-points 10000
uv run python phi2_tda.py local tda --output custom_results/
uv run python phi2_tda.py local reconstruct-mlps --num-layers 32
```

```bash
# Remote analysis scripts
uv run python analysis/remote_analysis.py --demo
uv run python analysis/remote_analysis.py --layer 5

# Local analysis scripts
uv run python scripts/00_download.py --model microsoft/phi-2
uv run python scripts/01_extract.py --max-points 10000
uv run python scripts/02_tda.py --output-dir custom_results/
uv run python mlp_reconstructor.py --input phi2_weights.h5 --output mlp_reconstructions/
```

## Key Ideas

### Latent-Space Strata
Phi-2 weights can be organized into topological strata:

- **Embedding Stratum**: Token embeddings (8,000 points × 25D)
- **Attention Strata**: Query (Q), Key (K), Value (V), Output (O) projections (12,000 points × 30D each)
- **MLP Strata**: Feed-forward layers FF1/FF2 (15,000 points × 35D each)

### Topological Features
- **β₀ (Connected Components)**: Number of disconnected clusters in weight space
- **β₁ (Loops)**: Number of 1-dimensional holes/cycles
- **Persistence**: Lifespan of topological features across scales

### Memory Management
The remote workflows automatically manage memory:
- **Streaming**: Loads only required data
- **Caching**: Intelligent temporary storage
- **Cleanup**: Automatic resource cleanup
- **Batching**: Processes data in manageable chunks

### Layer-wise Analysis

```python
result = quick_layer_analysis(layer_id=0)
analyzer = StreamingAnalyzer()
batch_result = analyzer.analyze_layer_range(0, 10)
```

### Stratum Comparison
```python
strata = ["Q_stratum", "K_stratum", "V_stratum", "O_stratum"]
comparison = comparative_stratum_analysis(strata)

for stratum in strata:
    pca_result = comparison['pca_analysis']['results'][stratum]
    print(f"{stratum}: {pca_result['explained_variance_ratio'][:3]}")
```

### Memory-Efficient TDA
```python
# subsampled data
tda_result = memory_efficient_tda("embedding", max_points=1000)
print(f"β₀: {tda_result['beta_0']}, β₁: {tda_result['beta_1']}")
```

## HuggingFace Dataset

The analysis results are available at: https://huggingface.co/datasets/totalorganfailure/phi2-weights

### Model Architecture and Data Schema

```mermaid
graph TD
    %% Input Processing (Smaller Figure on Right)
    subgraph InputSection ["Input Tokenization (Pre-Transformer)"]
        A[Input Token IDs] --> B[Token Embedding Layer]
        B --> |"embed_tokens.weight<br/>[51200 x 2048]"| C[Layer Norm 1]
    end

    %% Transformer Layers (32 layers, sequential)
    C --> D[Transformer Layer Block]
    
    %% Sequential Loop Annotation
    D --> |"h.0 → h.1 → ... → h.31<br/>(Sequential, 32x)"| D
    style D stroke:#ff0000,stroke-width:2px,stroke-dasharray:5 5

    %% Detailed Transformer Layer
    subgraph TransformerLayer ["Transformer Layer (x32)"]
        D1[Input Residual Stream<br/>2048D] 
        D1 --> D2[Layer Norm]
        D2 --> |"ln_1.weight/bias<br/>[2048]"| D3[Multi-Head Self-Attention]
        
        %% Attention Detail (32 heads in parallel)
        subgraph Attention ["Multi-Head Attention (32 heads in parallel)"]
            D3 --> D4[Query Projection]
            D3 --> D5[Key Projection] 
            D3 --> D6[Value Projection]
            D4 --> |"q_proj.weight<br/>[2048 x 2048]"| D7[Q: 32 heads x 64D]
            D5 --> |"k_proj.weight<br/>[2048 x 2048]"| D8[K: 32 heads x 64D]
            D6 --> |"v_proj.weight<br/>[2048 x 2048]"| D9[V: 32 heads x 64D]
            D7 --> D10[Apply RoPE]
            D8 --> D10[Apply RoPE]
            D9 --> D11[No RoPE]
            D10 --> D12[Scaled Dot-Product Attention]
            D11 --> D12
            D12 --> D13[Concatenate Heads]
            D13 --> |"dense.weight<br/>[2048 x 2048]"| D14[Output Projection]
        end
        
        D14 --> D15[Residual Connection 1]
        D1 --> D15
        D15 --> D16[Layer Norm 2] 
        D16 --> |"ln_2.weight/bias<br/>[2048]"| D17[MLP Feed-Forward]
        
        %% MLP Detail  
        subgraph MLP ["Feed-Forward Network"]
            D17 --> D18[Linear 1]
            D18 --> |"fc1.weight/bias<br/>[2048 x 8192]"| D19[GELU Activation]
            D19 --> D20[Linear 2]
            D20 --> |"fc2.weight/bias<br/>[8192 x 2048]"| D21[Output]
        end
        
        D21 --> D22[Residual Connection 2]
        D15 --> D22
        D22 --> D23[Layer Output<br/>2048D]
    end
    
    %% Output Processing
    D23 --> E[Final Layer Norm]
    E --> |"ln_f.weight/bias<br/>[2048]"| F[Language Model Head]
    F --> |"lm_head.weight<br/>[2048 x 51200]"| G[Logits Output]
    G --> H[Vocabulary Probabilities<br/>51200 tokens]
    
    %% Styling
    classDef weightMatrix fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef activation fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef residual fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef rope fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef inputSection fill:#f0f0f0,stroke:#666,stroke-width:1px
    
    class B,D4,D5,D6,D14,D18,D20,F weightMatrix
    class C,D2,D16,E activation
    class D15,D22 residual
    class D10 rope
    class InputSection inputSection
```

#### **phi2_weights.h5** (5GB) - Raw Weight Matrices
```
Total matrices: 257 (organized by layer and component)
├── Q_0 to Q_31: Query projections (2560, 2560) - "What am I looking for?"
├── K_0 to K_31: Key projections (2560, 2560) - "What information do I contain?"
├── V_0 to V_31: Value projections (2560, 2560) - "What should I pass along?"
├── O_0 to O_31: Output projections (2560, 2560) - "How do I combine attention?"
├── FF1_0 to FF1_31: MLP expansion (10240, 2560) - "Expand for processing"
├── FF2_0 to FF2_31: MLP contraction (2560, 10240) - "Contract back to residual"
└── embed: Token embeddings (51200, 2560) - "Vocabulary → hidden states"
```

#### **weight_summary.csv** - Statistical Analysis
```
Rows: 193 (one per weight matrix)
Columns:
├── Matrix: Weight matrix name (e.g., "Q_0", "FF1_15")
├── Stratum: Functional grouping (Q, K, V, O, FF1, FF2, embedding)
├── Layer: Transformer layer number (0-31, NaN for embedding)
├── Shape: Matrix dimensions as string
├── Parameters: Total parameter count
├── Mean: Average weight value
├── Std: Standard deviation of weights
├── Min: Minimum weight value
├── Max: Maximum weight value
└── Sparsity: Fraction of near-zero weights
```

#### **point_clouds/*.npz** - Processed Point Clouds for TDA
```
7 strata, each containing normalized weight vectors:
├── embedding: (8000, 25) - Token embedding vectors
├── Q_stratum: (12000, 30) - Query weight vectors from all layers
├── K_stratum: (12000, 30) - Key weight vectors from all layers  
├── V_stratum: (12000, 30) - Value weight vectors from all layers
├── O_stratum: (12000, 30) - Output weight vectors from all layers
├── FF1_stratum: (15000, 35) - MLP expansion vectors from all layers
└── FF2_stratum: (15000, 35) - MLP contraction vectors from all layers

Each .npz file contains:
├── points: L2-normalized weight vectors (float32)
├── original_dim: Original dimensionality before PCA
├── reduced_dim: Reduced dimensionality after PCA
└── metadata: Processing parameters
```

#### **analysis_stats.json** - Analysis Metadata
```json
{
  "analysis_date": "2025-07-18",
  "model": "microsoft/phi-2",
  "parameters": "2.7B",
  "layers": 32,
  "hidden_size": 2560,
  "files_organized": 32,
  "total_size_gb": 4.9,
  "strata_analyzed": 7
}
```

### Loading Examples

```python
# specific weight matrix
with loader.get_weights_file() as f:
    q_layer_0 = f['Q_0'][:]  # Shape: (2560, 2560)

# point cloud
embedding_pc = loader.load_point_cloud("embedding")
points = embedding_pc['points']  # Shape: (8000, 25)

# summary statistics
summary = loader.load_summary_data()
ff1_stats = summary[summary['Stratum'] == 'FF1']
```

## References

- "Topological Data Analysis for Neural Network Interpretability"
- "Persistent Homology in Machine Learning"
- "Mechanistic Interpretability with Topological Methods"
- Microsoft Phi-2 Model Paper

