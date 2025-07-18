# Phi-2 MLP Reconstructions

This directory contains reconstructed Multi-Layer Perceptrons (MLPs) from Microsoft's Phi-2 model.

## Contents

- `individual_mlps/`: Individual MLP state dictionaries (24 files)
- `phi2_mlps_complete.pth`: All 24 MLPs in a single file
- `mlp_metadata.json`: Architecture specifications
- `validation_report.json`: Reconstruction validation metrics

## Usage

```python
import torch
from mlp_reconstructor import Phi2MLP

# Load individual MLP
mlp = Phi2MLP(layer_idx=0)
mlp.load_state_dict(torch.load('individual_mlps/mlp_layer_0.pth'))

# Load complete collection
complete_collection = torch.load('phi2_mlps_complete.pth')
mlp_states = complete_collection['mlps']

# Use MLP
test_input = torch.randn(1, 2560)
with torch.no_grad():
    output = mlp(test_input)
```

## Architecture

Each MLP follows the pattern: `MLP(x) = W2(GELU(W1*x + b1)) + b2`

- Input dimension: 2560
- Hidden dimension: 10240  
- Output dimension: 2560
- Activation: GELU
