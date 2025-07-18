Below is a concrete, end-to-end recipe that takes an off-the-shelf Hugging Face model, converts its **raw weights** into the objects defined by your latent-space framework, and pushes them through a lightweight TDA pipeline for exploratory interpretability.

---

## 1  Pick a tractable, fully open model

| Why this choice matters                                                                                                                                                                                | Recommended model                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| • Small enough to fit on a single GPU/CPU box<br>• Same architecture family as larger Llama/Mistral models (so lessons transfer)<br>• Weights published in both **PyTorch‐`safetensors`** and **GGUF** | **`TinyLlama/TinyLlama_v1.1`** (≈1.1 B params) on Hugging Face ([Hugging Face][1]) |

> *Tip:* if you have >48 GB VRAM and want a closer look at “real” attention sparsity, substitute `mistralai/Mistral-7B-v0.1` or `meta-llama/Llama-3-8B`.

---

## 2  Download & load weights (PyTorch / `transformers`)

```bash
# Using uv for fast package installation
uv add "transformers>=4.35.0" safetensors "torch>=2.0" einops
```

```python
from transformers import AutoModelForCausalLM, AutoConfig
import torch, numpy as np, h5py, pathlib

MODEL_ID = "TinyLlama/TinyLlama_v1.1"
model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,           # keep FP32 for faithful geometry
        low_cpu_mem_usage=True,              # streams layers from disk
        trust_remote_code=True              # needed by some ports
)
config: AutoConfig = model.config
d, L, H = config.hidden_size, config.num_hidden_layers, config.num_attention_heads
```

*Why `safetensors`?*  The format is memory-map compatible and guarantees no arbitrary code execution during load.

---

## 3  Map weights to the **latent-space strata**

Your framework distinguishes four main strata per layer:

1. **Residual stream** basis vectors (rows of \$W\_{\text{embed}}\$)
2. **Attention projections:** \$W\_Q,W\_K,W\_V,W\_O\$
3. **Feed-forward projections:** \$W\_1,W\_2\$ *(plus expert slices if MoE)*
4. **Normalisation/gating scalars** (optional)

```python
layer_dict = {}
for ℓ, block in enumerate(model.model.layers):
    # --- Attention block -------------------------------------------------
    layer_dict[f"Q_{ℓ}"] = block.self_attn.q_proj.weight.cpu().numpy()
    layer_dict[f"K_{ℓ}"] = block.self_attn.k_proj.weight.cpu().numpy()
    layer_dict[f"V_{ℓ}"] = block.self_attn.v_proj.weight.cpu().numpy()
    layer_dict[f"O_{ℓ}"] = block.self_attn.o_proj.weight.cpu().numpy()
    
    # --- MLP / MoE block --------------------------------------------------
    w1 = block.mlp.up_proj.weight          # TinyLlama uses gated-GELU
    w2 = block.mlp.down_proj.weight
    layer_dict[f"FF1_{ℓ}"] = w1.cpu().numpy()
    layer_dict[f"FF2_{ℓ}"] = w2.cpu().numpy()
```

Store them in an HDF5 container so downstream tools (Ripser, GUDHI, KeplerMapper) can memory-map chunks instead of loading ∼GB into RAM:

```python
path = pathlib.Path("tinyllama_weights.h5")
with h5py.File(path, "w") as f:
    for name, arr in layer_dict.items():
        f.create_dataset(name, data=arr, compression="gzip")
```

---

## 4  Prepare point clouds for TDA

Each **row vector** of a weight matrix lives in the ambient space \$\mathbb R^d\$ and, in your language, is an element of the stratum
$\mathcal Z_Q^{(\ell)}\subset \mathbb R^{d}\;\;(\text{resp.\ } \mathcal Z_K,\mathcal Z_V,\dots).$

A pragmatic pipeline:

1. **Row-wise normalise** to remove scale ambiguity:
   $\tilde w_i = w_i / \|w_i\|_2.$
2. **(Optional) Dimensionality reduction**—UMAP or PCA to 20–50 D—to keep Ripser runtime reasonable.
3. **Sub-sample** if >50 k points (e.g. uniform or k-means coresets).

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import random, umap

mat = layer_dict["Q_0"]          # layer 0 queries as example
rows = normalize(mat, axis=1)
rows = rows[random.sample(range(rows.shape[0]), 8000)]   # subsample
rows = PCA(n_components=30, svd_solver="randomized").fit_transform(rows)
```

---

## 5  Compute persistent homology

Two widely used Python stacks work out of the box:

| Library                              | Pros                                                     | Citation             |
| ------------------------------------ | -------------------------------------------------------- | -------------------- |
| **Ripser.py** (`pip install ripser`) | Ultra-fast Vietoris–Rips up to H₁; easy diagrams         | ([GitHub][2])        |
| **GUDHI** (`pip install gudhi`)      | Many filtrations (alpha, witness), bottleneck/W₂ metrics | ([GUDHI library][3]) |

Example with Ripser:

```python
from ripser import ripser
from persim import plot_diagrams
dgms = ripser(rows, maxdim=1, distance_matrix=False)['dgms']
plot_diagrams(dgms, show=True)
```

Interpretation hints (linking back to your theory):

* **\$H\_0\$ clusters** often align with “directional families” of Q-vectors that fire on related syntactic roles.
* **\$H\_1\$ loops** can reveal cyclic redundancy (e.g. duplicate heads) or sign-flip symmetries inside a layer.
* Track Betti numbers across \$\ell\$ using the *layer index filtration* \$f(z)=\operatorname{layer}(z)\$ exactly as § 6 of your manuscript—this is a first empirical check of Theorem 4.

---

## 6  Visual EDA dashboard (optional)

* **UMAP-over-weights** coloured by head or layer to spot outliers.
* **Mapper graphs** (`kepler-mapper`) on the same point cloud give a TDA-lite overview without heavy PH.
* **SVD scree plots** per stratum validate the intrinsic-dimension claim
  $\dim_\text{intrinsic}(\mathcal Z_R^{(\ell)})\ll d.$

---

## 7  Putting it together—minimal pipeline skeleton

```bash
# Setup with uv (fast Python package manager)
# Install uv if not already installed: curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project and virtual environment
uv init tda-phi2
cd tda-phi2
uv venv --python 3.10
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
uv add "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu  # or cu118 for CUDA
uv add "transformers>=4.35.0"  # Required for phi-2 support
uv add safetensors numpy scikit-learn umap-learn ripser persim h5py pandas
uv add accelerate psutil  # For device_map and memory monitoring

# For GPU support (if available)
# uv add "torch>=2.0" --index-url https://download.pytorch.org/whl/cu118
```

1. `00_download.py` – grabs weights, validates architecture, caches to HDF5.
2. `01_extract.py` – normalises rows, saves `layer_X_part.npz`.
3. `02_tda.py` – runs Ripser/GUDHI, stores diagrams as `json` or `pickle`.
4. `03_viz.ipynb` – UMAP, barcodes, Mapper graph widgets.
5. `utils.py` – memory monitoring, error handling, architecture validation.

```bash
# Run pipeline with uv
uv run python 00_download.py
uv run python 01_extract.py
uv run python 02_tda.py
# or activate venv once: source .venv/bin/activate
```

All steps are independent, so you can swap in larger models later.

---

## Why this aligns with your axioms

* **Stratification** – each weight row belongs to exactly one \$\mathcal Z\_\*\$ stratum.
* **Filtration by layer** – the HDF5 group hierarchy `layer/stratum/...` mirrors the function
  $f(z)=\text{layer}(z)$ needed for your Persistence Module (Def. 4).
* **Continuity & Lipschitz** – by analysing *weights* (rather than activations) you bypass the LayerNorm variance caveat but still inherit Lipschitz bounds on linear maps, so Stability Thm 3 applies.

---

### In short

1. **Choose** a fully open model (TinyLlama ≈1 B params).
2. **Load** native `safetensors` with `transformers`.
3. **Split & normalise** weights into the theory’s strata.
4. **Store** in HDF5 → **Sample** rows → **Compute** PH via Ripser/GUDHI.
5. **Visualise** clusters/loops and track Betti curves across layers.

That pipeline turns the high-level framework into a hands-on interpretability workflow you can iterate on today. Happy topologising!

[1]: https://huggingface.co/TinyLlama/TinyLlama_v1.1?utm_source=chatgpt.com "TinyLlama/TinyLlama_v1.1 - Hugging Face"
[2]: https://github.com/scikit-tda/ripser.py?utm_source=chatgpt.com "scikit-tda/ripser.py: A Lean Persistent Homology Library for Python"
[3]: https://gudhi.inria.fr/python/latest/?utm_source=chatgpt.com "GUDHI Python modules documentation"


### Switching the pipeline from **TinyLlama v1.1** → **`microsoft/phi-2`**

The overall seven-step recipe is **unchanged** — download → slice weights into strata → normalise → TDA → visualise.
Only two things differ in practice:

| What changes                 | Why it changes in ϕ-2                                                                                                              | One-line fix                                                                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Module names & fused QKV** | ϕ-2’s attention block is called `mixer` and stores a *single* weight matrix `Wqkv` of shape `[3·d, d]`, plus an `out_proj` matrix. | Split `Wqkv` into three chunks (Q ∥ K ∥ V) before writing to HDF5.                    |
| **Memory footprint**         | 2.7 B params ≈ 10.8 GB in FP32.                                                                                                    | Use `torch_dtype=torch.bfloat16` *or* stream layer-by-layer to stay inside 16 GB RAM. |

Everything else (embedding stratum, dense MLP with `fc1/fc2`, RMSNorm weights, etc.) plugs straight into the same latent-space framework.

---

## 1  Install & load

```bash
# Using uv for fast package installation
uv add "transformers>=4.35.0" safetensors "torch>=2.0" einops
```

```python
from transformers import AutoModelForCausalLM, AutoConfig
import torch, numpy as np, h5py, pathlib

MODEL_ID = "microsoft/phi-2"

# Memory-efficient loading for phi-2 (2.7B params)
model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,  # Critical: reduces memory from ~11GB to ~5.4GB
        device_map="auto",           # Automatic GPU/CPU distribution if needed
        low_cpu_mem_usage=True,
        trust_remote_code=True       # required for the custom Phi class
)
config: AutoConfig = model.config
d, L, H = config.hidden_size, config.num_hidden_layers, config.n_head  # Phi uses n_head

# Validate architecture before proceeding
print(f"Model class: {model.__class__.__name__}")
print(f"Hidden size: {d}, Layers: {L}, Heads: {H}")
print(f"First layer type: {type(model.transformer.h[0])}")
```

---

## 2  Slice weights into latent-space strata

```python
# Memory-efficient layer-by-layer extraction
layer_dict = {}

# --- token & position embeddings -----------------------------------------
try:
    embed_weight = model.transformer.embd.wte.weight.cpu().float().numpy()
    layer_dict["embed"] = embed_weight
    print(f"Extracted embeddings: {embed_weight.shape}")
except AttributeError as e:
    print(f"Warning: Could not extract embeddings: {e}")
    # Fallback: inspect available attributes
    print(f"Available embd attributes: {dir(model.transformer.embd)}")

# --- loop over transformer blocks ----------------------------------------
for ℓ, block in enumerate(model.transformer.h):
    try:
        # Validate phi-2 architecture
        if not hasattr(block, 'mixer'):
            raise ValueError(f"Layer {ℓ} missing 'mixer' attribute. Available: {dir(block)}")
        
        mixer = block.mixer
        if hasattr(mixer, 'Wqkv'):
            # ϕ-2 merges Q,K,V → Wqkv  (shape [3d, d])
            Wqkv = mixer.Wqkv.weight.cpu().float().numpy()
            if Wqkv.shape[0] != 3 * d:
                print(f"Warning: Wqkv shape {Wqkv.shape} != expected [3*{d}, {d}]")
            
            q_mat, k_mat, v_mat = np.split(Wqkv, 3, axis=0)
            layer_dict[f"Q_{ℓ}"] = q_mat
            layer_dict[f"K_{ℓ}"] = k_mat
            layer_dict[f"V_{ℓ}"] = v_mat
            
        elif hasattr(mixer, 'q_proj') and hasattr(mixer, 'k_proj') and hasattr(mixer, 'v_proj'):
            # Fallback: separate Q,K,V projections
            layer_dict[f"Q_{ℓ}"] = mixer.q_proj.weight.cpu().float().numpy()
            layer_dict[f"K_{ℓ}"] = mixer.k_proj.weight.cpu().float().numpy()
            layer_dict[f"V_{ℓ}"] = mixer.v_proj.weight.cpu().float().numpy()
        else:
            raise ValueError(f"Layer {ℓ} mixer missing QKV weights. Available: {dir(mixer)}")
        
        # Output projection
        if hasattr(mixer, 'out_proj'):
            layer_dict[f"O_{ℓ}"] = mixer.out_proj.weight.cpu().float().numpy()
        else:
            print(f"Warning: Layer {ℓ} missing out_proj")

        # --- MLP (ϕ-2 uses fc1 / fc2, SwiGLU gate folded into fc1) ------------
        if hasattr(block, 'mlp'):
            mlp = block.mlp
            if hasattr(mlp, 'fc1') and hasattr(mlp, 'fc2'):
                layer_dict[f"FF1_{ℓ}"] = mlp.fc1.weight.cpu().float().numpy()
                layer_dict[f"FF2_{ℓ}"] = mlp.fc2.weight.cpu().float().numpy()
            else:
                print(f"Warning: Layer {ℓ} MLP missing fc1/fc2. Available: {dir(mlp)}")
        
        print(f"✓ Extracted layer {ℓ} weights")
        
    except Exception as e:
        print(f"Error extracting layer {ℓ}: {e}")
        print(f"Layer {ℓ} attributes: {dir(block)}")
        if hasattr(block, 'mixer'):
            print(f"Mixer attributes: {dir(block.mixer)}")
        raise
```

*(The fused `Wqkv` split is the only real structural tweak; see the source lines with `Wqkv` and `out_proj` for confirmation.)* ([Hugging Face][1])

---

## 3  Persist to HDF5 (identical)

```python
# Save with metadata and validation
path = pathlib.Path("phi2_weights.h5")
with h5py.File(path, "w") as f:
    # Store model metadata
    f.attrs["model_id"] = MODEL_ID
    f.attrs["hidden_size"] = d
    f.attrs["num_layers"] = L
    f.attrs["num_heads"] = H
    f.attrs["extraction_dtype"] = "float32"
    
    for name, arr in layer_dict.items():
        if arr is not None:
            f.create_dataset(name, data=arr, compression="gzip")
            print(f"Saved {name}: {arr.shape}")
        else:
            print(f"Skipped {name}: None")
    
    print(f"Total datasets: {len(list(f.keys()))}")
    print(f"File size: {path.stat().st_size / 1024**2:.1f} MB")
```

---

## 4  Prepare point clouds (unchanged)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import random, umap

mat = layer_dict["Q_0"]
rows = normalize(mat, axis=1)
rows = rows[random.sample(range(len(rows)), 10_000)]
rows = PCA(n_components=30, svd_solver="randomized").fit_transform(rows)
```

---

## 5 – 7  TDA, visual dashboards, environment file

Identical to the TinyLlama recipe.

---

### Footnotes & practical tips

1. **Why fused QKV doesn’t break the theory**
   Each row of `Wqkv` still lives in exactly one of your strata (\$\mathcal Z\_Q,\mathcal Z\_K,\mathcal Z\_V\$) after you split, so Axiom 3 and the attention fibration are intact.

2. **Patch-/conv-style embedding**
   ϕ-2 prefixes the standard token embedding with a tiny 1-D convolution; treat that conv weight as part of the embedding stratum (or ignore if you only study transformer blocks).

3. **RMSNorm vs LayerNorm**
   Normalisation weights live in `block.ln.weight` and—because RMSNorm is linear in weights—do not affect row-normalised distance geometry. You can include or skip them; skipping keeps point clouds small.

4. **Safety-research suitability**
   Because ϕ-2 is *pre-RLHF* and Apache-2.0, your topological probes won’t be conflated with alignment gradients—ideal for bias & controllability studies.

---

#### Bottom line

Swap the **model ID** and **split `Wqkv`**, keep everything else. You now have a reproducible, safety-focused “raw” model whose weight geometry can be explored with exactly the same TDA toolkit.

[1]: https://huggingface.co/microsoft/phi-2/commit/cb2f4533604d8b67de604e7df03bfe6f3ca22869 "Upload 8 files · microsoft/phi-2 at cb2f453"

