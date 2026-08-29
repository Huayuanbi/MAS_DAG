# ModernBERT Python implementation

This directory is a local source snapshot of the ModernBERT implementation
shipped with Hugging Face Transformers 5.10.2. `answerdotai/ModernBERT-large`
uses this shared architecture; the large-specific layer counts and dimensions
are supplied by the checkpoint's `config.json`.

- `modeling_modernbert.py`: complete generated PyTorch implementation
- `modular_modernbert.py`: editable modular source used to generate it
- `configuration_modernbert.py`: `ModernBertConfig`
- `__init__.py`: upstream package exports

The files retain their upstream Apache-2.0 license headers and depend on the
rest of the `transformers` package. For normal use:

```python
from transformers import ModernBertModel
```

## Graph Transformer block

`graph_transformer_block.py` keeps the pretrained parameter names and shapes
of one ModernBERT encoder layer, replaces sequence attention with sparse
attention over `edge_index`, and replaces RoPE with normalized-Laplacian
GraphPE. Self-loops are added automatically.

```python
import json
from types import SimpleNamespace

import torch

from modernbert_transformers import ModernBertGraphTransformerBlock

with open("/data1/yz/MAS_DAG/ModernBERT-large/config.json") as file:
    config = SimpleNamespace(**json.load(file))

block = ModernBertGraphTransformerBlock(
    config,
    layer_idx=1,
    graph_pe_dim=16,
)
block.load_modernbert_layer_from_safetensors(
    "/data1/yz/MAS_DAG/ModernBERT-large/model.safetensors",
    source_layer_idx=1,
)

node_features = torch.randn(5, config.hidden_size)
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
node_features = block(node_features, edge_index)
```

The GraphPE projection is the only new parameter. It starts at zero so loading
a pretrained layer does not immediately perturb its hidden-state distribution;
it receives gradients normally and becomes active during fine-tuning. For a
fixed graph, use `normalized_laplacian_graph_pe` once and pass the result via
`graph_pe` to avoid repeating the eigendecomposition.
