# AGP Minimal

A minimal extraction of AGP's Stage-II topology learner with both the original
GCN baseline and a query-conditioned Graph Transformer encoder.

## What is included

```text
query embedding + role-brief embedding (768)
        -> projection (256)
        -> 3 x TransformerConv (4 heads x 64)
        -> node keep probabilities
        -> directional pair MLP(h_source, h_target)
        -> directed edge probabilities
        -> node/edge supervision
        -> greedy DAG decoder
```

The JSON labels are read directly from the sibling AGP repository. In those
files, `mask[i] == 1` means node `i` was pruned. `edge_weight` is binarized in
the same way as the original Stage-II loss (`nonzero == edge`). The extended
format also carries node metadata plus per-edge token and wall-time costs.

## Dataset format

The file is a JSON array (not JSONL). Each record has this form:

For the complete Chinese data-generation contract and an annotated template,
see [`data/README.md`](data/README.md).

```json
{
  "task": "query text",
  "nodes": [
    {"id": "agent_0", "role": "planner", "role_brief": "Plans and decomposes the task."},
    {"id": "agent_1", "role": "critic", "role_brief": "Checks reasoning and finds errors."}
  ],
  "mask": [0, 0],
  "edge_weight": [[0.0, 1.0], [0.0, 0.0]],
  "edge_token_cost": [[0.0, 128.5], [0.0, 0.0]],
  "edge_time_cost": [[0.0, 0.42], [0.0, 0.0]]
}
```

All four graph fields must agree on `N`: `mask` has length `N`, and each edge
matrix has shape `[N, N]`. Matrix values are parsed as `float32`. Node IDs must
be unique. Node features are generated from `role_brief` at training time and
are not stored in the JSON.

Legacy AGP records remain valid. Missing `nodes` are synthesized as
`node_0 ... node_N`; missing cost matrices are filled with zeros. Costs are
loaded and validated but are not yet included in the optimization objective.

By default, `sentence-transformers/all-MiniLM-L6-v2` generates 384-dimensional
embeddings from the query text and each node's `role_brief`. The encoder is
loaded once and repeated role briefs are cached. `--offline-features` switches
to deterministic hash embeddings for tests without model access.
The Graph Transformer uses all directed node pairs except self-loops as its
attention graph. Thus every node can attend to every other candidate in one
layer. Its computation and memory are quadratic in the node count. The GCN
baseline remains available and uses a bidirectional chain placeholder because
the exported JSON does not retain AGP's per-example role graph.

## Run

```bash
conda activate agp
cd /home/yz/Documents/MAS_DAG/AGP-minimal

python debug.py
python train.py --epochs 1 --max-records 16
python -m unittest tests.test_smoke
```

Graph Transformer is the default. Select the old baseline with `--model gcn`.

Offline smoke run:

```bash
python debug.py --offline-features
python train.py --offline-features --epochs 1 --max-records 16
```

Debugger stages:

```bash
python debug.py --break-at model
python debug.py --break-at loss
python debug.py --break-at dag
```

By default the optimizer updates both GCN and MLP parameters. Use
`--gcn-only` to reproduce the original AGP optimizer behavior, which omits the
MLP parameters.

## Main replacement points

- `agp_minimal/features.py`: text encoder and feature cache.
- `agp_minimal/model.py`: GCN baseline, Graph Transformer, and edge heads.
- `agp_minimal/decoder.py`: replace order-dependent greedy DAG construction.
- `agp_minimal/losses.py`: add order, reachability, and connectivity losses.
