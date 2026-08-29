"""Graph Transformer block initialized from a ModernBERT encoder layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


def normalized_laplacian_graph_pe(
    edge_index: torch.Tensor,
    num_nodes: int,
    pe_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compute normalized-Laplacian GraphPE with deterministic signs.

    Directed edges are symmetrized for the eigendecomposition. The constant
    first eigenvector is omitted and unavailable dimensions are zero padded.
    For repeatedly used graphs, compute this once and pass it as ``graph_pe``.
    """
    if pe_dim <= 0:
        raise ValueError("pe_dim must be positive")
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if num_nodes == 0:
        return torch.empty((0, pe_dim), dtype=dtype, device=device)

    edge_index = edge_index.to(device=device, dtype=torch.long)
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if edge_index.numel():
        source, target = edge_index
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    adjacency.fill_diagonal_(0.0)

    degree = adjacency.sum(dim=-1)
    inv_sqrt_degree = degree.clamp_min(1.0).rsqrt()
    laplacian = torch.eye(num_nodes, dtype=torch.float32, device=device)
    laplacian -= inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    isolated = degree == 0
    laplacian[isolated, isolated] = 0.0

    with torch.no_grad():
        _, eigenvectors = torch.linalg.eigh(laplacian)
        usable_dim = min(pe_dim, max(num_nodes - 1, 0))
        graph_pe = torch.zeros((num_nodes, pe_dim), dtype=torch.float32, device=device)
        if usable_dim:
            selected = eigenvectors[:, 1 : usable_dim + 1]
            max_rows = selected.abs().argmax(dim=0)
            columns = torch.arange(usable_dim, device=device)
            signs = selected[max_rows, columns].sign()
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            graph_pe[:, :usable_dim] = selected * signs
    return graph_pe.to(dtype=dtype)


def _segment_softmax(
    scores: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Softmax edge scores over incoming edges for every target and head."""
    num_heads = scores.shape[-1]
    expanded_target = target[:, None].expand(-1, num_heads)
    maximum = scores.new_full((num_nodes, num_heads), -torch.inf)
    maximum.scatter_reduce_(0, expanded_target, scores, reduce="amax", include_self=True)
    numerator = (scores - maximum[target]).exp()
    denominator = scores.new_zeros((num_nodes, num_heads))
    denominator.scatter_add_(0, expanded_target, numerator)
    return numerator / denominator[target].clamp_min(torch.finfo(scores.dtype).tiny)


def _edges_with_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Validate edges, add self-loops, and remove duplicate directed edges."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    edge_index = edge_index.to(dtype=torch.long)
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes):
        raise ValueError("edge_index contains a node outside [0, num_nodes)")

    nodes = torch.arange(num_nodes, device=edge_index.device)
    self_loop_ids = nodes * num_nodes + nodes
    edge_ids = edge_index[0] * num_nodes + edge_index[1]
    edge_ids = torch.unique(torch.cat((edge_ids, self_loop_ids)), sorted=True)
    return torch.stack(
        (torch.div(edge_ids, num_nodes, rounding_mode="floor"), edge_ids.remainder(num_nodes)),
        dim=0,
    )


class ModernBertGraphAttention(nn.Module):
    """ModernBERT MHA weights with sparse, edge-restricted graph attention."""

    def __init__(self, config: Any, graph_pe_dim: int):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = float(config.attention_dropout)

        # Names and shapes match ModernBertAttention exactly.
        self.Wqkv = nn.Linear(
            self.hidden_size,
            3 * self.hidden_size,
            bias=bool(config.attention_bias),
        )
        self.Wo = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=bool(config.attention_bias),
        )
        self.out_drop = nn.Dropout(self.attention_dropout)

        # The only attention parameter absent from a ModernBERT checkpoint.
        self.graph_pe_projection = nn.Linear(graph_pe_dim, self.hidden_size, bias=False)
        nn.init.zeros_(self.graph_pe_projection.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        edge_index: torch.Tensor,
        graph_pe: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [num_nodes, hidden_size]")
        num_nodes = hidden_states.shape[0]
        expected_pe_shape = (num_nodes, self.graph_pe_projection.in_features)
        if graph_pe.shape != expected_pe_shape:
            raise ValueError(f"graph_pe must have shape {expected_pe_shape}")

        edge_index = _edges_with_self_loops(edge_index.to(hidden_states.device), num_nodes)
        source, target = edge_index
        positioned_states = hidden_states + self.graph_pe_projection(
            graph_pe.to(device=hidden_states.device, dtype=hidden_states.dtype)
        )
        qkv = self.Wqkv(positioned_states).view(
            num_nodes, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=1)

        scores = (query[target].float() * key[source].float()).sum(dim=-1)
        scores = scores * (self.head_dim**-0.5)
        attention = _segment_softmax(scores, target, num_nodes).to(value.dtype)
        attention = F.dropout(attention, p=self.attention_dropout, training=self.training)

        messages = value[source] * attention.unsqueeze(-1)
        attended = value.new_zeros((num_nodes, self.num_heads, self.head_dim))
        attended.index_add_(0, target, messages)
        output = self.out_drop(self.Wo(attended.reshape(num_nodes, self.hidden_size)))
        weights = (edge_index, attention) if return_attention_weights else None
        return output, weights


class ModernBertGraphMLP(nn.Module):
    """ModernBERT gated MLP with checkpoint-compatible parameter names."""

    def __init__(self, config: Any):
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        self.Wi = nn.Linear(
            hidden_size,
            2 * intermediate_size,
            bias=bool(config.mlp_bias),
        )
        self.Wo = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=bool(config.mlp_bias),
        )
        self.dropout = float(config.mlp_dropout)
        self.activation = str(config.hidden_activation)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        value, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        if self.activation == "gelu":
            value = F.gelu(value)
        elif self.activation in {"silu", "swish"}:
            value = F.silu(value)
        elif self.activation == "relu":
            value = F.relu(value)
        else:
            raise ValueError(f"unsupported hidden_activation: {self.activation}")
        return self.Wo(F.dropout(value * gate, p=self.dropout, training=self.training))


class ModernBertGraphTransformerBlock(nn.Module):
    """GraphPE block whose pretrained tensors match one ModernBERT layer.

    ``edge_index[0]`` contains source nodes and ``edge_index[1]`` target nodes.
    Self-loops are added automatically. GraphPE is computed online when it is
    not supplied explicitly.
    """

    _GRAPH_ONLY_KEYS = frozenset({"attn.graph_pe_projection.weight"})

    def __init__(self, config: Any, *, layer_idx: int = 1, graph_pe_dim: int = 16):
        super().__init__()
        if graph_pe_dim <= 0:
            raise ValueError("graph_pe_dim must be positive")
        self.layer_idx = int(layer_idx)
        self.graph_pe_dim = int(graph_pe_dim)
        hidden_size = int(config.hidden_size)
        norm_kwargs = {
            "normalized_shape": hidden_size,
            "eps": float(config.norm_eps),
            "bias": bool(config.norm_bias),
        }

        # ModernBERT layer zero relies on the embedding LayerNorm.
        self.attn_norm = nn.Identity() if self.layer_idx == 0 else nn.LayerNorm(**norm_kwargs)
        self.attn = ModernBertGraphAttention(config, graph_pe_dim=self.graph_pe_dim)
        self.mlp_norm = nn.LayerNorm(**norm_kwargs)
        self.mlp = ModernBertGraphMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        edge_index: torch.Tensor,
        graph_pe: torch.Tensor | None = None,
        *,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if graph_pe is None:
            graph_pe = normalized_laplacian_graph_pe(
                edge_index,
                hidden_states.shape[0],
                self.graph_pe_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        attended, attention_weights = self.attn(
            self.attn_norm(hidden_states),
            edge_index,
            graph_pe,
            return_attention_weights=return_attention_weights,
        )
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        if return_attention_weights:
            assert attention_weights is not None
            return hidden_states, attention_weights
        return hidden_states

    def load_modernbert_layer_state_dict(
        self,
        layer_state_dict: Mapping[str, torch.Tensor],
    ) -> tuple[str, ...]:
        """Strictly load every ModernBERT tensor; leave only GraphPE new."""
        target_state = self.state_dict()
        expected_source_keys = set(target_state) - self._GRAPH_ONLY_KEYS
        source_keys = set(layer_state_dict)
        missing = sorted(expected_source_keys - source_keys)
        unexpected = sorted(source_keys - expected_source_keys)
        if missing or unexpected:
            raise ValueError(
                "ModernBERT layer state_dict is incompatible: "
                f"missing={missing}, unexpected={unexpected}"
            )

        incompatible = self.load_state_dict(dict(layer_state_dict), strict=False)
        if set(incompatible.missing_keys) != self._GRAPH_ONLY_KEYS or incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected load result: {incompatible}")
        return tuple(sorted(incompatible.missing_keys))

    def load_modernbert_layer_from_safetensors(
        self,
        checkpoint: str | Path,
        *,
        source_layer_idx: int | None = None,
        checkpoint_prefix: str = "model.layers",
    ) -> tuple[str, ...]:
        """Load one layer directly without materializing the full checkpoint."""
        source_layer_idx = self.layer_idx if source_layer_idx is None else int(source_layer_idx)
        if (source_layer_idx == 0) != (self.layer_idx == 0):
            raise ValueError(
                "layer 0 has no attn_norm; use layer_idx=0 only with source layer 0"
            )
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError("safetensors is required to load a checkpoint") from exc

        prefix = f"{checkpoint_prefix}.{source_layer_idx}."
        target_keys = set(self.state_dict()) - self._GRAPH_ONLY_KEYS
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            missing = sorted(key for key in target_keys if prefix + key not in available)
            if missing:
                raise ValueError(f"checkpoint is missing layer tensors: {missing}")
            for key in target_keys:
                tensors[key] = handle.get_tensor(prefix + key)
        return self.load_modernbert_layer_state_dict(tensors)


__all__ = [
    "ModernBertGraphAttention",
    "ModernBertGraphMLP",
    "ModernBertGraphTransformerBlock",
    "normalized_laplacian_graph_pe",
]
