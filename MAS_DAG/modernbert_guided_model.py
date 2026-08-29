"""ModernBERT-guided graph topology model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from modernbert_transformers import ModernBertGraphTransformerBlock

from .model import TopologyOutput


class ModernBertGuidedGraphTopologyModel(torch.nn.Module):
    """Frozen ModernBERT guidance with a graph-native trainable main path.

    Every graph node is one independent BERT batch item. Selected BERT layer
    outputs are pooled over their token dimension. The first selected layer
    starts the graph path, while later selected layers enter through
    zero-initialized bridges before their corresponding graph blocks.

    This initial implementation handles one graph per call:

    * ``input_ids`` and ``attention_mask``: ``[num_nodes, seq_len]``
    * ``edge_index``: ``[2, num_edges]`` (source row, target row)
    * returned directed edge probabilities: ``[num_nodes, num_nodes]``
    """

    def __init__(
        self,
        bert: torch.nn.Module,
        guidance_layers: tuple[int, ...] = (6, 12, 21),
        graph_pe_dim: int = 16,
        mask_hidden_dim: int = 128,
        edge_hidden_dim: int = 128,
        dropout: float = 0.1,
        node_threshold: float = 0.55,
        freeze_bert: bool = True,
        initialize_graph_from_bert: bool = True,
    ) -> None:
        super().__init__()
        if not guidance_layers:
            raise ValueError("guidance_layers must contain at least one layer")
        if tuple(sorted(set(guidance_layers))) != guidance_layers:
            raise ValueError("guidance_layers must be strictly increasing")

        self.bert = bert
        self.guidance_layers = guidance_layers
        self.graph_pe_dim = graph_pe_dim
        self.node_threshold = node_threshold
        self.bert_is_frozen = freeze_bert

        config = bert.config
        num_bert_layers = int(config.num_hidden_layers)
        if guidance_layers[0] < 0 or guidance_layers[-1] >= num_bert_layers:
            raise ValueError(
                f"guidance layers must be in [0, {num_bert_layers}), got {guidance_layers}"
            )
        hidden_size = int(config.hidden_size)
        self.hidden_size = hidden_size

        self.graph_blocks = torch.nn.ModuleList(
            ModernBertGraphTransformerBlock(
                config,
                layer_idx=layer_idx,
                graph_pe_dim=graph_pe_dim,
            )
            for layer_idx in guidance_layers
        )
        self.guidance_bridges = torch.nn.ModuleList(
            torch.nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in guidance_layers[1:]
        )
        for bridge in self.guidance_bridges:
            torch.nn.init.zeros_(bridge.weight)

        if initialize_graph_from_bert:
            bert_layers = self._resolve_bert_layers(bert)
            for source_layer_idx, graph_block in zip(guidance_layers, self.graph_blocks):
                graph_block.load_modernbert_layer_state_dict(
                    bert_layers[source_layer_idx].state_dict()
                )

        self.mask_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, mask_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(mask_hidden_dim, 1),
        )
        self.edge_head = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_size, edge_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(edge_hidden_dim, 1),
        )
        self.set_bert_trainable(not freeze_bert)

    @staticmethod
    def _resolve_bert_layers(bert: torch.nn.Module) -> torch.nn.ModuleList:
        """Accept either ModernBertModel or a ModernBERT task wrapper."""
        if hasattr(bert, "layers"):
            return bert.layers
        if hasattr(bert, "model") and hasattr(bert.model, "layers"):
            return bert.model.layers
        raise TypeError("bert must expose `.layers` or `.model.layers`")

    def set_bert_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze the semantic branch without changing graph modules."""
        self.bert_is_frozen = not trainable
        self.bert.requires_grad_(trainable)
        if self.bert_is_frozen:
            self.bert.eval()

    def train(self, mode: bool = True) -> "ModernBertGuidedGraphTopologyModel":
        super().train(mode)
        if self.bert_is_frozen:
            self.bert.eval()
        return self

    @staticmethod
    def _masked_mean_pool(
        token_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce ``[N, L, D]`` token states to ``[N, D]`` node states."""
        mask = attention_mask.unsqueeze(-1).to(token_hidden.dtype)
        return (token_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def _encode_guidance(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        context = torch.no_grad() if self.bert_is_frozen else torch.enable_grad()
        with context:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        if outputs.hidden_states is None:
            raise RuntimeError("ModernBERT did not return hidden states")

        # hidden_states[0] is the embedding output; encoder layer k is k + 1.
        return [
            self._masked_mean_pool(outputs.hidden_states[layer_idx + 1], attention_mask)
            for layer_idx in self.guidance_layers
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        edge_index: torch.Tensor,
        graph_pe: torch.Tensor | None = None,
    ) -> TopologyOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [num_nodes, seq_len]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        num_nodes = input_ids.shape[0]

        guidance = self._encode_guidance(input_ids, attention_mask)
        if graph_pe is None:
            # A complete candidate graph has no meaningful Laplacian positions.
            # Callers with a meaningful prior graph can pass a real GraphPE.
            graph_pe = guidance[0].new_zeros((num_nodes, self.graph_pe_dim))
        elif graph_pe.shape != (num_nodes, self.graph_pe_dim):
            raise ValueError(
                f"graph_pe must have shape {(num_nodes, self.graph_pe_dim)}, "
                f"got {tuple(graph_pe.shape)}"
            )

        # The first BERT tap is the non-zero semantic input to the graph path.
        encoded = self.graph_blocks[0](guidance[0], edge_index, graph_pe)
        for node_guidance, bridge, graph_block in zip(
            guidance[1:], self.guidance_bridges, self.graph_blocks[1:]
        ):
            encoded = graph_block(
                encoded + bridge(node_guidance),
                edge_index,
                graph_pe,
            )

        node_log_features = F.log_softmax(encoded, dim=-1)
        node_prob = torch.sigmoid(self.mask_head(encoded).squeeze(-1))
        hard_mask = (node_prob > self.node_threshold).to(node_prob.dtype)
        node_mask = node_prob + (hard_mask - node_prob).detach()

        source = encoded[:, None, :].expand(num_nodes, num_nodes, -1)
        target = encoded[None, :, :].expand(num_nodes, num_nodes, -1)
        edge_pairs = torch.cat((source, target), dim=-1)
        edge_logits = self.edge_head(edge_pairs).squeeze(-1)
        edge_prob = torch.sigmoid(edge_logits)
        off_diagonal = 1.0 - torch.eye(
            num_nodes, dtype=edge_prob.dtype, device=edge_prob.device
        )
        edge_prob = edge_prob * off_diagonal
        return TopologyOutput(
            node_log_features=node_log_features,
            node_prob=node_prob,
            node_mask=node_mask,
            edge_embeddings=encoded,
            edge_logits=edge_logits,
            edge_prob=edge_prob,
        )


__all__ = ["ModernBertGuidedGraphTopologyModel"]
