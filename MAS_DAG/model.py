from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv


def min_max_norm(tensor: torch.Tensor) -> torch.Tensor:
    minimum = tensor.min()
    maximum = tensor.max()
    if torch.isclose(minimum, maximum):
        return tensor
    return 2.0 * (tensor - minimum) / (maximum - minimum) - 1.0


@dataclass
class TopologyOutput:
    node_log_features: torch.Tensor
    node_prob: torch.Tensor
    node_mask: torch.Tensor
    edge_embeddings: torch.Tensor
    edge_logits: torch.Tensor
    edge_prob: torch.Tensor


class AGPTopologyModel(torch.nn.Module):
    """Minimal extraction of AGP/AGP/gnn/gcn.py plus its MLP edge head."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 16,
        output_dim: int = 384,
        mask_hidden_dim: int = 64,
        edge_dim: int = 16,
        dropout: float = 0.0,
        node_threshold: float = 0.55,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.mask_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, mask_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mask_hidden_dim, 1),
        )
        self.edge_head = torch.nn.Sequential(
            torch.nn.Linear(output_dim, edge_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_dim, edge_dim),
        )
        self.dropout = dropout
        self.node_threshold = node_threshold

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> TopologyOutput:
        hidden = F.relu(self.conv1(node_features, edge_index))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        encoded = self.conv2(hidden, edge_index)

        node_log_features = F.log_softmax(encoded, dim=1)
        node_prob = torch.sigmoid(self.mask_head(hidden).squeeze(-1))
        hard_mask = (node_prob > self.node_threshold).to(node_prob.dtype)
        node_mask = node_prob + (hard_mask - node_prob).detach()

        edge_embeddings = self.edge_head(node_log_features)
        edge_logits = min_max_norm(edge_embeddings @ edge_embeddings.t())
        edge_prob = torch.sigmoid(edge_logits)
        off_diagonal = 1.0 - torch.eye(
            edge_prob.shape[0], dtype=edge_prob.dtype, device=edge_prob.device
        )
        edge_prob = edge_prob * off_diagonal
        return TopologyOutput(
            node_log_features=node_log_features,
            node_prob=node_prob,
            node_mask=node_mask,
            edge_embeddings=edge_embeddings,
            edge_logits=edge_logits,
            edge_prob=edge_prob,
        )


class GraphTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        model_dim: int,
        heads: int,
        ff_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError("model_dim must be divisible by heads")
        self.attention = TransformerConv(
            model_dim,
            model_dim // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            root_weight=False,
        )
        self.norm1 = torch.nn.LayerNorm(model_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(model_dim, ff_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ff_hidden_dim, model_dim),
        )
        self.norm2 = torch.nn.LayerNorm(model_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        attended = self.attention(features, edge_index)
        features = self.norm1(features + self.dropout(attended))
        return self.norm2(features + self.dropout(self.feed_forward(features)))


class GraphTransformerTopologyModel(torch.nn.Module):
    """Global graph attention encoder with directional node-pair edge scoring."""

    def __init__(
        self,
        input_dim: int = 768,
        model_dim: int = 256,
        heads: int = 4,
        num_layers: int = 3,
        ff_hidden_dim: int = 512,
        mask_hidden_dim: int = 128,
        edge_hidden_dim: int = 128,
        dropout: float = 0.1,
        node_threshold: float = 0.55,
    ) -> None:
        super().__init__()
        self.input_projection = torch.nn.Sequential(
            torch.nn.Linear(input_dim, model_dim),
            torch.nn.LayerNorm(model_dim),
        )
        self.layers = torch.nn.ModuleList(
            GraphTransformerBlock(model_dim, heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        )
        self.mask_head = torch.nn.Sequential(
            torch.nn.Linear(model_dim, mask_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(mask_hidden_dim, 1),
        )
        self.edge_head = torch.nn.Sequential(
            torch.nn.Linear(2 * model_dim, edge_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(edge_hidden_dim, 1),
        )
        self.node_threshold = node_threshold

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> TopologyOutput:
        encoded = self.input_projection(node_features)
        for layer in self.layers:
            encoded = layer(encoded, edge_index)

        node_log_features = F.log_softmax(encoded, dim=1)
        node_prob = torch.sigmoid(self.mask_head(encoded).squeeze(-1))
        hard_mask = (node_prob > self.node_threshold).to(node_prob.dtype)
        node_mask = node_prob + (hard_mask - node_prob).detach()

        num_nodes = encoded.shape[0]
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
