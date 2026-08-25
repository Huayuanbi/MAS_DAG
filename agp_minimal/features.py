from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import NodeSpec


DEFAULT_TEXT_MODEL = str(
    Path(__file__).resolve().parents[2] / "models" / "all-MiniLM-L6-v2"
)


def bidirectional_chain_edge_index(
    num_nodes: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Placeholder role graph; the learned edge candidate space remains dense."""
    pairs = []
    for i in range(num_nodes - 1):
        pairs.extend(((i, i + 1), (i + 1, i)))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()


def fully_connected_edge_index(
    num_nodes: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """All directed node pairs except self-loops, for global graph attention."""
    nodes = torch.arange(num_nodes, device=device)
    source = nodes.repeat_interleave(num_nodes)
    target = nodes.repeat(num_nodes)
    keep = source != target
    return torch.stack((source[keep], target[keep]), dim=0)


class DeterministicFeatureBuilder:
    """Offline placeholder for 384-d role-brief and query text encoders."""

    def __init__(self, embedding_dim: int = 384, seed: int = 17) -> None:
        self.embedding_dim = embedding_dim
        self.seed = seed

    def _text_embedding(self, text: str, namespace: str) -> torch.Tensor:
        payload = f"{namespace}\0{text}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        text_seed = int.from_bytes(digest[:8], "little") % (2**31)
        generator = torch.Generator().manual_seed(text_seed + self.seed)
        return F.normalize(torch.randn(self.embedding_dim, generator=generator), dim=0)

    def _role_embeddings(self, nodes: tuple[NodeSpec, ...]) -> torch.Tensor:
        return torch.stack(
            [self._text_embedding(node.role_brief, "role") for node in nodes]
        )

    def __call__(
        self,
        task: str,
        nodes: tuple[NodeSpec, ...],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        role = self._role_embeddings(nodes)
        query = self._text_embedding(task, "query").unsqueeze(0).repeat(len(nodes), 1)
        return torch.cat((role, query), dim=1).to(device)


class SentenceTransformerFeatureBuilder:
    """Generate AGP's 384-d role/query embeddings with one reusable encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        model_device: str | None = None,
        encoder=None,
    ) -> None:
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model_name, device=model_device)
        self.encoder = encoder
        self.model_name = model_name
        self.embedding_dim = int(self.encoder.get_sentence_embedding_dimension())
        if self.embedding_dim != 384:
            raise ValueError(
                f"AGPTopologyModel expects 384-d text embeddings, but "
                f"{model_name!r} produces {self.embedding_dim}"
            )
        self._role_cache: dict[str, torch.Tensor] = {}

    def _encode(self, texts: list[str]) -> torch.Tensor:
        embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return torch.as_tensor(embeddings, dtype=torch.float32).detach().cpu()

    def _role_embeddings(self, nodes: tuple[NodeSpec, ...]) -> torch.Tensor:
        missing = list(
            dict.fromkeys(
                node.role_brief
                for node in nodes
                if node.role_brief not in self._role_cache
            )
        )
        if missing:
            encoded = self._encode(missing)
            self._role_cache.update(zip(missing, encoded))
        return torch.stack([self._role_cache[node.role_brief] for node in nodes])

    def __call__(
        self,
        task: str,
        nodes: tuple[NodeSpec, ...],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        role = self._role_embeddings(nodes)
        query = self._encode([task]).repeat(len(nodes), 1)
        return torch.cat((role, query), dim=1).to(device)
