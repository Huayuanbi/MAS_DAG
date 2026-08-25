from __future__ import annotations

import torch


def _has_path(adjacency: torch.Tensor, source: int, target: int) -> bool:
    stack = [source]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(torch.nonzero(adjacency[node], as_tuple=False).flatten().tolist())
    return False


def decode_greedy_dag(
    edge_prob: torch.Tensor,
    node_mask: torch.Tensor,
    threshold: float | None = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """AGP-style row-major edge selection with incremental cycle rejection."""
    n = edge_prob.shape[0]
    adjacency = torch.zeros((n, n), dtype=torch.bool, device=edge_prob.device)
    for source in range(n):
        for target in range(n):
            if source == target or node_mask[source] == 0 or node_mask[target] == 0:
                continue
            selected = (
                bool(edge_prob[source, target] > threshold)
                if threshold is not None
                else bool(torch.rand((), generator=generator) < edge_prob[source, target].cpu())
            )
            if selected and not _has_path(adjacency, target, source):
                adjacency[source, target] = True
    return adjacency


def is_dag(adjacency: torch.Tensor) -> bool:
    adjacency = adjacency.bool()
    in_degree = adjacency.sum(dim=0).to(torch.long)
    queue = torch.nonzero(in_degree == 0, as_tuple=False).flatten().tolist()
    visited = 0
    while queue:
        source = queue.pop(0)
        visited += 1
        for target in torch.nonzero(adjacency[source], as_tuple=False).flatten().tolist():
            in_degree[target] -= 1
            if in_degree[target] == 0:
                queue.append(target)
    return visited == adjacency.shape[0]
