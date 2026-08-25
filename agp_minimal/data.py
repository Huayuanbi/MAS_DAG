from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class NodeSpec:
    id: str | int
    role: str
    role_brief: str


@dataclass(frozen=True)
class TopologyExample:
    task: str
    nodes: tuple[NodeSpec, ...]
    prune_mask: torch.Tensor
    edge_weight: torch.Tensor
    edge_token_cost: torch.Tensor
    edge_time_cost: torch.Tensor
    reward: float | None
    pair_group: int | None = None

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)


@dataclass(frozen=True)
class RewardPair:
    """Two candidate graphs for the same task, ordered by scalar reward."""

    preferred: TopologyExample
    rejected: TopologyExample

    @property
    def reward_gap(self) -> float:
        assert self.preferred.reward is not None
        assert self.rejected.reward is not None
        return self.preferred.reward - self.rejected.reward


class AGPJsonDataset(Dataset[TopologyExample]):
    def __init__(self, path: str | Path, max_records: int | None = None) -> None:
        self.path = Path(path)
        self._node_pool_cache: dict[Path, list[dict]] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if max_records is not None:
            records = records[:max_records]
        expanded_records = [
            candidate
            for group_index, record in enumerate(records)
            for candidate in self._expand_record(record, group_index)
        ]
        self.examples = [self._parse(record) for record in expanded_records]

    def _load_node_pool(self, reference: str) -> list[dict]:
        """Load a node pool referenced relative to the dataset JSON file."""
        pool_path = Path(reference)
        if not pool_path.is_absolute():
            pool_path = self.path.parent / pool_path
        pool_path = pool_path.resolve()

        if pool_path not in self._node_pool_cache:
            try:
                with pool_path.open("r", encoding="utf-8") as handle:
                    pool = json.load(handle)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"node_pool file not found: {reference!r} "
                    f"(resolved to {pool_path})"
                ) from exc
            if not isinstance(pool, dict) or not isinstance(pool.get("nodes"), list):
                raise ValueError(
                    f"node_pool {reference!r} must be an object containing a nodes array"
                )
            if not pool["nodes"]:
                raise ValueError(f"node_pool {reference!r} must not be empty")
            self._node_pool_cache[pool_path] = pool["nodes"]
        return self._node_pool_cache[pool_path]

    def _expand_record(self, record: dict, group_index: int = 0) -> list[dict]:
        """Flatten a grouped task's candidate graphs; accept legacy flat records."""
        if "graphs" not in record:
            return [record]

        graphs = record["graphs"]
        if not isinstance(graphs, list) or not graphs:
            raise ValueError("graphs must be a non-empty list")
        if "task" not in record:
            raise ValueError("grouped records require a top-level task")
        if "nodes" in record and "node_pool" in record:
            raise ValueError("use either nodes or node_pool, not both")
        if "nodes" in record:
            nodes = record["nodes"]
        elif isinstance(record.get("node_pool"), str) and record["node_pool"]:
            nodes = self._load_node_pool(record["node_pool"])
        else:
            raise ValueError("grouped records require top-level nodes or node_pool")

        expanded = []
        for index, graph in enumerate(graphs):
            if not isinstance(graph, dict):
                raise ValueError(f"graphs[{index}] must be an object")
            if any(
                key in graph for key in ("task", "nodes", "node_pool", "graphs")
            ):
                raise ValueError(
                    f"graphs[{index}] must not override task, nodes, node_pool, or graphs"
                )
            expanded.append(
                {
                    "task": record["task"],
                    "nodes": nodes,
                    **graph,
                    "_pair_group": group_index,
                }
            )
        return expanded

    @staticmethod
    def _parse(record: dict) -> TopologyExample:
        prune_mask = torch.tensor(record["mask"], dtype=torch.float32)
        n = prune_mask.numel()

        def parse_square_matrix(
            name: str, default: torch.Tensor | None = None
        ) -> torch.Tensor:
            raw_matrix = record.get(name)
            if raw_matrix is None:
                if default is None:
                    raise ValueError(f"missing required matrix: {name}")
                matrix = default.clone()
            else:
                matrix = torch.as_tensor(raw_matrix, dtype=torch.float32)
            if matrix.shape != (n, n):
                raise ValueError(
                    f"{name} must have shape ({n}, {n}), got {tuple(matrix.shape)}"
                )
            if not torch.isfinite(matrix).all():
                raise ValueError(f"{name} must contain only finite floating-point values")
            return matrix

        edge_weight = parse_square_matrix("edge_weight")
        edge_token_cost = parse_square_matrix(
            "edge_token_cost", torch.zeros_like(edge_weight)
        )
        edge_time_cost = parse_square_matrix(
            "edge_time_cost", torch.zeros_like(edge_weight)
        )

        raw_reward = record.get("reward")
        if raw_reward is None:
            reward = None
        elif isinstance(raw_reward, bool) or not isinstance(raw_reward, (int, float)):
            raise ValueError("reward must be a finite number or null")
        else:
            reward = float(raw_reward)
            if not math.isfinite(reward):
                raise ValueError("reward must be a finite number or null")

        raw_nodes = record.get("nodes")
        if raw_nodes is None:
            nodes = tuple(
                NodeSpec(
                    id=i,
                    role=f"node_{i}",
                    role_brief=f"Generic agent at position {i}.",
                )
                for i in range(n)
            )
        else:
            if len(raw_nodes) != n:
                raise ValueError(f"nodes must contain {n} entries, got {len(raw_nodes)}")
            parsed_nodes = []
            ids: set[str] = set()
            for index, raw_node in enumerate(raw_nodes):
                node_id = raw_node.get("id", index)
                normalized_id = str(node_id)
                if normalized_id in ids:
                    raise ValueError(f"node ids must be unique, duplicate: {node_id!r}")
                ids.add(normalized_id)
                role = str(raw_node.get("role", f"node_{index}"))
                role_brief = str(raw_node.get("role_brief", role)).strip()
                if not role_brief:
                    raise ValueError(f"nodes[{index}].role_brief must not be empty")
                parsed_nodes.append(
                    NodeSpec(
                        id=node_id,
                        role=role,
                        role_brief=role_brief,
                    )
                )
            nodes = tuple(parsed_nodes)

        return TopologyExample(
            task=record["task"],
            nodes=nodes,
            prune_mask=prune_mask,
            edge_weight=edge_weight,
            edge_token_cost=edge_token_cost,
            edge_time_cost=edge_time_cost,
            reward=reward,
            pair_group=record.get("_pair_group"),
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TopologyExample:
        return self.examples[index]


class PairwiseRewardDataset(Dataset[RewardPair]):
    """Create preferred/rejected graph pairs within each task and node set."""

    def __init__(
        self,
        dataset: AGPJsonDataset,
        min_reward_gap: float = 0.0,
    ) -> None:
        if min_reward_gap < 0:
            raise ValueError("min_reward_gap must be non-negative")

        groups: dict[object, list[TopologyExample]] = {}
        for example in dataset.examples:
            if example.reward is not None:
                key: object = (
                    ("grouped", example.pair_group)
                    if example.pair_group is not None
                    else ("legacy", example.task, example.nodes)
                )
                groups.setdefault(key, []).append(example)

        pairs: list[RewardPair] = []
        for candidates in groups.values():
            for left, right in itertools.combinations(candidates, 2):
                assert left.reward is not None and right.reward is not None
                gap = abs(left.reward - right.reward)
                if gap <= min_reward_gap:
                    continue
                preferred, rejected = (
                    (left, right) if left.reward > right.reward else (right, left)
                )
                pairs.append(RewardPair(preferred=preferred, rejected=rejected))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> RewardPair:
        return self.pairs[index]
