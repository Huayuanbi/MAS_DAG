from __future__ import annotations

import random
from typing import Any, Iterable, Sequence

from .topology_sampling import SampledTopology, generate_random_dag, validate_topology


def make_topology(
    name: str,
    *,
    num_nodes: int,
    finalizer: int,
    active: Iterable[int],
    edges: Iterable[tuple[int, int]],
    order: Sequence[int] | None = None,
) -> SampledTopology:
    active_tuple = tuple(active)
    active_set = set(active_tuple)
    resolved_order = tuple(
        node for node in (order or tuple(range(num_nodes))) if node in active_set
    )
    adjacency = [[0] * num_nodes for _ in range(num_nodes)]
    for source, target in edges:
        adjacency[source][target] = 1
    topology = SampledTopology(
        generator=name,
        mask=tuple(0 if node in active_set else 1 for node in range(num_nodes)),
        adjacency=tuple(tuple(row) for row in adjacency),
        topological_order=resolved_order,
    )
    validate_topology(topology, finalizer)
    return topology


def mmlu_manual_topologies() -> list[SampledTopology]:
    return [
        make_topology(
            "finalizer_only", num_nodes=6, finalizer=5, active=(5,), edges=(), order=(5,)
        ),
        make_topology(
            "reasoner_finalize", num_nodes=6, finalizer=5,
            active=(2, 5), edges=((2, 5),),
        ),
        make_topology(
            "analyze_reason_finalize", num_nodes=6, finalizer=5,
            active=(0, 2, 5), edges=((0, 2), (2, 5)),
        ),
        make_topology(
            "star", num_nodes=6, finalizer=5, active=range(6),
            edges=((0, 5), (1, 5), (2, 5), (3, 5), (4, 5)),
        ),
        make_topology(
            "parallel_reasoners_finalize", num_nodes=6, finalizer=5,
            active=(2, 3, 5), edges=((2, 5), (3, 5)),
        ),
        make_topology(
            "parallel_reasoners_critic", num_nodes=6, finalizer=5,
            active=(2, 3, 4, 5), edges=((2, 4), (3, 4), (4, 5)),
        ),
        make_topology(
            "full_reasoning_workflow", num_nodes=6, finalizer=5,
            active=range(6),
            edges=((0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)),
        ),
    ]


def pool_constraints(pool: dict[str, Any]) -> tuple[list[list[int]], dict[int, tuple[int, ...]]]:
    nodes = pool["nodes"]
    ids = {str(node["id"]): index for index, node in enumerate(nodes)}
    allowed = pool.get("allowed_edge")
    if not isinstance(allowed, list):
        raise ValueError(f"node pool {pool.get('id')!r} has no allowed_edge matrix")
    requirements = {
        ids[str(target)]: tuple(ids[str(source)] for source in sources)
        for target, sources in pool.get("required_predecessors", {}).items()
    }
    return allowed, requirements


def semantic_random_topologies(
    pool: dict[str, Any],
    *,
    seed: int,
    count: int = 5,
    excluded_signatures: set[tuple] | None = None,
) -> list[SampledTopology]:
    nodes = pool["nodes"]
    finalizer = next(
        index for index, node in enumerate(nodes)
        if str(node["id"]) == str(pool["finalizer_id"])
    )
    allowed, requirements = pool_constraints(pool)
    order = tuple(index for index in range(len(nodes)) if index != finalizer) + (finalizer,)
    weights = [0.0] * len(nodes)
    for active_count, weight in ((2, 0.15), (3, 0.25), (4, 0.25), (5, 0.20), (6, 0.10), (7, 0.05)):
        if active_count <= len(nodes):
            weights[active_count - 1] = weight
    rng = random.Random(seed)
    signatures = set(excluded_signatures or ())
    result = []
    for _ in range(count):
        for _attempt in range(500):
            topology = generate_random_dag(
                len(nodes), finalizer,
                active_count_weights=weights,
                rng=rng,
                fixed_order=order,
                allowed_adjacency=allowed,
                required_predecessors=requirements,
            )
            if topology.signature not in signatures:
                signatures.add(topology.signature)
                result.append(topology)
                break
        else:
            raise RuntimeError("could not generate a unique semantic random topology")
    return result


def validate_random_against_pool(topology: SampledTopology, pool: dict[str, Any]) -> None:
    nodes = pool["nodes"]
    finalizer = next(
        index for index, node in enumerate(nodes)
        if str(node["id"]) == str(pool["finalizer_id"])
    )
    validate_topology(topology, finalizer)
    allowed, requirements = pool_constraints(pool)
    for source, row in enumerate(topology.adjacency):
        for target, edge in enumerate(row):
            if edge and not allowed[source][target]:
                raise ValueError(f"forbidden random edge {source}->{target}")
    active = set(topology.active_nodes)
    for target, sources in requirements.items():
        if target in active and not any(
            source in active and topology.adjacency[source][target]
            for source in sources
        ):
            raise ValueError(f"node {target} lacks a required predecessor edge")
