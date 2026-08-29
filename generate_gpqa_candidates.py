from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from MAS_DAG import generate_candidate_suite
from MAS_DAG.topology_sampling import SampledTopology, validate_topology


CORE_NODES = (0, 1, 5, 6, 7, 10)


def make_topology(name: str, active: tuple[int, ...], edges: tuple[tuple[int, int], ...]) -> SampledTopology:
    num_nodes = 11
    finalizer = 10
    adjacency = [[0] * num_nodes for _ in range(num_nodes)]
    for source, target in edges:
        adjacency[source][target] = 1
    order = tuple(node for node in range(num_nodes) if node in active and node != finalizer) + (finalizer,)
    topology = SampledTopology(
        generator=name,
        mask=tuple(0 if node in active else 1 for node in range(num_nodes)),
        adjacency=tuple(tuple(row) for row in adjacency),
        topological_order=order,
    )
    validate_topology(topology, finalizer)
    return topology


def workflow_topologies() -> list[SampledTopology]:
    return [
        make_topology(
            "domain_experts_reason_critic",
            (0, 2, 3, 4, 5, 7, 10),
            ((0, 2), (0, 3), (0, 4), (2, 5), (3, 5), (4, 5), (5, 7), (7, 10)),
        ),
        make_topology(
            "parallel_solvers_verify",
            (5, 6, 9, 10),
            ((5, 9), (6, 9), (9, 10)),
        ),
        make_topology(
            "solve_critique_revise",
            (5, 7, 8, 10),
            ((5, 7), (5, 8), (7, 8), (8, 10)),
        ),
        make_topology(
            "parallel_domain_experts_judge",
            (2, 3, 4, 9, 10),
            ((2, 9), (3, 9), (4, 9), (9, 10)),
        ),
    ]


def expand_core_topology(topology: SampledTopology) -> SampledTopology:
    """Embed an original six-node candidate graph in the 11-role GPQA pool."""
    mapping = dict(enumerate(CORE_NODES))
    adjacency = [[0] * 11 for _ in range(11)]
    for source, row in enumerate(topology.adjacency):
        for target, edge in enumerate(row):
            if edge:
                adjacency[mapping[source]][mapping[target]] = 1
    expanded = SampledTopology(
        generator=topology.generator,
        mask=tuple(0 if node in {mapping[i] for i in topology.active_nodes} else 1 for node in range(11)),
        adjacency=tuple(tuple(row) for row in adjacency),
        topological_order=tuple(mapping[node] for node in topology.topological_order),
    )
    validate_topology(expanded, 10)
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 12 sampled plus four workflow GPQA graphs.")
    parser.add_argument("--input", type=Path, default=Path("data/gpqa/sample.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/gpqa/candidate_graphs.json"))
    parser.add_argument("--node-pool", type=Path, default=Path("data/node_pools/gpqa_diamond_11_roles.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-count", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool = json.loads(args.node_pool.read_text(encoding="utf-8"))
    if len(pool["nodes"]) != 11 or pool["finalizer_id"] != "agent_10":
        raise ValueError("GPQA generator expects the gpqa_diamond_11_roles node pool")
    node_pool_ref = os.path.relpath(args.node_pool.resolve(), args.output.resolve().parent)
    workflows = workflow_topologies()
    records = []
    with args.input.open(encoding="utf-8") as handle:
        sources = [json.loads(line) for line in handle if line.strip()]
    if args.limit is not None:
        sources = sources[: args.limit]
    for query_index, source in enumerate(sources):
        sampled_core = generate_candidate_suite(
            6,
            5,
            random_count=args.random_count,
            seed=args.seed + query_index,
            fixed_order=tuple(range(6)),
        )
        sampled = [expand_core_topology(topology) for topology in sampled_core]
        topologies = list(sampled) + workflows
        graphs = []
        for graph_index, topology in enumerate(topologies):
            graph = topology.to_graph_record()
            graph["id"] = f"q{query_index:04d}_g{graph_index:02d}"
            graph["graph_group"] = "sampled" if graph_index < len(sampled) else "workflow_addition"
            graphs.append(graph)
        records.append({
            "task": source["question"],
            "reference_answer": source["answer"],
            "reference_solution": "",
            "source_metadata": {key: value for key, value in source.items() if key not in ("question", "answer")},
            "sampling_seed": args.seed + query_index,
            "node_pool": node_pool_ref,
            "evaluator": "multiple_choice",
            "topology_policy": {
                "sampled_graphs": len(sampled),
                "workflow_additions": len(workflows),
                "core_nodes_for_sampled_graphs": list(CORE_NODES),
            },
            "graphs": graphs,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"queries={len(records)} graphs={sum(len(x['graphs']) for x in records)} graphs_per_query={len(records[0]['graphs']) if records else 0} output={args.output}")


if __name__ == "__main__":
    main()
