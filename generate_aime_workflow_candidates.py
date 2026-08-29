from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from MAS_DAG.topology_sampling import SampledTopology, validate_topology


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "aime" / "sample.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "aime" / "workflow_candidate_graphs.json"
DEFAULT_POOL = ROOT / "data" / "node_pools" / "aime_workflow_13_roles.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate four semantically valid AIME workflow seed graphs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--node-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_topology(
    name: str,
    *,
    num_nodes: int,
    finalizer: int,
    active: Iterable[int],
    edges: Iterable[tuple[int, int]],
    order: Iterable[int],
) -> SampledTopology:
    active_tuple = tuple(active)
    mask = tuple(0 if node in active_tuple else 1 for node in range(num_nodes))
    adjacency = [[0] * num_nodes for _ in range(num_nodes)]
    for source, target in edges:
        adjacency[source][target] = 1
    topology = SampledTopology(
        generator=name,
        mask=mask,
        adjacency=tuple(tuple(row) for row in adjacency),
        topological_order=tuple(order),
    )
    validate_topology(topology, finalizer)
    return topology


def build_workflow_topologies(num_nodes: int = 13, finalizer: int = 12) -> list[SampledTopology]:
    """Four AIME mother graphs with intentional, role-valid information flow."""
    return [
        make_topology(
            "plan_critique_refine_solve",
            num_nodes=num_nodes,
            finalizer=finalizer,
            active=(0, 1, 2, 5, 12),
            edges=((0, 1), (0, 2), (1, 2), (2, 5), (5, 12)),
            order=(0, 1, 2, 5, 12),
        ),
        make_topology(
            "parallel_solvers_verify",
            num_nodes=num_nodes,
            finalizer=finalizer,
            active=(3, 4, 5, 11, 12),
            edges=((3, 11), (4, 11), (5, 11), (11, 12)),
            order=(3, 4, 5, 11, 12),
        ),
        make_topology(
            "solve_critique_revise",
            num_nodes=num_nodes,
            finalizer=finalizer,
            active=(5, 6, 7, 12),
            edges=((5, 6), (5, 7), (6, 7), (7, 12)),
            order=(5, 6, 7, 12),
        ),
        make_topology(
            "parallel_solve_cross_check",
            num_nodes=num_nodes,
            finalizer=finalizer,
            active=(3, 4, 8, 9, 10, 12),
            edges=((3, 8), (4, 9), (8, 10), (9, 10), (10, 12)),
            order=(3, 4, 8, 9, 10, 12),
        ),
    ]


def load_jsonl(path: Path, limit: int | None) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "question" not in record or "answer" not in record:
                raise ValueError(
                    f"{path}:{line_number} must contain question and answer"
                )
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def split_reference(raw_answer: object) -> tuple[str, str]:
    value = str(raw_answer)
    if "####" not in value:
        raise ValueError("sample answer must contain the #### delimiter")
    solution, answer = value.rsplit("####", 1)
    return solution.strip(), answer.strip()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    with args.node_pool.open("r", encoding="utf-8") as handle:
        pool = json.load(handle)
    nodes = pool.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("node pool must contain a nodes list")
    finalizers = [
        index
        for index, node in enumerate(nodes)
        if str(node.get("id")) == str(pool.get("finalizer_id"))
    ]
    if len(finalizers) != 1:
        raise ValueError("node pool finalizer_id must identify exactly one node")
    finalizer = finalizers[0]
    topologies = build_workflow_topologies(len(nodes), finalizer)
    node_pool_reference = os.path.relpath(
        args.node_pool.resolve(), start=args.output.resolve().parent
    )

    dataset = []
    for query_index, source in enumerate(load_jsonl(args.input, args.limit)):
        solution, answer = split_reference(source["answer"])
        graphs = []
        for graph_index, topology in enumerate(topologies):
            graph = topology.to_graph_record()
            graph["id"] = f"q{query_index:04d}_wf{graph_index:02d}"
            graphs.append(graph)
        metadata = {
            key: value
            for key, value in source.items()
            if key not in ("question", "answer")
        }
        dataset.append(
            {
                "task": str(source["question"]),
                "reference_answer": answer,
                "reference_solution": solution,
                "source_metadata": metadata,
                "sampling_seed": args.seed + query_index,
                "node_pool": node_pool_reference,
                "evaluator": "math",
                "topology_policy": {
                    "family": "aime_workflow_mother_graphs_v1",
                    "semantically_constrained": True,
                    "independent_solver_isolation": True,
                    "per_role_token_budgets": True
                },
                "graphs": graphs,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(args.output)
    print(
        f"queries={len(dataset)} graphs={sum(len(x['graphs']) for x in dataset)} "
        f"families={[item.generator for item in topologies]} output={args.output}"
    )


if __name__ == "__main__":
    main()
