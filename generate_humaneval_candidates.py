from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from MAS_DAG import generate_candidate_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 HumanEval sample JSONL 转换为分组候选 DAG。"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/humaneval/sample.jsonl")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/humaneval/candidate_graphs.json"),
    )
    parser.add_argument(
        "--node-pool", type=Path,
        default=Path("data/node_pools/humaneval_6_roles.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-count", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_sample(path: Path, limit: int | None) -> list[dict]:
    required = {"task_id", "prompt", "canonical_solution", "test", "entry_point"}
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = required - record.keys()
            if missing:
                raise ValueError(
                    f"{path}:{line_number} missing fields: "
                    f"{', '.join(sorted(missing))}"
                )
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def resolve_pool(pool: object, path: Path) -> tuple[list[dict], int]:
    if not isinstance(pool, dict) or not isinstance(pool.get("nodes"), list):
        raise ValueError(f"node pool must contain a nodes list: {path}")
    nodes = pool["nodes"]
    finalizers = [
        index for index, node in enumerate(nodes)
        if str(node.get("id")) == str(pool.get("finalizer_id"))
    ]
    if len(finalizers) != 1:
        raise ValueError("node pool finalizer_id must identify exactly one node")
    return nodes, finalizers[0]


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.random_count < 0:
        raise ValueError("random-count must be non-negative")

    source_records = load_sample(args.input, args.limit)
    nodes, finalizer = resolve_pool(load_json(args.node_pool), args.node_pool)
    fixed_order = tuple(index for index in range(len(nodes)) if index != finalizer) + (
        finalizer,
    )
    pool_reference = os.path.relpath(
        args.node_pool.resolve(), start=args.output.resolve().parent
    )

    dataset = []
    for query_index, source in enumerate(source_records):
        query_seed = args.seed + query_index
        graphs = []
        for graph_index, topology in enumerate(generate_candidate_suite(
            num_nodes=len(nodes), finalizer=finalizer,
            random_count=args.random_count, seed=query_seed,
            fixed_order=fixed_order,
        )):
            graph = topology.to_graph_record()
            graph["id"] = f"q{query_index:04d}_g{graph_index:02d}"
            graphs.append(graph)
        dataset.append({
            "task": str(source["prompt"]),
            "reference_answer": str(source["canonical_solution"]),
            "source_metadata": {
                "task_id": source["task_id"],
                "prompt": source["prompt"],
                "test": source["test"],
                "entry_point": source["entry_point"],
            },
            "sampling_seed": query_seed,
            "node_pool": pool_reference,
            "evaluator": "humaneval",
            "topology_policy": {
                "fixed_order": list(fixed_order),
                "anchors_shared_across_tasks": True,
            },
            "graphs": graphs,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(args.output)
    print(
        f"queries={len(dataset)} "
        f"graphs={sum(len(record['graphs']) for record in dataset)} "
        f"evaluator=humaneval node_pool={args.node_pool} output={args.output}"
    )


if __name__ == "__main__":
    main()
