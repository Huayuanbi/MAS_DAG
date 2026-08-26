from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from MAS_DAG import generate_candidate_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 sample JSONL 和角色池生成固定语义顺序的候选 DAG。"
    )
    parser.add_argument("--input", type=Path, required=True, help="sample JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-pool", type=Path, required=True)
    parser.add_argument(
        "--evaluator", choices=("math", "gsm8k"), required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-count", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path, limit: int | None) -> list[dict]:
    records: list[dict] = []
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


def split_reference(raw_answer: object, evaluator: str) -> tuple[str, str]:
    value = str(raw_answer)
    if "####" not in value:
        raise ValueError("sample answer must contain the #### delimiter")
    solution, answer = value.rsplit("####", 1)
    answer = answer.strip()
    if evaluator == "gsm8k":
        answer = answer.replace(",", "")
    return solution.strip(), answer


def resolve_pool(pool: object, path: Path) -> tuple[list[dict], int]:
    if not isinstance(pool, dict) or not isinstance(pool.get("nodes"), list):
        raise ValueError(f"node pool must contain a nodes list: {path}")
    nodes = pool["nodes"]
    finalizers = [
        index
        for index, node in enumerate(nodes)
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

    source_records = load_jsonl(args.input, args.limit)
    nodes, finalizer = resolve_pool(load_json(args.node_pool), args.node_pool)
    fixed_order = tuple(index for index in range(len(nodes)) if index != finalizer) + (
        finalizer,
    )
    node_pool_reference = os.path.relpath(
        args.node_pool.resolve(), start=args.output.resolve().parent
    )

    dataset: list[dict] = []
    for query_index, source in enumerate(source_records):
        query_seed = args.seed + query_index
        solution, answer = split_reference(source["answer"], args.evaluator)
        topologies = generate_candidate_suite(
            num_nodes=len(nodes),
            finalizer=finalizer,
            random_count=args.random_count,
            seed=query_seed,
            fixed_order=fixed_order,
        )
        graphs = []
        for graph_index, topology in enumerate(topologies):
            graph = topology.to_graph_record()
            graph["id"] = f"q{query_index:04d}_g{graph_index:02d}"
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
                "sampling_seed": query_seed,
                "node_pool": node_pool_reference,
                "evaluator": args.evaluator,
                "topology_policy": {
                    "fixed_order": list(fixed_order),
                    "anchors_shared_across_tasks": True,
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
        f"queries={len(dataset)} "
        f"graphs={sum(len(record['graphs']) for record in dataset)} "
        f"evaluator={args.evaluator} node_pool={args.node_pool} "
        f"fixed_order={fixed_order} output={args.output}"
    )


if __name__ == "__main__":
    main()
