from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from MAS_DAG import generate_candidate_suite


def json_default(value: object) -> object:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a stratified MMLU-Pro split with SingleAgent and star BestMAS."
    )
    parser.add_argument("--input", type=Path, default=Path("data/mmlu_pro/test.parquet"))
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("data/mmlu_pro/train70_seed42.jsonl"),
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=Path("data/mmlu_pro/train70_single_best_candidates.json"),
    )
    parser.add_argument(
        "--node-pool",
        type=Path,
        default=Path("data/node_pools/mmlu_pro_6_roles.json"),
    )
    parser.add_argument("--fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.fraction < 1:
        raise ValueError("fraction must be between zero and one")

    rows = args.input.parent.joinpath(args.input.name)
    records = pd.read_parquet(rows).to_dict(orient="records")
    by_category: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_category[str(record["category"])].append(record)

    categories = sorted(by_category)
    exact = {category: len(by_category[category]) * args.fraction for category in categories}
    allocations = {category: math.floor(exact[category]) for category in categories}
    target_size = round(len(records) * args.fraction)
    remainder_order = sorted(
        categories,
        key=lambda category: (-(exact[category] - allocations[category]), category),
    )
    for category in remainder_order[: target_size - sum(allocations.values())]:
        allocations[category] += 1

    rng = random.Random(args.seed)
    selected: list[dict] = []
    for category in categories:
        group = by_category[category]
        rng.shuffle(group)
        selected.extend(group[: allocations[category]])
    rng.shuffle(selected)

    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    with args.sample_output.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(
                json.dumps(record, ensure_ascii=False, default=json_default) + "\n"
            )

    pool = json.loads(args.node_pool.read_text(encoding="utf-8"))
    nodes = pool["nodes"]
    finalizer = next(
        index for index, node in enumerate(nodes) if node["id"] == pool["finalizer_id"]
    )
    fixed_order = tuple(index for index in range(len(nodes)) if index != finalizer) + (
        finalizer,
    )
    pool_reference = os.path.relpath(
        args.node_pool.resolve(), args.candidate_output.resolve().parent
    )
    candidates = []
    for query_index, source in enumerate(selected):
        suite = generate_candidate_suite(
            len(nodes),
            finalizer,
            random_count=5,
            seed=args.seed + query_index,
            fixed_order=fixed_order,
        )
        chosen = [suite[5], suite[1]]  # finalizer_only, star
        graphs = []
        for label, topology in zip(("single", "best_mas"), chosen, strict=True):
            graph = topology.to_graph_record()
            graph["id"] = f"q{query_index:05d}_{label}"
            graphs.append(graph)
        options = "\n".join(
            f"({chr(65 + index)}) {option}"
            for index, option in enumerate(source["options"])
        )
        candidates.append(
            {
                "task": f"{source['question']}\n\nOptions:\n{options}",
                "reference_answer": source["answer"],
                "reference_solution": source.get("cot_content", ""),
                "source_metadata": {
                    key: source[key]
                    for key in ("question_id", "category", "src", "answer_index")
                },
                "sampling_seed": args.seed + query_index,
                "node_pool": pool_reference,
                "evaluator": "mmlu_pro",
                "screening_graphs": {
                    "single_agent": "finalizer_only",
                    "best_mas": "star",
                },
                "graphs": graphs,
            }
        )

    args.candidate_output.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(
        f"source={len(records)} selected={len(selected)} graphs={2 * len(selected)} "
        f"seed={args.seed} allocations={allocations}"
    )


if __name__ == "__main__":
    main()
