from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from MAS_DAG import DEFAULT_GSM8K_ROLES, generate_candidate_suite


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "gsm8k" / "sample.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "gsm8k" / "candidate_graphs.json"
DEFAULT_NODE_POOL = ROOT / "data" / "node_pools" / "gsm8k_6_roles.json"
NODE_POOL_ID = "gsm8k_6_roles_v1"

ROLE_BRIEFS = {
    "problem_parser": (
        "Extract the quantities, relationships, constraints, units, and exact "
        "question from the word problem without guessing the final answer."
    ),
    "cot_solver": (
        "Solve the math word problem independently with concise, step-by-step "
        "arithmetic reasoning and provide a final numeric answer."
    ),
    "equation_solver": (
        "Translate the problem into variables and equations, solve them "
        "symbolically, and provide a final numeric answer."
    ),
    "python_calculator": (
        "Turn the required calculations into restricted Python code, execute it "
        "to verify the arithmetic, and report the numeric result."
    ),
    "critic": (
        "Inspect available solutions for interpretation, logic, arithmetic, and "
        "unit errors, then state corrections and the verified numeric answer."
    ),
    "finalizer": (
        "Solve independently when no evidence is available; otherwise compare "
        "the available agent outputs, resolve disagreements, and return the final "
        "numeric answer."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign reproducible candidate DAGs to sampled GSM8K queries."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--node-pool", type=Path, default=DEFAULT_NODE_POOL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-count", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
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


def split_reference_answer(raw_answer: str) -> tuple[str, str]:
    if "####" not in raw_answer:
        raise ValueError("GSM8K answer does not contain the expected #### delimiter")
    solution, answer = raw_answer.rsplit("####", 1)
    return solution.strip(), answer.strip().replace(",", "")


def node_records() -> list[dict]:
    return [
        {
            "id": f"agent_{index}",
            "role": role,
            "role_brief": ROLE_BRIEFS[role],
        }
        for index, role in enumerate(DEFAULT_GSM8K_ROLES)
    ]


def build_dataset(
    source_records: list[dict],
    *,
    seed: int,
    random_count: int,
    node_pool_reference: str,
) -> list[dict]:
    nodes = node_records()
    output = []
    for query_index, source in enumerate(source_records):
        query_seed = seed + query_index
        solution, answer = split_reference_answer(str(source["answer"]))
        topologies = generate_candidate_suite(
            num_nodes=len(nodes),
            finalizer=len(nodes) - 1,
            random_count=random_count,
            seed=query_seed,
        )
        graphs = []
        for graph_index, topology in enumerate(topologies):
            graph = topology.to_graph_record()
            graph["id"] = f"q{query_index:04d}_g{graph_index:02d}"
            graphs.append(graph)
        output.append(
            {
                "task": str(source["question"]),
                "reference_answer": answer,
                "reference_solution": solution,
                "sampling_seed": query_seed,
                "node_pool": node_pool_reference,
                "graphs": graphs,
            }
        )
    return output


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input, limit=args.limit)
    node_pool_reference = os.path.relpath(
        args.node_pool.resolve(), start=args.output.resolve().parent
    )
    dataset = build_dataset(
        records,
        seed=args.seed,
        random_count=args.random_count,
        node_pool_reference=node_pool_reference,
    )

    args.node_pool.parent.mkdir(parents=True, exist_ok=True)
    node_pool = {
        "id": NODE_POOL_ID,
        "finalizer_id": "agent_5",
        "nodes": node_records(),
    }
    node_pool_temporary = args.node_pool.with_suffix(args.node_pool.suffix + ".tmp")
    with node_pool_temporary.open("w", encoding="utf-8") as handle:
        json.dump(node_pool, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    node_pool_temporary.replace(args.node_pool)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(args.output)
    graph_count = sum(len(record["graphs"]) for record in dataset)
    print(
        f"queries={len(dataset)} graphs={graph_count} output={args.output} "
        f"node_pool={args.node_pool}"
    )


if __name__ == "__main__":
    main()
