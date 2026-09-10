from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


AVERAGED_SCALARS = (
    "total_input_tokens",
    "total_output_tokens",
    "wall_time_seconds",
)
AVERAGED_ARRAYS = (
    "edge_token_cost",
    "edge_time_cost",
    "node_input_token_cost",
    "node_output_token_cost",
    "node_time_cost",
)
RUNTIME_ONLY = {
    "answer_evaluator",
    "evaluation_error",
    "execution_error",
    "node_finish_reason",
    "node_outputs",
    "prediction",
}


def stable_signature(graph: dict[str, Any]) -> tuple[Any, ...]:
    return (
        graph.get("id"),
        graph.get("generator"),
        graph.get("mask"),
        graph.get("edge_weight"),
        graph.get("topological_order"),
    )


def mean_nested(values: list[Any]) -> Any:
    array = np.asarray(values, dtype=np.float64)
    mean = array.mean(axis=0)
    return float(mean) if mean.ndim == 0 else mean.tolist()


def merge_runs(runs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not runs or any(len(run) != len(runs[0]) for run in runs):
        raise ValueError("all runs must contain the same number of questions")
    merged_records = []
    for question_index, base_record in enumerate(runs[0]):
        records = [run[question_index] for run in runs]
        if any(record.get("task") != base_record.get("task") for record in records):
            raise ValueError(f"question {question_index}: task mismatch across runs")
        if any(len(record["graphs"]) != len(base_record["graphs"]) for record in records):
            raise ValueError(f"question {question_index}: graph count mismatch")

        merged = copy.deepcopy(base_record)
        merged_graphs = []
        for graph_index, base_graph in enumerate(base_record["graphs"]):
            graphs = [record["graphs"][graph_index] for record in records]
            if any(stable_signature(graph) != stable_signature(base_graph) for graph in graphs):
                raise ValueError(
                    f"question {question_index} graph {graph_index}: topology mismatch"
                )
            if any(graph.get("execution_status") != "completed" for graph in graphs):
                raise ValueError(
                    f"question {question_index} graph {graph_index}: incomplete trial"
                )
            accuracies = [float(graph["accuracy"]) for graph in graphs]
            output = {
                key: copy.deepcopy(value)
                for key, value in base_graph.items()
                if key not in RUNTIME_ONLY
            }
            output["reward"] = sum(accuracies) / len(accuracies)
            output["accuracy"] = output["reward"]
            output["success_count"] = int(sum(accuracies))
            output["num_trials"] = len(accuracies)
            output["reward_trials"] = accuracies
            output["execution_status"] = "completed"
            for key in AVERAGED_SCALARS + AVERAGED_ARRAYS:
                if all(key in graph for graph in graphs):
                    output[key] = mean_nested([graph[key] for graph in graphs])
            merged_graphs.append(output)
        merged["graphs"] = merged_graphs
        merged["reward_aggregation"] = {
            "method": "mean_accuracy",
            "num_trials": len(runs),
        }
        merged_records.append(merged)
    return merged_records


def stratum(record: dict[str, Any]) -> tuple[int, int]:
    rewards = [float(graph["reward"]) for graph in record["graphs"]]
    mean = sum(rewards) / len(rewards)
    spread = max(rewards) - min(rewards)
    difficulty_bin = 0 if mean < 1 / 3 else 1 if mean < 2 / 3 else 2
    spread_bin = 0 if spread <= 0.4 else 1
    return difficulty_bin, spread_bin


def stable_rank(record: dict[str, Any], seed: int) -> str:
    payload = f"{seed}\0{record['task']}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_records(
    records: list[dict[str, Any]], validation_size: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_size < len(records):
        raise ValueError("validation_size must be between zero and dataset size")
    groups: dict[tuple[int, int], list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(stratum(record), []).append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: stable_rank(records[index], seed))

    exact = {key: validation_size * len(indices) / len(records) for key, indices in groups.items()}
    allocations = {key: int(value) for key, value in exact.items()}
    remaining = validation_size - sum(allocations.values())
    order = sorted(groups, key=lambda key: (exact[key] - allocations[key], len(groups[key])), reverse=True)
    for key in order[:remaining]:
        allocations[key] += 1

    validation_indices = {
        index
        for key, indices in groups.items()
        for index in indices[: allocations[key]]
    }
    train = [record for index, record in enumerate(records) if index not in validation_indices]
    validation = [record for index, record in enumerate(records) if index in validation_indices]
    return train, validation


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge GPQA repeated runs into mean rewards.")
    parser.add_argument(
        "--pattern",
        default="data/gpqa/train_candidate_graphs_scored_with_outputs_v{version}.json",
    )
    parser.add_argument("--versions", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--output", type=Path, default=Path("data/gpqa/GPQA-train-mean5.json"))
    parser.add_argument("--train-output", type=Path, default=Path("data/gpqa/GPQA-train-mean5-train80.json"))
    parser.add_argument("--validation-output", type=Path, default=Path("data/gpqa/GPQA-train-mean5-validation20.json"))
    parser.add_argument("--validation-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    runs = [
        json.loads(Path(args.pattern.format(version=version)).read_text(encoding="utf-8"))
        for version in args.versions
    ]
    merged = merge_runs(runs)
    train, validation = split_records(merged, args.validation_size, args.seed)
    write_json(args.output, merged)
    write_json(args.train_output, train)
    write_json(args.validation_output, validation)
    print(
        f"merged={len(merged)} train={len(train)} validation={len(validation)} "
        f"trials={len(runs)}"
    )


if __name__ == "__main__":
    main()
