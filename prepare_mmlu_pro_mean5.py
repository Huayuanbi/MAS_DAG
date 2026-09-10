from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


AVERAGED_FIELDS = (
    "total_input_tokens",
    "total_output_tokens",
    "wall_time_seconds",
    "edge_token_cost",
    "edge_time_cost",
    "node_input_token_cost",
    "node_output_token_cost",
    "node_time_cost",
)
RUNTIME_ONLY = {
    "accuracy",
    "answer_evaluator",
    "evaluation_error",
    "execution_error",
    "execution_status",
    "node_finish_reason",
    "node_outputs",
    "prediction",
    "reward",
}


def signature(graph: dict[str, Any]) -> tuple[Any, ...]:
    return (
        graph.get("id"),
        graph.get("generator"),
        graph.get("mask"),
        graph.get("edge_weight"),
        graph.get("topological_order"),
    )


def mean_nested(values: list[Any]) -> Any:
    result = np.asarray(values, dtype=np.float64).mean(axis=0)
    return float(result) if result.ndim == 0 else result.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge sharded MMLU-Pro runs and preserve failed trials as missing."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/mmlu_pro/train70_single_best_candidates.json"),
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("data/mmlu_pro/train70_mean5_shards"),
    )
    parser.add_argument(
        "--output-with-missing",
        type=Path,
        default=Path("data/mmlu_pro/train70_single_best_mean5_with_missing.json"),
    )
    parser.add_argument(
        "--output-clean",
        type=Path,
        default=Path("data/mmlu_pro/train70_single_best_mean5_clean.json"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/mmlu_pro/train70_single_best_mean5_summary.json"),
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()

    base = json.loads(args.input.read_text(encoding="utf-8"))
    total = len(base)
    shard_size = (total + args.shards - 1) // args.shards
    trials: list[list[list[dict[str, Any] | None]]] = [
        [[None for _ in range(args.runs)] for _ in record["graphs"]]
        for record in base
    ]

    for run in range(1, args.runs + 1):
        for shard in range(args.shards):
            path = args.shard_dir / f"run{run}_shard{shard}.json"
            records = json.loads(path.read_text(encoding="utf-8"))
            if len(records) != total:
                raise ValueError(f"{path}: expected {total} questions, got {len(records)}")
            start = shard * shard_size
            stop = min(total, start + shard_size)
            for question_index in range(start, stop):
                source = records[question_index]
                expected = base[question_index]
                if source.get("task") != expected.get("task"):
                    raise ValueError(f"{path}: task mismatch at question {question_index}")
                for graph_index, graph in enumerate(source["graphs"]):
                    if signature(graph) != signature(expected["graphs"][graph_index]):
                        raise ValueError(
                            f"{path}: graph mismatch at question {question_index}, graph {graph_index}"
                        )
                    trials[question_index][graph_index][run - 1] = copy.deepcopy(graph)

    merged = copy.deepcopy(base)
    missing_by_run: Counter[int] = Counter()
    missing_by_generator: Counter[str] = Counter()
    missing_by_category: Counter[str] = Counter()
    incomplete_question_ids: set[int] = set()

    for question_index, record in enumerate(merged):
        category = str(record["source_metadata"]["category"])
        question_id = int(record["source_metadata"]["question_id"])
        outputs = []
        for graph_index, base_graph in enumerate(record["graphs"]):
            graph_trials = trials[question_index][graph_index]
            if any(graph is None for graph in graph_trials):
                raise ValueError(f"unassigned trial at question {question_index}, graph {graph_index}")
            concrete = [graph for graph in graph_trials if graph is not None]
            reward_trials: list[float | None] = []
            missing_trials = []
            completed = []
            for run_index, graph in enumerate(concrete, start=1):
                if graph.get("execution_status") == "completed":
                    reward_trials.append(float(graph["accuracy"]))
                    completed.append(graph)
                else:
                    reward_trials.append(None)
                    missing_trials.append(
                        {
                            "run": run_index,
                            "status": "missing_context_overflow",
                            "original_execution_status": graph.get("execution_status"),
                            "reason": graph.get("execution_error", "unknown execution failure"),
                        }
                    )
                    missing_by_run[run_index] += 1
                    missing_by_generator[str(base_graph.get("generator"))] += 1
                    missing_by_category[category] += 1
                    incomplete_question_ids.add(question_id)

            output = {
                key: copy.deepcopy(value)
                for key, value in base_graph.items()
                if key not in RUNTIME_ONLY
            }
            output["num_trials"] = args.runs
            output["num_observed_trials"] = len(completed)
            output["missing_trial_count"] = len(missing_trials)
            output["reward_trials"] = reward_trials
            output["missing_trials"] = missing_trials
            if len(completed) == args.runs:
                accuracies = [float(graph["accuracy"]) for graph in completed]
                output["reward"] = sum(accuracies) / args.runs
                output["accuracy"] = output["reward"]
                output["success_count"] = int(sum(accuracies))
                output["execution_status"] = "completed"
                for field in AVERAGED_FIELDS:
                    if all(field in graph for graph in completed):
                        output[field] = mean_nested([graph[field] for graph in completed])
            else:
                output["reward"] = None
                output["accuracy"] = None
                output["success_count"] = None
                output["execution_status"] = "missing_trials"
                if completed:
                    output["observed_reward"] = sum(
                        float(graph["accuracy"]) for graph in completed
                    ) / len(completed)
            outputs.append(output)
        record["graphs"] = outputs
        record["reward_aggregation"] = {
            "method": "mean_accuracy",
            "required_trials": args.runs,
            "missing_policy": "do_not_impute; exclude incomplete question from clean set",
        }

    clean = [
        record
        for record in merged
        if all(graph["execution_status"] == "completed" for graph in record["graphs"])
    ]
    delta_distribution: Counter[str] = Counter()
    relation_distribution: Counter[str] = Counter()
    category_stats: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"single": [], "best_mas": [], "delta": []}
    )
    for record in clean:
        single, best = record["graphs"]
        ps, pm = float(single["reward"]), float(best["reward"])
        delta = round(pm - ps, 10)
        delta_distribution[f"{delta:.1f}"] += 1
        relation_distribution[
            "best_mas_better" if delta > 0 else "single_better" if delta < 0 else "tie"
        ] += 1
        stats = category_stats[str(record["source_metadata"]["category"])]
        stats["single"].append(ps)
        stats["best_mas"].append(pm)
        stats["delta"].append(delta)

    summary = {
        "source_questions": total,
        "clean_questions": len(clean),
        "excluded_questions": total - len(clean),
        "total_expected_trials": total * 2 * args.runs,
        "missing_trials": sum(missing_by_run.values()),
        "missing_by_run": dict(sorted(missing_by_run.items())),
        "missing_by_generator": dict(sorted(missing_by_generator.items())),
        "missing_by_category": dict(sorted(missing_by_category.items())),
        "incomplete_question_ids": sorted(incomplete_question_ids),
        "delta_distribution": dict(sorted(delta_distribution.items(), key=lambda item: float(item[0]))),
        "relation_distribution": dict(relation_distribution),
        "overall": {
            "single_mean": float(np.mean([r["graphs"][0]["reward"] for r in clean])),
            "best_mas_mean": float(np.mean([r["graphs"][1]["reward"] for r in clean])),
        },
        "by_category": {
            category: {
                "questions": len(values["single"]),
                "single_mean": float(np.mean(values["single"])),
                "best_mas_mean": float(np.mean(values["best_mas"])),
                "delta_mean": float(np.mean(values["delta"])),
            }
            for category, values in sorted(category_stats.items())
        },
    }

    for path, value in (
        (args.output_with_missing, merged),
        (args.output_clean, clean),
        (args.summary, summary),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
