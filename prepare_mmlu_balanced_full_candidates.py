from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from MAS_DAG import generate_candidate_suite


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/training_set_inventory/mmlu_pro_balanced_difficulty_train_seed42/selected.json"
OUTPUT_ROOT = ROOT / "data/mmlu_pro_balanced_660"
POOL = ROOT / "data/node_pools/mmlu_pro_6_roles.json"
RUNTIME_ONLY = {
    "prediction", "node_outputs", "execution_error", "evaluation_error"
}
AVERAGED = (
    "total_input_tokens", "total_output_tokens", "wall_time_seconds",
    "node_input_token_cost", "node_output_token_cost", "node_time_cost",
    "edge_token_cost", "edge_time_cost",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def signature(graph: dict[str, Any]) -> tuple[Any, ...]:
    return (
        graph.get("generator"), graph.get("mask"), graph.get("edge_weight"),
        graph.get("topological_order"),
    )


def mean_nested(values: list[Any]) -> Any:
    mean = np.asarray(values, dtype=np.float64).mean(axis=0)
    return float(mean) if mean.ndim == 0 else mean.tolist()


def full_suite(record: dict[str, Any], query_index: int) -> list[dict[str, Any]]:
    pool = load(POOL)
    nodes = pool["nodes"]
    finalizer = next(i for i, node in enumerate(nodes) if node["id"] == pool["finalizer_id"])
    order = tuple(i for i in range(len(nodes)) if i != finalizer) + (finalizer,)
    suite = generate_candidate_suite(
        len(nodes), finalizer, random_count=5,
        seed=int(record["sampling_seed"]), fixed_order=order,
    )
    graphs = []
    question_id = record["source_metadata"]["question_id"]
    for graph_index, topology in enumerate(suite):
        graph = topology.to_graph_record()
        graph["id"] = f"mmlu_{question_id}_g{graph_index:02d}"
        graphs.append(graph)

    existing = {graph["generator"]: graph for graph in record["graphs"]}
    if signature(graphs[1])[:-1] != signature(existing["star"])[:-1]:
        raise ValueError(f"question {query_index}: star topology mismatch")
    if signature(graphs[5])[:-1] != signature(existing["finalizer_only"])[:-1]:
        raise ValueError(f"question {query_index}: finalizer topology mismatch")
    return graphs


def generate() -> None:
    records = load(SOURCE)
    if len(records) != 660:
        raise ValueError(f"expected 660 selected questions, got {len(records)}")
    pending = []
    for index, source in enumerate(records):
        record = {key: copy.deepcopy(value) for key, value in source.items() if key != "graphs"}
        record["node_pool"] = os.path.relpath(POOL, OUTPUT_ROOT)
        suite = full_suite(source, index)
        record["graphs"] = [graph for i, graph in enumerate(suite) if i not in (1, 5)]
        pending.append(record)
    shards = (pending[:330], pending[330:])
    for shard, values in enumerate(shards):
        write(OUTPUT_ROOT / f"candidates_shard{shard}.json", values)
    print("generated questions=660 shards=2 questions_per_shard=330 graphs_per_question=10")


def aggregate() -> None:
    original = load(SOURCE)
    final_records = []
    source_offset = 0
    for shard in range(2):
        runs = [load(OUTPUT_ROOT / f"runs/run{run}_shard{shard}.json") for run in range(1, 6)]
        if any(len(run) != 330 for run in runs):
            raise ValueError(f"shard {shard}: expected 330 records in every run")
        for local_index in range(330):
            source = original[source_offset + local_index]
            trials_by_signature: dict[str, list[dict[str, Any]]] = {}
            for run in runs:
                for graph in run[local_index]["graphs"]:
                    key = json.dumps(signature(graph), sort_keys=True)
                    trials_by_signature.setdefault(key, []).append(graph)
            suite = full_suite(source, source_offset + local_index)
            existing = {graph["generator"]: graph for graph in source["graphs"]}
            merged_graphs = []
            for graph in suite:
                if graph["generator"] in ("star", "finalizer_only"):
                    merged_graphs.append(copy.deepcopy(existing[graph["generator"]]))
                    continue
                key = json.dumps(signature(graph), sort_keys=True)
                trials = trials_by_signature.get(key, [])
                if len(trials) != 5 or any(t.get("execution_status") != "completed" for t in trials):
                    statuses = [t.get("execution_status") for t in trials]
                    raise ValueError(f"question {source_offset + local_index} graph {graph['id']}: {statuses}")
                accuracies = [float(t["accuracy"]) for t in trials]
                output = {k: copy.deepcopy(v) for k, v in trials[0].items() if k not in RUNTIME_ONLY}
                output.update(
                    reward=sum(accuracies) / 5,
                    accuracy=sum(accuracies) / 5,
                    success_count=int(sum(accuracies)),
                    num_trials=5,
                    reward_trials=accuracies,
                    execution_status="completed",
                )
                for field in AVERAGED:
                    if all(field in trial for trial in trials):
                        output[field] = mean_nested([trial[field] for trial in trials])
                merged_graphs.append(output)
            record = {key: copy.deepcopy(value) for key, value in source.items() if key != "graphs"}
            record["node_pool"] = "../node_pools/mmlu_pro_6_roles.json"
            record["graphs"] = merged_graphs
            record["reward_aggregation"] = {"method": "mean_accuracy", "num_trials": 5}
            final_records.append(record)
        source_offset += 330
    write(OUTPUT_ROOT / "MMLU-Pro-balanced-660-full-candidates-mean5.json", final_records)
    print("aggregated questions=660 graphs=7920 trials=5")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("generate", "aggregate"), required=True)
    args = parser.parse_args()
    generate() if args.mode == "generate" else aggregate()


if __name__ == "__main__":
    main()
