from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def signature(graph: dict[str, Any]) -> str:
    return json.dumps(
        (
            graph.get("id"),
            graph.get("generator"),
            graph.get("mask"),
            graph.get("edge_weight"),
        ),
        sort_keys=True,
        separators=(",", ":"),
    )


def prepare_runs(candidates: list[dict[str, Any]], runs: int) -> None:
    for run in range(1, runs + 1):
        screening_path = ROOT / f"data/aime/single_best_screening_run{run}.json"
        output_path = ROOT / f"data/aime/full_candidate_mean5_run{run}.json"
        if output_path.exists():
            existing = read(output_path)
            if len(existing) != len(candidates) or any(
                len(old["graphs"]) != len(new["graphs"])
                or old["task"] != new["task"]
                or any(
                    signature(old_graph) != signature(new_graph)
                    for old_graph, new_graph in zip(old["graphs"], new["graphs"])
                )
                for old, new in zip(existing, candidates)
            ):
                raise ValueError(f"existing output is incompatible: {output_path}")
            completed = sum(
                graph.get("execution_status") == "completed"
                for record in existing
                for graph in record["graphs"]
            )
            print(
                f"preserved={output_path.relative_to(ROOT)} completed={completed}"
            )
            continue
        screening = read(screening_path)
        if len(screening) != len(candidates):
            raise ValueError(f"run {run}: screening question count differs")

        output = copy.deepcopy(candidates)
        reused = 0
        for question_index, (target, source) in enumerate(zip(output, screening)):
            if target["task"] != source["task"]:
                raise ValueError(f"run {run}, question {question_index}: task mismatch")
            by_signature = {signature(graph): graph for graph in target["graphs"]}
            for completed in source["graphs"]:
                key = signature(completed)
                if key not in by_signature:
                    raise ValueError(
                        f"run {run}, question {question_index}: screening graph mismatch"
                    )
                by_signature[key].clear()
                by_signature[key].update(copy.deepcopy(completed))
                reused += 1
        if reused != len(candidates) * 2:
            raise ValueError(f"run {run}: expected to reuse 40 graphs, got {reused}")
        write(output_path, output)
        print(f"prepared={output_path.relative_to(ROOT)} reused={reused}")


def aggregate(candidates: list[dict[str, Any]], runs: int) -> None:
    sources = [
        read(ROOT / f"data/aime/full_candidate_mean5_run{run}.json")
        for run in range(1, runs + 1)
    ]
    result: list[dict[str, Any]] = []
    for question_index, base in enumerate(candidates):
        record = {key: copy.deepcopy(value) for key, value in base.items() if key != "graphs"}
        graphs = []
        for graph_index, candidate in enumerate(base["graphs"]):
            trials = [source[question_index]["graphs"][graph_index] for source in sources]
            if any(signature(trial) != signature(candidate) for trial in trials):
                raise ValueError(
                    f"question {question_index}, graph {graph_index}: graph mismatch"
                )
            if any(trial.get("execution_status") != "completed" for trial in trials):
                statuses = [trial.get("execution_status") for trial in trials]
                raise ValueError(
                    f"question {question_index}, graph {graph_index}: incomplete {statuses}"
                )
            graph = copy.deepcopy(candidate)
            rewards = [float(trial["accuracy"]) for trial in trials]
            graph.update(
                execution_status="completed",
                reward=sum(rewards) / runs,
                accuracy=sum(rewards) / runs,
                success_count=int(sum(rewards)),
                num_trials=runs,
                reward_trials=rewards,
            )
            for field in (
                "total_input_tokens",
                "total_output_tokens",
                "wall_time_seconds",
            ):
                values = [float(trial[field]) for trial in trials]
                graph[f"mean_{field}"] = sum(values) / runs
            graphs.append(graph)
        record["graphs"] = graphs
        result.append(record)

    output_path = ROOT / "data/aime/AIME-full-candidates-mean5.json"
    write(output_path, result)
    print(
        f"aggregated={output_path.relative_to(ROOT)} questions={len(result)} "
        f"graphs={sum(len(record['graphs']) for record in result)} trials={runs}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "aggregate"), required=True)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    candidates = read(ROOT / "data/aime/candidate_graphs.json")
    if args.mode == "prepare":
        prepare_runs(candidates, args.runs)
    else:
        aggregate(candidates, args.runs)


if __name__ == "__main__":
    main()
